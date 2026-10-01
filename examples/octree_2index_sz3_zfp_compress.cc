// ============================================================================
// Octree 索引压缩对比实验：两棵交错 octree 的叶子 low/high 改用 SZ3 或 ZFP 一维压缩
// 输入参数与原 octree 索引程序一致，新增 --compressor sz3|zfp
// ============================================================================
#include <vector>
#include <queue>
#include <set>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>
#include <utility>
#include <mpi.h>
#include <zstd.h>
#include <adios2.h>
#include <scidx_avl.h>
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <scidx_octree_interval.h>   // OctreeNode, buildOctree
#include <SZ3/api/sz.hpp>
#include <zfp.h>

// ------------------------------- 工具函数 ------------------------------------

std::vector<size_t> positionToIndices(size_t position, const std::vector<size_t>& shape)
{
    std::vector<size_t> indices;
    size_t remainingPosition = position;
    for (auto dimensionSize = shape.rbegin(); dimensionSize != shape.rend(); ++dimensionSize)
    {
        indices.insert(indices.begin(), remainingPosition % *dimensionSize);
        remainingPosition /= *dimensionSize;
    }
    return indices;
}

size_t indicesToPosition(const std::vector<size_t>& shape, const std::vector<size_t>& indices)
{
    size_t position = indices[0];
    size_t multiplier = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        multiplier *= shape[i - 1];
        position += indices[i] * multiplier;
    }
    return position;
}

void saveBigBlockMinMax(const std::string& filename,
                        const std::vector<std::vector<std::vector<ScidxInterval<double>>>>& all_steps)
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    for (size_t step = 0; step < all_steps.size(); ++step) {
        for (size_t big = 0; big < all_steps[step].size(); ++big) {
            double minVal = std::numeric_limits<double>::max();
            double maxVal = std::numeric_limits<double>::lowest();
            for (const auto& interval : all_steps[step][big]) {
                minVal = std::min(minVal, interval.low);
                maxVal = std::max(maxVal, interval.high);
            }
            outFile.write(reinterpret_cast<const char*>(&minVal), sizeof(double));
            outFile.write(reinterpret_cast<const char*>(&maxVal), sizeof(double));
        }
    }
}

// ------------------------- Skilling 三维 Hilbert 编码 ------------------------
uint64_t hilbertIndex3D(uint32_t x, uint32_t y, uint32_t z, int bits)
{
    uint32_t X[3] = {x, y, z};
    uint32_t M = 1u << (bits - 1), P, Q, t;

    for (Q = M; Q > 1; Q >>= 1) {
        P = Q - 1;
        for (int i = 0; i < 3; i++) {
            if (X[i] & Q) X[0] ^= P;
            else { t = (X[0] ^ X[i]) & P; X[0] ^= t; X[i] ^= t; }
        }
    }
    for (int i = 1; i < 3; i++) X[i] ^= X[i - 1];
    t = 0;
    for (Q = M; Q > 1; Q >>= 1)
        if (X[2] & Q) t ^= Q - 1;
    for (int i = 0; i < 3; i++) X[i] ^= t;

    uint64_t h = 0;
    for (int b = bits - 1; b >= 0; b--)
        for (int i = 0; i < 3; i++)
            h = (h << 1) | ((X[i] >> b) & 1u);
    return h;
}

// 返回按 Hilbert 顺序排列的块位置（块位置按 indicesToPosition：第 0 维变化最快）
std::vector<size_t> hilbertOrder(const std::vector<size_t>& blockCount)
{
    size_t n0 = blockCount[0], n1 = blockCount[1], n2 = blockCount[2];
    size_t total = n0 * n1 * n2;
    size_t maxDim = std::max({n0, n1, n2});
    int bits = 1;
    while ((size_t(1) << bits) < maxDim) bits++;

    std::vector<uint64_t> keys(total);
    for (size_t k = 0; k < n2; k++)
        for (size_t j = 0; j < n1; j++)
            for (size_t i = 0; i < n0; i++)
                keys[i + j * n0 + k * n0 * n1] = hilbertIndex3D((uint32_t)i, (uint32_t)j, (uint32_t)k, bits);

    std::vector<size_t> order(total);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&keys](size_t a, size_t b) { return keys[a] < keys[b]; });
    return order;
}

// ------------------------------ SZ3 一维压缩 ---------------------------------
size_t compress1D_SZ3(const std::vector<double>& data, double eb, const std::string& outFile)
{
    if (data.empty()) return 0;

    SZ3::Config conf(data.size());
    conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;   // SZ3 默认算法
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = eb;

    std::vector<double> copy(data);
    size_t cmpSize = 0;
    char* cmpData = SZ_compress(conf, copy.data(), cmpSize);

    std::ofstream out(outFile, std::ios::binary);
    out.write(cmpData, cmpSize);
    delete[] cmpData;
    return cmpSize;
}

// ------------------------------ ZFP 一维压缩 ---------------------------------
size_t compress1D_ZFP(const std::vector<double>& data, double eb, const std::string& outFile)
{
    if (data.empty()) return 0;

    std::vector<double> copy(data);
    zfp_field* field = zfp_field_1d(copy.data(), zfp_type_double, copy.size());
    zfp_stream* zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, eb);

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    std::vector<unsigned char> buffer(bufsize);
    bitstream* stream = stream_open(buffer.data(), bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    size_t zfpSize = zfp_compress(zfp, field);
    if (zfpSize == 0)
        std::cerr << "ZFP compression failed: " << outFile << std::endl;

    std::ofstream out(outFile, std::ios::binary);
    out.write(reinterpret_cast<char*>(buffer.data()), zfpSize);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    return zfpSize;
}

// ------------------ 建一棵 octree，并用 SZ3/ZFP 压缩叶子 low/high -------------
struct TreeSizes {
    size_t numLeaves = 0;
    size_t rawBytes = 0;     // low + high 原始大小
    size_t cmpBytes = 0;     // low + high 压缩后大小
};

TreeSizes buildAndCompressOctree(const std::vector<ScidxInterval<double>>& intervals,
                                 const std::vector<size_t>& blockCount,
                                 bool useHilbert,
                                 const std::string& treeID,
                                 const std::string& compressor,
                                 double error_bound,
                                 const std::string& tag,
                                 int mpi_rank)
{
    TreeSizes sz;
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<double> blockMins(intervals.size()), blockMaxs(intervals.size());
    for (size_t i = 0; i < intervals.size(); i++) {
        blockMins[i] = intervals[i].low;
        blockMaxs[i] = intervals[i].high;
    }

    // 与原程序一致地建 octree（内部节点由叶子推出，压缩只需叶子）
    std::vector<OctreeNode<double>> octree = buildOctree(blockMins, blockMaxs, blockCount);

    auto t2 = std::chrono::high_resolution_clock::now();

    // 叶子顺序：UNIFORM 与原方法一致按 Hilbert 重排，STAGGER 保持原始块顺序
    std::vector<double> lows, highs;
    if (useHilbert) {
        std::vector<size_t> order = hilbertOrder(blockCount);
        lows.reserve(order.size());
        highs.reserve(order.size());
        for (size_t p : order) {
            lows.push_back(blockMins[p]);
            highs.push_back(blockMaxs[p]);
        }
    } else {
        lows = blockMins;
        highs = blockMaxs;
    }

    sz.numLeaves = lows.size();
    sz.rawBytes  = (lows.size() + highs.size()) * sizeof(double);

    if (compressor == "sz3") {
        sz.cmpBytes = compress1D_SZ3(lows,  error_bound, treeID + "-sz3_low")
                    + compress1D_SZ3(highs, error_bound, treeID + "-sz3_high");
    } else {
        sz.cmpBytes = compress1D_ZFP(lows,  error_bound, treeID + "-zfp_low")
                    + compress1D_ZFP(highs, error_bound, treeID + "-zfp_high");
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "[Rank " << mpi_rank << "] [" << tag << "] build octree: "
              << std::chrono::duration<double>(t2 - t1).count() << " s, "
              << compressor << " compress + io: "
              << std::chrono::duration<double>(t3 - t2).count() << " s" << std::endl;

    std::cout << "[Rank " << mpi_rank << "] [" << tag << "] leaves = " << sz.numLeaves
              << ", order = " << (useHilbert ? "hilbert" : "native")
              << ", raw = " << sz.rawBytes << " B"
              << ", " << compressor << " = " << sz.cmpBytes << " B"
              << " (CR " << (sz.cmpBytes ? (double)sz.rawBytes / sz.cmpBytes : 0.0) << ")" << std::endl;

    return sz;
}

// ================================== main =====================================
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::string inputFileName;
    std::string variableName;
    std::string variableType;
    std::string compressor = "sz3";          // 新增：sz3 或 zfp
    size_t nDim = 0;

    size_t beginStepNum = 0;
    size_t endStepNum = 0;
    std::vector<size_t> targetSteps;

    std::vector<size_t> blockShape;
    std::vector<size_t> smallBlockShape;

    double relative_error_bound = 1E-3;
    size_t extraValue = 0;

    // ------------------------------ 参数解析 --------------------------------
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc)
            inputFileName = argv[++i];
        else if (arg == "--variable_name" && i + 1 < argc)
            variableName = argv[++i];
        else if (arg == "--dimensions" && i + 1 < argc)
            nDim = std::stoul(argv[++i]);
        else if (arg == "--compressor" && i + 1 < argc)
            compressor = argv[++i];
        else if (arg == "--block_shape" && i + 1 < argc) {
            for (size_t j = 0; j < nDim; j++)
                blockShape.push_back(std::stoul(argv[++i]));
        }
        else if (arg == "--begin_step") {
            if (i + 1 < argc) beginStepNum = atoi(argv[i + 1]);
            else { std::cerr << "--begin_step option requires one argument." << std::endl; return 1; }
        }
        else if (arg == "--end_step") {
            if (i + 1 < argc) endStepNum = atoi(argv[i + 1]);
            else { std::cerr << "--end_step option requires one argument." << std::endl; return 1; }
        }
        else if (arg == "--target_steps") {
            while (i + 1 < argc && std::isdigit(argv[i + 1][0]))
                targetSteps.push_back(std::stoul(argv[++i]));
        }
        else if (arg == "--small_block_shape") {
            if (nDim && (int)(i + nDim) < argc) {
                for (size_t j = i + 1; j < i + 1 + nDim; j++)
                    smallBlockShape.push_back(atoi(argv[j]));
            } else {
                std::cerr << "--small_block_shape option requires [# of dimensions] argument." << std::endl;
                return 1;
            }
        }
        else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        }
    }

    std::transform(compressor.begin(), compressor.end(), compressor.begin(), ::tolower);
    if (compressor != "sz3" && compressor != "zfp") {
        if (mpi_rank == 0) std::cerr << "Error: --compressor must be sz3 or zfp" << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (inputFileName.empty() || variableName.empty() || nDim != 3 || smallBlockShape.size() != nDim) {
        if (mpi_rank == 0) std::cerr << "Error: missing or invalid arguments (octree requires 3D)." << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::vector<size_t> halfBlockShape(nDim);
    for (size_t i = 0; i < nDim; i++)
        halfBlockShape[i] = smallBlockShape[i] / 2;

    // --------------------- 阶段 1、2：读数据、算两套区间 ----------------------
    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);

    std::vector<std::vector<std::vector<ScidxInterval<double>>>> all_steps_uniform;
    std::vector<std::vector<std::vector<ScidxInterval<double>>>> all_steps_stagger;

    size_t step = 0;
    std::set<size_t> targetSet(targetSteps.begin(), targetSteps.end());

    double total_adios_read_time = 0.0;
    double total_partition_time  = 0.0;

    std::vector<size_t> uniformBlockCountOnEachDim_global(nDim);
    std::vector<size_t> staggerBlockCountOnEachDim_global(nDim);
    bool dimsSaved = false;

    while (reader_engine.BeginStep() == adios2::StepStatus::OK)
    {
        if (!targetSet.empty() && targetSet.find(step) == targetSet.end()) {
            reader_engine.EndStep();
            step++;
            continue;
        }

        auto var = reader_io.InquireVariable<double>(variableName);
        variableType = reader_io.VariableType(variableName);

        size_t varElements = 1;
        for (size_t i = 0; i < nDim; i++)
            varElements *= var.Shape()[i];

        if (variableType == "double")
        {
            std::vector<double> varData(varElements);

            double t_adios_start = MPI_Wtime();
            reader_engine.Get(var, varData.data(), adios2::Mode::Sync);
            double t_adios_end = MPI_Wtime();
            total_adios_read_time += (t_adios_end - t_adios_start);

            double t_partition_start = MPI_Wtime();

            // 均匀分块
            std::vector<size_t> uniformBlockCountOnEachDim(nDim);
            size_t total_uniform_blocks = 1;
            for (size_t i = 0; i < nDim; i++) {
                uniformBlockCountOnEachDim[i] = (var.Shape()[i] + smallBlockShape[i] - 1) / smallBlockShape[i];
                total_uniform_blocks *= uniformBlockCountOnEachDim[i];
            }

            // 错位分块（与原程序一致：开头半块并入第一个整块）
            std::vector<size_t> staggerBlockCountOnEachDim(nDim);
            size_t total_stagger_blocks = 1;
            for (size_t i = 0; i < nDim; i++) {
                staggerBlockCountOnEachDim[i] = var.Shape()[i] / smallBlockShape[i];
                total_stagger_blocks *= staggerBlockCountOnEachDim[i];
            }

            if (!dimsSaved) {
                uniformBlockCountOnEachDim_global = uniformBlockCountOnEachDim;
                staggerBlockCountOnEachDim_global = staggerBlockCountOnEachDim;
                dimsSaved = true;
            }

            std::vector<double> uniform_mins(total_uniform_blocks, std::numeric_limits<double>::max());
            std::vector<double> uniform_maxs(total_uniform_blocks, std::numeric_limits<double>::lowest());
            std::vector<double> stagger_mins(total_stagger_blocks, std::numeric_limits<double>::max());
            std::vector<double> stagger_maxs(total_stagger_blocks, std::numeric_limits<double>::lowest());

            for (size_t p = 0; p < varElements; p++)
            {
                std::vector<size_t> global_id = positionToIndices(p, var.Shape());

                std::vector<size_t> uniform_block_id(nDim);
                for (size_t i = 0; i < nDim; i++)
                    uniform_block_id[i] = global_id[i] / smallBlockShape[i];
                size_t uniform_pos = indicesToPosition(uniformBlockCountOnEachDim, uniform_block_id);
                uniform_mins[uniform_pos] = std::min(uniform_mins[uniform_pos], varData[p]);
                uniform_maxs[uniform_pos] = std::max(uniform_maxs[uniform_pos], varData[p]);

                std::vector<size_t> stagger_block_id(nDim);
                for (size_t i = 0; i < nDim; i++) {
                    if (global_id[i] < halfBlockShape[i] + smallBlockShape[i])
                        stagger_block_id[i] = 0;
                    else
                        stagger_block_id[i] = (global_id[i] - halfBlockShape[i]) / smallBlockShape[i];
                }
                size_t stagger_pos = indicesToPosition(staggerBlockCountOnEachDim, stagger_block_id);
                stagger_mins[stagger_pos] = std::min(stagger_mins[stagger_pos], varData[p]);
                stagger_maxs[stagger_pos] = std::max(stagger_maxs[stagger_pos], varData[p]);
            }

            std::vector<ScidxInterval<double>> uniformIntervals(total_uniform_blocks);
            for (size_t b = 0; b < total_uniform_blocks; b++) {
                uniformIntervals[b].low  = uniform_mins[b];
                uniformIntervals[b].high = uniform_maxs[b];
            }
            std::vector<ScidxInterval<double>> staggerIntervals(total_stagger_blocks);
            for (size_t b = 0; b < total_stagger_blocks; b++) {
                staggerIntervals[b].low  = stagger_mins[b];
                staggerIntervals[b].high = stagger_maxs[b];
            }

            double t_partition_end = MPI_Wtime();
            total_partition_time += (t_partition_end - t_partition_start);

            all_steps_uniform.push_back({uniformIntervals});
            all_steps_stagger.push_back({staggerIntervals});
        }

        reader_engine.EndStep();
        step++;
    }
    reader_engine.Close();

    // ------------------------------ 输出目录 --------------------------------
    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    std::string rootDir = "/expanse/lustre/scratch/sdi/temp_project/octree_index_compression_" + compressor + "/";
    std::string caseName = inputFileBaseName + "_" + safeVarName + "_"
                         + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                         + std::to_string(extraValue);
    std::string subDir_uniform = rootDir + caseName + "_octree_stagger1_index/";
    std::string subDir_stagger = rootDir + caseName + "_octree_stagger2_index/";

    if (mpi_rank == 0) {
        std::filesystem::create_directories(subDir_uniform);
        std::filesystem::create_directories(subDir_stagger);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------ 误差界 ----------------------------------
    double global_min = std::numeric_limits<double>::max();
    double global_max = std::numeric_limits<double>::lowest();
    for (const auto* all : {&all_steps_uniform, &all_steps_stagger})
        for (const auto& step_blocks : *all)
            for (const auto& big_block : step_blocks)
                for (const auto& interval : big_block) {
                    global_min = std::min(global_min, interval.low);
                    global_max = std::max(global_max, interval.high);
                }

    double error_bound = relative_error_bound * (global_max - global_min);
    if (mpi_rank == 0) {
        std::cout << "compressor = " << compressor << std::endl;
        std::cout << "error_bound = " << error_bound << std::endl;
        saveBigBlockMinMax(subDir_uniform + "big_block_minmax", all_steps_uniform);
        saveBigBlockMinMax(subDir_stagger + "big_block_minmax", all_steps_stagger);
    }

    size_t numSteps = all_steps_uniform.size();

    // ---------------------- 阶段 3、4：两棵 octree 建树 + 压缩 -----------------
    double local_time[2] = {0.0, 0.0};
    size_t localSizes[2][2] = {{0}};   // [uniform/stagger][raw, cmp]

    for (int which = 0; which < 2; which++)
    {
        const auto& all_steps = (which == 0) ? all_steps_uniform : all_steps_stagger;
        const auto& blockCount = (which == 0) ? uniformBlockCountOnEachDim_global
                                              : staggerBlockCountOnEachDim_global;
        const std::string& subDir = (which == 0) ? subDir_uniform : subDir_stagger;
        const std::string tag = (which == 0) ? "UNIFORM" : "STAGGER";
        const bool useHilbert = (which == 0);   // 与原方法一致：T1 用 Hilbert 顺序，T2 用原始顺序

        if (mpi_rank == 0)
            std::cout << "\n=== Building " << tag << " Octree index (" << compressor << ") ===" << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        for (size_t s = 0; s < numSteps; ++s) {
            size_t actualStep = beginStepNum + s;
            size_t numBigBlocks = all_steps[s].size();

            for (size_t big = 0; big < numBigBlocks; ++big) {
                size_t task_id = s * numBigBlocks + big;
                if (task_id % mpi_size != (size_t)mpi_rank) continue;

                std::string treeID = subDir + std::to_string(actualStep) + "-" + std::to_string(big);
                TreeSizes sz = buildAndCompressOctree(all_steps[s][big], blockCount, useHilbert,
                                                      treeID, compressor, error_bound, tag, mpi_rank);
                localSizes[which][0] += sz.rawBytes;
                localSizes[which][1] += sz.cmpBytes;
            }
        }

        local_time[which] = MPI_Wtime() - t_start;
    }

    // ------------------------------ 汇总输出 --------------------------------
    double max_time[2] = {0.0, 0.0};
    MPI_Reduce(local_time, max_time, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    unsigned long long localU[4] = {localSizes[0][0], localSizes[0][1], localSizes[1][0], localSizes[1][1]};
    unsigned long long globalU[4] = {0, 0, 0, 0};
    MPI_Reduce(localU, globalU, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        auto cr = [](unsigned long long raw, unsigned long long cmp) {
            return cmp ? (double)raw / cmp : 0.0;
        };
        std::cout << "\n========== SUMMARY (octree, " << compressor << ") ==========" << std::endl;
        std::cout << "[Stage 1] ADIOS read:                      " << total_adios_read_time << " s" << std::endl;
        std::cout << "[Stage 2] Partition + minmax + interval:   " << total_partition_time  << " s" << std::endl;
        std::cout << "[Stage 3] UNIFORM Octree build + compress: " << max_time[0] << " s" << std::endl;
        std::cout << "[Stage 4] STAGGER Octree build + compress: " << max_time[1] << " s" << std::endl;
        std::cout << "[UNIFORM] raw = " << globalU[0] << " B, " << compressor << " = " << globalU[1]
                  << " B, CR = " << cr(globalU[0], globalU[1]) << std::endl;
        std::cout << "[STAGGER] raw = " << globalU[2] << " B, " << compressor << " = " << globalU[3]
                  << " B, CR = " << cr(globalU[2], globalU[3]) << std::endl;
        std::cout << "[BOTH]    raw = " << globalU[0] + globalU[2] << " B, " << compressor << " = "
                  << globalU[1] + globalU[3] << " B, CR = "
                  << cr(globalU[0] + globalU[2], globalU[1] + globalU[3]) << std::endl;
        std::cout << "==================================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
// ============================================================================
// 索引压缩对比实验：两套交错 AVL 索引的 low/high 改用 SZ3 或 ZFP 一维压缩
// 输入参数与原索引构建程序一致，新增 --compressor sz3|zfp
// ============================================================================
#include <vector>
#include <queue>
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <chrono>
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
#include <SZ3/api/sz.hpp>
#include <zfp.h>

// ============================================================================
// !!! 请按 scidx_avl.h 中 ScidxAVLNode / SkippedNode 的实际成员名修改这里 !!!
// ============================================================================
#define NODE_LOW(n)   ((n)->interval.low)
#define NODE_HIGH(n)  ((n)->interval.high)
#define NODE_ID(n)    ((n)->id)
#define SKIP_LOW(s)   ((s).interval.low)
#define SKIP_HIGH(s)  ((s).interval.high)
#define SKIP_ID(s)    ((s).id)
// ============================================================================

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

// 与原程序完全相同：树结构 bit-packing + Zstd
size_t saveTreeStructureToByte(const std::vector<int>& fullTreeStructure, const std::string& treeID)
{
    std::string filename = treeID + "-treeStructure";
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return 0;
    }

    size_t structureSize = fullTreeStructure.size();
    outFile.write(reinterpret_cast<const char*>(&structureSize), sizeof(size_t));

    std::vector<uint8_t> packedBytes;
    uint8_t currentByte = 0;
    int bitIndex = 0;
    for (int bit : fullTreeStructure) {
        currentByte |= (bit & 1) << bitIndex;
        if (++bitIndex == 8) {
            packedBytes.push_back(currentByte);
            currentByte = 0;
            bitIndex = 0;
        }
    }
    if (bitIndex > 0) packedBytes.push_back(currentByte);

    size_t bound = ZSTD_compressBound(packedBytes.size());
    std::vector<uint8_t> compressed(bound);
    size_t compressedSize = ZSTD_compress(compressed.data(), bound,
                                          packedBytes.data(), packedBytes.size(), 3);
    compressed.resize(compressedSize);

    outFile.write(reinterpret_cast<const char*>(&compressedSize), sizeof(size_t));
    outFile.write(reinterpret_cast<const char*>(compressed.data()), compressedSize);

    return sizeof(size_t) * 2 + compressedSize;
}

// ------------------------- 按原层序取出 low / high / id -----------------------
// 顺序：allLevels 逐层、每层从左到右；最后接 skippedAllIntervals
void extractLowHighID(const std::vector<std::vector<ScidxAVLNode<double>*>>& allLevels,
                      const std::vector<SkippedNode<double>>& skipped,
                      std::vector<double>& lows,
                      std::vector<double>& highs,
                      std::vector<uint64_t>& ids)
{
    for (const auto& level : allLevels) {
        for (const auto* node : level) {
            if (node == nullptr) continue;   // 层序中的空位
            lows.push_back(NODE_LOW(node));
            highs.push_back(NODE_HIGH(node));
            ids.push_back(static_cast<uint64_t>(NODE_ID(node)));
        }
    }
    for (const auto& s : skipped) {
        lows.push_back(SKIP_LOW(s));
        highs.push_back(SKIP_HIGH(s));
        ids.push_back(static_cast<uint64_t>(SKIP_ID(s)));
    }
}

// ------------------------------ SZ3 一维压缩 ---------------------------------
size_t compress1D_SZ3(const std::vector<double>& data, double eb, const std::string& outFile)
{
    if (data.empty()) return 0;

    SZ3::Config conf(data.size());
    conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;   // SZ3 默认算法
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = eb;

    std::vector<double> copy(data);             // 部分 SZ3 版本会改写输入
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
    zfp_stream_set_accuracy(zfp, eb);           // 误差有界模式

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    std::vector<unsigned char> buffer(bufsize);
    bitstream* stream = stream_open(buffer.data(), bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    size_t zfpSize = zfp_compress(zfp, field);
    if (zfpSize == 0) {
        std::cerr << "ZFP compression failed: " << outFile << std::endl;
    }

    std::ofstream out(outFile, std::ios::binary);
    out.write(reinterpret_cast<char*>(buffer.data()), zfpSize);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    return zfpSize;
}

// ------------------------ 块编号：原样 + Zstd（两种方案相同） -----------------
size_t compressIDs_Zstd(const std::vector<uint64_t>& ids, const std::string& outFile)
{
    if (ids.empty()) return 0;
    size_t rawBytes = ids.size() * sizeof(uint64_t);
    size_t bound = ZSTD_compressBound(rawBytes);
    std::vector<uint8_t> compressed(bound);
    size_t cSize = ZSTD_compress(compressed.data(), bound, ids.data(), rawBytes, 3);

    std::ofstream out(outFile, std::ios::binary);
    out.write(reinterpret_cast<const char*>(compressed.data()), cSize);
    return cSize;
}

// ------------------------ 建一棵 AVL 树并用 SZ3/ZFP 压缩 ----------------------
struct TreeSizes {
    size_t numIntervals = 0;
    size_t rawValueBytes = 0;   // low + high 原始大小
    size_t valueBytes = 0;      // low + high 压缩后大小
    size_t structBytes = 0;     // 树结构文件大小
    size_t idBytes = 0;         // 块编号压缩后大小
};

TreeSizes buildAndCompressTree(const std::vector<ScidxInterval<double>>& intervals,
                               const std::string& treeID,
                               const std::string& compressor,
                               double error_bound,
                               const std::string& tag,
                               int mpi_rank)
{
    TreeSizes sz;
    auto t1 = std::chrono::high_resolution_clock::now();

    // ===== 建树：与原程序完全一致 =====
    ScidxAVLIntervalTree<double> avlIntervalTree;
    std::vector<SkippedNode<double>> skippedSameLowIntervals;
    std::vector<SkippedNode<double>> skippedZeroLowIntervals;

    for (size_t i = 0; i < intervals.size(); i++) {
        if (intervals[i].low == 0) {
            skippedZeroLowIntervals.push_back(SkippedNode<double>(intervals[i], i));
            continue;
        }
        avlIntervalTree.insertNode(i, intervals[i], skippedSameLowIntervals);
    }

    if (avlIntervalTree.getRoot() == nullptr) {
        ScidxInterval<double> dummyInterval;
        dummyInterval.low = 0; dummyInterval.high = 0;
        std::vector<SkippedNode<double>> dummySkipped;
        avlIntervalTree.insertNode(static_cast<size_t>(-1), dummyInterval, dummySkipped);
    }

    std::vector<SkippedNode<double>> skippedAllIntervals;
    skippedAllIntervals.insert(skippedAllIntervals.end(),
                               skippedZeroLowIntervals.begin(), skippedZeroLowIntervals.end());
    skippedAllIntervals.insert(skippedAllIntervals.end(),
                               skippedSameLowIntervals.begin(), skippedSameLowIntervals.end());

    if (skippedAllIntervals.empty()) {
        ScidxInterval<double> dummyInterval;
        dummyInterval.low = 0; dummyInterval.high = 0;
        skippedAllIntervals.emplace_back(dummyInterval, static_cast<size_t>(-1));
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // ===== 层序展开 + 树结构：与原程序完全一致 =====
    std::vector<int> fullTreeStructure;
    std::vector<std::vector<ScidxAVLNode<double>*>> allLevels;
    levelOrderTraversal(avlIntervalTree.getRoot(), avlIntervalTree.getTreeHeight() + 1,
                        allLevels, fullTreeStructure);
    sz.structBytes = saveTreeStructureToByte(fullTreeStructure, treeID);

    // ===== 替换 computeOptimizedAVL：同一层序的 low/high 交给 SZ3 或 ZFP =====
    std::vector<double> lows, highs;
    std::vector<uint64_t> ids;
    extractLowHighID(allLevels, skippedAllIntervals, lows, highs, ids);

    sz.numIntervals  = lows.size();
    sz.rawValueBytes = (lows.size() + highs.size()) * sizeof(double);

    if (compressor == "sz3") {
        sz.valueBytes = compress1D_SZ3(lows,  error_bound, treeID + "-sz3_low")
                      + compress1D_SZ3(highs, error_bound, treeID + "-sz3_high");
    } else {
        sz.valueBytes = compress1D_ZFP(lows,  error_bound, treeID + "-zfp_low")
                      + compress1D_ZFP(highs, error_bound, treeID + "-zfp_high");
    }
    sz.idBytes = compressIDs_Zstd(ids, treeID + "-ids_zstd");

    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "[Rank " << mpi_rank << "] [" << tag << "] build avl tree: "
              << std::chrono::duration<double>(t2 - t1).count() << " s, "
              << compressor << " compress + io: "
              << std::chrono::duration<double>(t3 - t2).count() << " s" << std::endl;

    std::cout << "[Rank " << mpi_rank << "] [" << tag << "] intervals = " << sz.numIntervals
              << ", raw low+high = " << sz.rawValueBytes << " B"
              << ", " << compressor << " low+high = " << sz.valueBytes << " B"
              << " (CR " << (sz.valueBytes ? (double)sz.rawValueBytes / sz.valueBytes : 0.0) << ")"
              << ", treeStructure = " << sz.structBytes << " B"
              << ", ids = " << sz.idBytes << " B" << std::endl;

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

    std::vector<size_t> blockShape;          // 读入但不使用，仅兼容脚本参数
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
        else if (arg == "--block_shape" && i + 1 < argc)
        {
            for (size_t j = 0; j < nDim; j++)
                blockShape.push_back(std::stoul(argv[++i]));
        }
        else if (arg == "--begin_step")
        {
            if (i + 1 < argc) beginStepNum = atoi(argv[i + 1]);
            else { std::cerr << "--begin_step option requires one argument." << std::endl; return 1; }
        }
        else if (arg == "--end_step")
        {
            if (i + 1 < argc) endStepNum = atoi(argv[i + 1]);
            else { std::cerr << "--end_step option requires one argument." << std::endl; return 1; }
        }
        else if (arg == "--target_steps")
        {
            while (i + 1 < argc && std::isdigit(argv[i + 1][0]))
                targetSteps.push_back(std::stoul(argv[++i]));
        }
        else if (arg == "--small_block_shape")
        {
            if (nDim && (int)(i + nDim) < argc)
            {
                for (size_t j = i + 1; j < i + 1 + nDim; j++)
                    smallBlockShape.push_back(atoi(argv[j]));
            }
            else
            {
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
        if (mpi_rank == 0)
            std::cerr << "Error: --compressor must be sz3 or zfp" << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (inputFileName.empty() || variableName.empty() || nDim == 0 || smallBlockShape.size() != nDim) {
        if (mpi_rank == 0)
            std::cerr << "Error: missing or invalid arguments." << std::endl;
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

            // 错位分块
            std::vector<size_t> staggerBlockCountOnEachDim(nDim);
            size_t total_stagger_blocks = 1;
            for (size_t i = 0; i < nDim; i++) {
                staggerBlockCountOnEachDim[i] = var.Shape()[i] / smallBlockShape[i] + 1;
                total_stagger_blocks *= staggerBlockCountOnEachDim[i];
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
                    if (global_id[i] < halfBlockShape[i])
                        stagger_block_id[i] = 0;
                    else
                        stagger_block_id[i] = 1 + (global_id[i] - halfBlockShape[i]) / smallBlockShape[i];
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

    std::string rootDir = "/expanse/lustre/scratch/sdi/temp_project/index_compression_" + compressor + "/";
    std::string caseName = inputFileBaseName + "_" + safeVarName + "_"
                         + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                         + std::to_string(extraValue);
    std::string subDir_uniform = rootDir + caseName + "_stagger1_index/";
    std::string subDir_stagger = rootDir + caseName + "_stagger2_index/";

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
    }

    if (mpi_rank == 0) {
        saveBigBlockMinMax(subDir_uniform + "big_block_minmax", all_steps_uniform);
        saveBigBlockMinMax(subDir_stagger + "big_block_minmax", all_steps_stagger);
    }

    size_t numSteps = all_steps_uniform.size();

    // ---------------------- 阶段 3、4：两套索引建树 + 压缩 --------------------
    double local_uniform = 0.0, local_stagger = 0.0;
    size_t localSizes[2][4] = {{0}};   // [uniform/stagger][raw, value, struct, id]

    for (int which = 0; which < 2; which++)
    {
        const auto& all_steps = (which == 0) ? all_steps_uniform : all_steps_stagger;
        const std::string& subDir = (which == 0) ? subDir_uniform : subDir_stagger;
        const std::string tag = (which == 0) ? "UNIFORM" : "STAGGER";

        if (mpi_rank == 0)
            std::cout << "\n=== Building " << tag << " index (" << compressor << ") ===" << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        for (size_t s = 0; s < numSteps; ++s) {
            size_t actualStep = beginStepNum + s;
            size_t numBigBlocks = all_steps[s].size();

            for (size_t big = 0; big < numBigBlocks; ++big) {
                size_t task_id = s * numBigBlocks + big;
                if (task_id % mpi_size != (size_t)mpi_rank) continue;

                std::string treeID = subDir + std::to_string(actualStep) + "-" + std::to_string(big);
                TreeSizes sz = buildAndCompressTree(all_steps[s][big], treeID, compressor,
                                                    error_bound, tag, mpi_rank);
                localSizes[which][0] += sz.rawValueBytes;
                localSizes[which][1] += sz.valueBytes;
                localSizes[which][2] += sz.structBytes;
                localSizes[which][3] += sz.idBytes;
            }
        }

        double t_end = MPI_Wtime();
        if (which == 0) local_uniform = t_end - t_start;
        else            local_stagger = t_end - t_start;
    }

    // ------------------------------ 汇总输出 --------------------------------
    double max_uniform = 0.0, max_stagger = 0.0;
    MPI_Reduce(&local_uniform, &max_uniform, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stagger, &max_stagger, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    unsigned long long globalSizes[2][4] = {{0}};
    for (int w = 0; w < 2; w++)
        for (int k = 0; k < 4; k++) {
            unsigned long long v = localSizes[w][k];
            MPI_Reduce(&v, &globalSizes[w][k], 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        }

    if (mpi_rank == 0) {
        std::cout << "\n========== SUMMARY (" << compressor << ") ==========" << std::endl;
        std::cout << "[Stage 1] ADIOS read:                    " << total_adios_read_time << " s" << std::endl;
        std::cout << "[Stage 2] Partition + minmax + interval: " << total_partition_time  << " s" << std::endl;
        std::cout << "[Stage 3] UNIFORM build + compress:      " << max_uniform           << " s" << std::endl;
        std::cout << "[Stage 4] STAGGER build + compress:      " << max_stagger           << " s" << std::endl;

        const char* names[2] = {"UNIFORM", "STAGGER"};
        for (int w = 0; w < 2; w++) {
            unsigned long long raw = globalSizes[w][0], val = globalSizes[w][1];
            std::cout << "[" << names[w] << "] raw low+high = " << raw << " B"
                      << ", " << compressor << " low+high = " << val << " B"
                      << " (value CR " << (val ? (double)raw / val : 0.0) << ")"
                      << ", treeStructure = " << globalSizes[w][2] << " B"
                      << ", ids = " << globalSizes[w][3] << " B" << std::endl;
        }
        std::cout << "==========================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
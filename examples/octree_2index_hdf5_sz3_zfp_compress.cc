// ============================================================================
// NYX (HDF5, float32) Octree 索引压缩对比实验：叶子 low/high 改用 SZ3 或 ZFP 一维压缩
// 输入参数与原 NYX octree 程序一致，新增 --compressor sz3|zfp
// ============================================================================
#include <vector>
#include <queue>
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
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>
#include <utility>
#include <zstd.h>
#include <mpi.h>
#include <H5Cpp.h>
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

std::vector<float> readHDF5Dataset(const std::string& filename,
                                   const std::string& datasetName,
                                   std::vector<size_t>& outShape)
{
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(datasetName);
    H5::DataSpace dataspace = dataset.getSpace();

    int rank = dataspace.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    dataspace.getSimpleExtentDims(dims.data(), nullptr);

    outShape.resize(rank);
    size_t totalElements = 1;
    for (int i = 0; i < rank; i++) {
        outShape[i] = static_cast<size_t>(dims[i]);
        totalElements *= outShape[i];
    }

    if (dataset.getDataType().getSize() != sizeof(float))
        throw std::runtime_error("Dataset '" + datasetName + "' is not float32.");

    std::cout << "[HDF5] Opening dataset: " << datasetName << ", total elements = "
              << totalElements << std::endl;

    std::vector<float> data(totalElements);
    auto t0 = std::chrono::high_resolution_clock::now();
    dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[HDF5] Read Time: " << std::chrono::duration<double>(t1 - t0).count()
              << " seconds" << std::endl;
    return data;
}

void saveBigBlockMinMax(const std::string& filename,
                        const std::vector<ScidxInterval<float>>& intervals)
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    for (const auto& interval : intervals) {
        minVal = std::min(minVal, interval.low);
        maxVal = std::max(maxVal, interval.high);
    }
    outFile.write(reinterpret_cast<const char*>(&minVal), sizeof(float));
    outFile.write(reinterpret_cast<const char*>(&maxVal), sizeof(float));
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

// 按 Hilbert 顺序排列的块位置（块位置按 indicesToPosition：第 0 维变化最快）
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
size_t compress1D_SZ3(const std::vector<float>& data, double eb, const std::string& outFile)
{
    if (data.empty()) return 0;
    SZ3::Config conf(data.size());
    conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = eb;

    std::vector<float> copy(data);
    size_t cmpSize = 0;
    char* cmpData = SZ_compress(conf, copy.data(), cmpSize);

    std::ofstream out(outFile, std::ios::binary);
    out.write(cmpData, cmpSize);
    delete[] cmpData;
    return cmpSize;
}

// ------------------------------ ZFP 一维压缩 ---------------------------------
size_t compress1D_ZFP(const std::vector<float>& data, double eb, const std::string& outFile)
{
    if (data.empty()) return 0;
    std::vector<float> copy(data);
    zfp_field* field = zfp_field_1d(copy.data(), zfp_type_float, copy.size());
    zfp_stream* zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, eb);

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    std::vector<unsigned char> buffer(bufsize);
    bitstream* stream = stream_open(buffer.data(), bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    size_t zfpSize = zfp_compress(zfp, field);
    if (zfpSize == 0) std::cerr << "ZFP compression failed: " << outFile << std::endl;

    std::ofstream out(outFile, std::ios::binary);
    out.write(reinterpret_cast<char*>(buffer.data()), zfpSize);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    return zfpSize;
}

// ------------------ 建一棵 octree，并用 SZ3/ZFP 压缩叶子 low/high -------------
void buildAndCompressOctree(const std::vector<ScidxInterval<float>>& intervals,
                            const std::vector<size_t>& blockCount,
                            bool useHilbert,
                            const std::string& treeID,
                            const std::string& compressor,
                            float error_bound,
                            const std::string& tag,
                            size_t& rawOut, size_t& cmpOut)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> blockMins(intervals.size()), blockMaxs(intervals.size());
    for (size_t i = 0; i < intervals.size(); i++) {
        blockMins[i] = intervals[i].low;
        blockMaxs[i] = intervals[i].high;
    }

    // 与原程序一致地建 octree（内部节点由叶子推出，压缩只需叶子）
    {
        std::vector<OctreeNode<float>> octree = buildOctree(blockMins, blockMaxs, blockCount);
    }   // 用完即释放，降低峰值内存

    auto t2 = std::chrono::high_resolution_clock::now();

    // 叶子顺序：UNIFORM 按 Hilbert 重排（与原方法一致），STAGGER 保持原始块顺序
    std::vector<float> lows, highs;
    if (useHilbert) {
        std::vector<size_t> order = hilbertOrder(blockCount);
        lows.reserve(order.size());
        highs.reserve(order.size());
        for (size_t p : order) {
            lows.push_back(blockMins[p]);
            highs.push_back(blockMaxs[p]);
        }
    } else {
        lows.swap(blockMins);
        highs.swap(blockMaxs);
    }

    size_t rawBytes = (lows.size() + highs.size()) * sizeof(float);
    size_t cmpBytes = 0;
    if (compressor == "sz3") {
        cmpBytes = compress1D_SZ3(lows,  error_bound, treeID + "-sz3_low")
                 + compress1D_SZ3(highs, error_bound, treeID + "-sz3_high");
    } else {
        cmpBytes = compress1D_ZFP(lows,  error_bound, treeID + "-zfp_low")
                 + compress1D_ZFP(highs, error_bound, treeID + "-zfp_high");
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "[" << tag << "] build octree: "
              << std::chrono::duration<double>(t2 - t1).count() << " s, "
              << compressor << " compress + io: "
              << std::chrono::duration<double>(t3 - t2).count() << " s" << std::endl;
    std::cout << "[" << tag << "] leaves = " << lows.size()
              << ", order = " << (useHilbert ? "hilbert" : "native")
              << ", raw = " << rawBytes << " B"
              << ", " << compressor << " = " << cmpBytes << " B"
              << " (CR " << (cmpBytes ? (double)rawBytes / cmpBytes : 0.0) << ")" << std::endl;

    rawOut = rawBytes;
    cmpOut = cmpBytes;
}

// ================================== main =====================================
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);   // scidx 库内部依赖 MPI 环境

    std::string inputFileName;
    std::string variableName;
    std::string compressor = "sz3";          // 新增：sz3 或 zfp
    size_t nDim = 0;
    size_t beginStepNum = 0;
    size_t endStepNum = 0;
    std::vector<size_t> smallBlockShape;
    double relative_error_bound = 1E-3;
    size_t extraValue = 0;

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
        else if (arg == "--begin_step") {
            if (i + 1 < argc) beginStepNum = atoi(argv[i + 1]);
            else { std::cerr << "--begin_step option requires one argument." << std::endl; return 1; }
        }
        else if (arg == "--end_step") {
            if (i + 1 < argc) endStepNum = atoi(argv[i + 1]);
            else { std::cerr << "--end_step option requires one argument." << std::endl; return 1; }
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
        std::cerr << "Error: --compressor must be sz3 or zfp" << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (inputFileName.empty() || variableName.empty() || nDim != 3 || smallBlockShape.size() != nDim) {
        std::cerr << "Error: missing or invalid arguments (octree requires 3D)." << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::vector<size_t> halfBlockShape(nDim);
    for (size_t i = 0; i < nDim; i++)
        halfBlockShape[i] = smallBlockShape[i] / 2;

    // ----------------------------- 读数据 -----------------------------------
    std::vector<size_t> dataShape;
    std::vector<float> varData;
    try {
        varData = readHDF5Dataset(inputFileName, variableName, dataShape);
    } catch (...) {
        std::cerr << "Failed to read HDF5 dataset. Aborting." << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (dataShape.size() != nDim) {
        std::cerr << "Error: --dimensions does not match dataset rank." << std::endl;
        MPI_Finalize();
        return 1;
    }
    size_t totalElements = varData.size();

    // ---------------- 均匀分块 + 错位分块（与原 NYX octree 程序一致） ----------------
    std::vector<size_t> uniformBlockCount(nDim);
    size_t total_uniform_blocks = 1;
    for (size_t i = 0; i < nDim; i++) {
        uniformBlockCount[i] = (dataShape[i] + smallBlockShape[i] - 1) / smallBlockShape[i];
        total_uniform_blocks *= uniformBlockCount[i];
    }

    std::vector<size_t> staggerBlockCount(nDim);
    size_t total_stagger_blocks = 1;
    for (size_t i = 0; i < nDim; i++) {
        staggerBlockCount[i] = dataShape[i] / smallBlockShape[i];   // 开头半块并入第一个整块
        total_stagger_blocks *= staggerBlockCount[i];
    }

    auto partitionStart = std::chrono::high_resolution_clock::now();

    std::vector<float> uniform_mins(total_uniform_blocks, std::numeric_limits<float>::max());
    std::vector<float> uniform_maxs(total_uniform_blocks, std::numeric_limits<float>::lowest());
    std::vector<float> stagger_mins(total_stagger_blocks, std::numeric_limits<float>::max());
    std::vector<float> stagger_maxs(total_stagger_blocks, std::numeric_limits<float>::lowest());

    for (size_t p = 0; p < totalElements; p++)
    {
        std::vector<size_t> elem_global_id = positionToIndices(p, dataShape);

        std::vector<size_t> uniform_block_id(nDim);
        for (size_t i = 0; i < nDim; i++)
            uniform_block_id[i] = elem_global_id[i] / smallBlockShape[i];
        size_t uniform_pos = indicesToPosition(uniformBlockCount, uniform_block_id);
        uniform_mins[uniform_pos] = std::min(uniform_mins[uniform_pos], varData[p]);
        uniform_maxs[uniform_pos] = std::max(uniform_maxs[uniform_pos], varData[p]);

        std::vector<size_t> stagger_block_id(nDim);
        for (size_t i = 0; i < nDim; i++) {
            if (elem_global_id[i] < halfBlockShape[i] + smallBlockShape[i])
                stagger_block_id[i] = 0;
            else
                stagger_block_id[i] = (elem_global_id[i] - halfBlockShape[i]) / smallBlockShape[i];
        }
        size_t stagger_pos = indicesToPosition(staggerBlockCount, stagger_block_id);
        stagger_mins[stagger_pos] = std::min(stagger_mins[stagger_pos], varData[p]);
        stagger_maxs[stagger_pos] = std::max(stagger_maxs[stagger_pos], varData[p]);
    }

    std::vector<float>().swap(varData);

    std::vector<ScidxInterval<float>> uniformIntervals(total_uniform_blocks);
    for (size_t b = 0; b < total_uniform_blocks; b++) {
        uniformIntervals[b].low  = uniform_mins[b];
        uniformIntervals[b].high = uniform_maxs[b];
    }
    std::vector<ScidxInterval<float>> staggerIntervals(total_stagger_blocks);
    for (size_t b = 0; b < total_stagger_blocks; b++) {
        staggerIntervals[b].low  = stagger_mins[b];
        staggerIntervals[b].high = stagger_maxs[b];
    }
    std::vector<float>().swap(uniform_mins);
    std::vector<float>().swap(uniform_maxs);
    std::vector<float>().swap(stagger_mins);
    std::vector<float>().swap(stagger_maxs);

    auto partitionEnd = std::chrono::high_resolution_clock::now();
    double partitionTime = std::chrono::duration<double>(partitionEnd - partitionStart).count();

    // ------------------------------ 误差界 ----------------------------------
    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();
    for (const auto* v : {&uniformIntervals, &staggerIntervals})
        for (const auto& interval : *v) {
            global_min = std::min(global_min, interval.low);
            global_max = std::max(global_max, interval.high);
        }
    float error_bound = static_cast<float>(relative_error_bound) * (global_max - global_min);
    std::cout << "compressor = " << compressor << std::endl;
    std::cout << "error_bound = " << error_bound << std::endl;

    // ------------------------------ 输出目录 --------------------------------
    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    std::string rootDir = "/expanse/lustre/scratch/sdi/temp_project/nyx_octree_index_compression_" + compressor + "/";
    std::string caseName = inputFileBaseName + "_" + safeVarName + "_"
                         + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                         + std::to_string(extraValue);
    std::string subDir_uniform = rootDir + caseName + "_hdf5_octree_uniform_index/";
    std::string subDir_stagger = rootDir + caseName + "_hdf5_octree_stagger_index/";

    std::filesystem::create_directories(subDir_uniform);
    std::filesystem::create_directories(subDir_stagger);

    saveBigBlockMinMax(subDir_uniform + "big_block_minmax", uniformIntervals);
    saveBigBlockMinMax(subDir_stagger + "big_block_minmax", staggerIntervals);

    // ---------------------------- 建树 + 压缩 -------------------------------
    size_t rawU = 0, cmpU = 0, rawS = 0, cmpS = 0;

    auto tU0 = std::chrono::high_resolution_clock::now();
    std::cout << "\n=== Building UNIFORM Octree index (" << compressor << ") ===" << std::endl;
    buildAndCompressOctree(uniformIntervals, uniformBlockCount, true,
                           subDir_uniform + std::to_string(beginStepNum) + "-0",
                           compressor, error_bound, "UNIFORM", rawU, cmpU);
    auto tU1 = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== Building STAGGER Octree index (" << compressor << ") ===" << std::endl;
    buildAndCompressOctree(staggerIntervals, staggerBlockCount, false,
                           subDir_stagger + std::to_string(beginStepNum) + "-0",
                           compressor, error_bound, "STAGGER", rawS, cmpS);
    auto tS1 = std::chrono::high_resolution_clock::now();

    auto cr = [](size_t raw, size_t cmp) { return cmp ? (double)raw / cmp : 0.0; };

    std::cout << "\n========== SUMMARY (octree, " << compressor << ") ==========" << std::endl;
    std::cout << "[Stage 2] Partition + minmax + interval:   " << partitionTime << " s" << std::endl;
    std::cout << "[Stage 3] UNIFORM Octree build + compress: "
              << std::chrono::duration<double>(tU1 - tU0).count() << " s" << std::endl;
    std::cout << "[Stage 4] STAGGER Octree build + compress: "
              << std::chrono::duration<double>(tS1 - tU1).count() << " s" << std::endl;
    std::cout << "[UNIFORM] raw = " << rawU << " B, " << compressor << " = " << cmpU
              << " B, CR = " << cr(rawU, cmpU) << std::endl;
    std::cout << "[STAGGER] raw = " << rawS << " B, " << compressor << " = " << cmpS
              << " B, CR = " << cr(rawS, cmpS) << std::endl;
    std::cout << "[BOTH]    raw = " << rawU + rawS << " B, " << compressor << " = " << cmpU + cmpS
              << " B, CR = " << cr(rawU + rawS, cmpU + cmpS) << std::endl;
    std::cout << "==================================================" << std::endl;

    MPI_Finalize();
    return 0;
}
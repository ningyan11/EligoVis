// ============================================================================
// NYX (HDF5, float32) AVL 索引压缩对比实验：low/high 改用 SZ3 或 ZFP 一维压缩
// 输入参数与原 NYX 索引程序一致，新增 --compressor sz3|zfp
// ============================================================================
#include <vector>
#include <queue>
#include <algorithm>
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
#include <SZ3/api/sz.hpp>
#include <zfp.h>

// ============================================================================
// !!! 与 avl_2index_sz3_zfp_compress.cc 中改好的成员名保持一致 !!!
// ============================================================================
#define NODE_LOW(n)   ((n)->interval.low)
#define NODE_HIGH(n)  ((n)->interval.high)
#define NODE_ID(n)    ((n)->id)
#define SKIP_LOW(s)   ((s).interval.low)
#define SKIP_HIGH(s)  ((s).interval.high)
#define SKIP_ID(s)    ((s).id)
// ============================================================================

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

// ------------------------------ HDF5 读取 -----------------------------------
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

// 与原程序相同：树结构 bit-packing + Zstd
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

// ---------------------- 按原层序取出 low / high / id -------------------------
void extractLowHighID(const std::vector<std::vector<ScidxAVLNode<float>*>>& allLevels,
                      const std::vector<SkippedNode<float>>& skipped,
                      std::vector<float>& lows,
                      std::vector<float>& highs,
                      std::vector<uint64_t>& ids)
{
    for (const auto& level : allLevels) {
        for (const auto* node : level) {
            if (node == nullptr) continue;
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

// ------------------------ 块编号：原样 + Zstd --------------------------------
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
void buildAndCompressAVL(const std::vector<ScidxInterval<float>>& intervals,
                         const std::string& treeID,
                         const std::string& compressor,
                         float error_bound,
                         const std::string& tag)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    // ===== 建树：与原程序一致 =====
    ScidxAVLIntervalTree<float> avlIntervalTree;
    std::vector<SkippedNode<float>> skippedSameLowIntervals;
    std::vector<SkippedNode<float>> skippedZeroLowIntervals;

    for (size_t i = 0; i < intervals.size(); i++) {
        if (intervals[i].low == 0) {
            skippedZeroLowIntervals.push_back(SkippedNode<float>(intervals[i], i));
            continue;
        }
        avlIntervalTree.insertNode(i, intervals[i], skippedSameLowIntervals);
    }

    if (avlIntervalTree.getRoot() == nullptr) {
        ScidxInterval<float> dummyInterval;
        dummyInterval.low = 0; dummyInterval.high = 0;
        std::vector<SkippedNode<float>> dummySkipped;
        avlIntervalTree.insertNode(static_cast<size_t>(-1), dummyInterval, dummySkipped);
    }

    std::vector<SkippedNode<float>> skippedAllIntervals;
    skippedAllIntervals.insert(skippedAllIntervals.end(),
                               skippedZeroLowIntervals.begin(), skippedZeroLowIntervals.end());
    skippedAllIntervals.insert(skippedAllIntervals.end(),
                               skippedSameLowIntervals.begin(), skippedSameLowIntervals.end());

    if (skippedAllIntervals.empty()) {
        ScidxInterval<float> dummyInterval;
        dummyInterval.low = 0; dummyInterval.high = 0;
        skippedAllIntervals.emplace_back(dummyInterval, static_cast<size_t>(-1));
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // ===== 层序展开 + 树结构：与原程序一致 =====
    std::vector<int> fullTreeStructure;
    std::vector<std::vector<ScidxAVLNode<float>*>> allLevels;
    levelOrderTraversal(avlIntervalTree.getRoot(), avlIntervalTree.getTreeHeight() + 1,
                        allLevels, fullTreeStructure);
    size_t structBytes = saveTreeStructureToByte(fullTreeStructure, treeID);

    // ===== 替换 computeOptimizedAVL：同一层序的 low/high 交给 SZ3 或 ZFP =====
    std::vector<float> lows, highs;
    std::vector<uint64_t> ids;
    extractLowHighID(allLevels, skippedAllIntervals, lows, highs, ids);

    size_t rawBytes = (lows.size() + highs.size()) * sizeof(float);
    size_t valueBytes = 0;
    if (compressor == "sz3") {
        valueBytes = compress1D_SZ3(lows,  error_bound, treeID + "-sz3_low")
                   + compress1D_SZ3(highs, error_bound, treeID + "-sz3_high");
    } else {
        valueBytes = compress1D_ZFP(lows,  error_bound, treeID + "-zfp_low")
                   + compress1D_ZFP(highs, error_bound, treeID + "-zfp_high");
    }
    size_t idBytes = compressIDs_Zstd(ids, treeID + "-ids_zstd");

    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "[" << tag << "] build avl tree: "
              << std::chrono::duration<double>(t2 - t1).count() << " s, "
              << compressor << " compress + io: "
              << std::chrono::duration<double>(t3 - t2).count() << " s" << std::endl;
    std::cout << "[" << tag << "] intervals = " << lows.size()
              << ", raw low+high = " << rawBytes << " B"
              << ", " << compressor << " low+high = " << valueBytes << " B"
              << " (CR " << (valueBytes ? (double)rawBytes / valueBytes : 0.0) << ")"
              << ", treeStructure = " << structBytes << " B"
              << ", ids = " << idBytes << " B" << std::endl;
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
    float relative_error_bound = 1E-3;
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
    if (inputFileName.empty() || variableName.empty() || nDim == 0 || smallBlockShape.size() != nDim) {
        std::cerr << "Error: missing or invalid arguments." << std::endl;
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
    float gmin = *std::min_element(varData.begin(), varData.end());
    float gmax = *std::max_element(varData.begin(), varData.end());
    float error_bound = relative_error_bound * (gmax - gmin);
    std::cout << "compressor = " << compressor << std::endl;
    std::cout << "error_bound = " << error_bound << std::endl;

    // ------------------- 均匀分块 + 错位分块（与原程序一致） -------------------
    std::vector<size_t> uniformBlockCountOnEachDim(nDim);
    size_t total_uniform_blocks = 1;
    for (size_t i = 0; i < nDim; i++) {
        uniformBlockCountOnEachDim[i] = (dataShape[i] + smallBlockShape[i] - 1) / smallBlockShape[i];
        total_uniform_blocks *= uniformBlockCountOnEachDim[i];
    }

    std::vector<size_t> staggerBlockCountOnEachDim(nDim);
    size_t total_stagger_blocks = 1;
    for (size_t i = 0; i < nDim; i++) {
        staggerBlockCountOnEachDim[i] = dataShape[i] / smallBlockShape[i] + 1;
        total_stagger_blocks *= staggerBlockCountOnEachDim[i];
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
        size_t uniform_pos = indicesToPosition(uniformBlockCountOnEachDim, uniform_block_id);
        uniform_mins[uniform_pos] = std::min(uniform_mins[uniform_pos], varData[p]);
        uniform_maxs[uniform_pos] = std::max(uniform_maxs[uniform_pos], varData[p]);

        std::vector<size_t> stagger_block_id(nDim);
        for (size_t i = 0; i < nDim; i++) {
            if (elem_global_id[i] < halfBlockShape[i])
                stagger_block_id[i] = 0;
            else
                stagger_block_id[i] = 1 + (elem_global_id[i] - halfBlockShape[i]) / smallBlockShape[i];
        }
        size_t stagger_pos = indicesToPosition(staggerBlockCountOnEachDim, stagger_block_id);
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

    // ------------------------------ 输出目录 --------------------------------
    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    std::string rootDir = "/expanse/lustre/scratch/sdi/temp_project/nyx_index_compression_" + compressor + "/";
    std::string caseName = inputFileBaseName + "_" + safeVarName + "_"
                         + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                         + std::to_string(extraValue);
    std::string subDir_uniform = rootDir + caseName + "_hdf5_uniform_index/";
    std::string subDir_stagger = rootDir + caseName + "_hdf5_stagger_index/";

    std::filesystem::create_directories(subDir_uniform);
    std::filesystem::create_directories(subDir_stagger);

    saveBigBlockMinMax(subDir_uniform + "big_block_minmax", uniformIntervals);
    saveBigBlockMinMax(subDir_stagger + "big_block_minmax", staggerIntervals);

    // ---------------------------- 建树 + 压缩 -------------------------------
    auto buildStart = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== Building UNIFORM index (" << compressor << ") ===" << std::endl;
    buildAndCompressAVL(uniformIntervals, subDir_uniform + std::to_string(beginStepNum) + "-0",
                        compressor, error_bound, "UNIFORM");

    std::cout << "\n=== Building STAGGER index (" << compressor << ") ===" << std::endl;
    buildAndCompressAVL(staggerIntervals, subDir_stagger + std::to_string(beginStepNum) + "-0",
                        compressor, error_bound, "STAGGER");

    auto buildEnd = std::chrono::high_resolution_clock::now();
    double buildTime = std::chrono::duration<double>(buildEnd - buildStart).count();

    std::cout << "\n========== TIME SUMMARY (" << compressor << ") ==========" << std::endl;
    std::cout << "[Stage1] Partition + minmax + interval build: " << partitionTime << " s" << std::endl;
    std::cout << "[Stage2] UNIFORM+STAGGER build + compress:     " << buildTime << " s" << std::endl;
    std::cout << "===================================================" << std::endl;

    MPI_Finalize();
    return 0;
}
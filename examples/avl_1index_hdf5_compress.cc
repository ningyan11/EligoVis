#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <random>
#include <scidx_avl.h>
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <utility>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>
#include <chrono>
#include <zstd.h>
#include <mpi.h>     // scidx库内部（如computeOptimizedAVL）依赖MPI环境已初始化，
                     // 即使本程序单进程运行，也必须先MPI_Init，否则库内部调用
                     // MPI_Comm_rank会触发 "called before MPI_INIT" 崩溃
#include <H5Cpp.h>   // HDF5 C++ API


void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
}

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

size_t indicesToPosition(const std::vector<size_t>& shape, const std::vector<size_t>& indices) {
    size_t position = indices[0];
    size_t multiplier = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        multiplier *= shape[i - 1];
        position += indices[i] * multiplier;
    }
    return position;
}

// ============================================================
// 从 HDF5 文件里读取一个 float32 dataset（与压缩程序 readHDF5Dataset 一致）
// ============================================================
std::vector<float> readHDF5Dataset(const std::string& filename,
                                    const std::string& datasetName,
                                    std::vector<size_t>& outShape) {
    try {
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

        // 检查数据类型大小，确保是 4 字节 (float32)
        H5::DataType dtype = dataset.getDataType();
        if (dtype.getSize() != sizeof(float)) {
            throw std::runtime_error(
                "Dataset '" + datasetName + "' element size is " +
                std::to_string(dtype.getSize()) +
                " bytes, expected 4 bytes (float32). "
                "如果是 float64, 请改用 double 版本的索引代码。");
        }

        std::cout << "[HDF5] Opening dataset: " << datasetName << std::endl;
        std::cout << "[HDF5] Shape: (";
        for (size_t i = 0; i < outShape.size(); i++) {
            std::cout << outShape[i] << (i + 1 < outShape.size() ? ", " : "");
        }
        std::cout << "), total elements = " << totalElements
                  << " (~" << (totalElements * sizeof(float)) / (1024.0*1024.0*1024.0)
                  << " GB)" << std::endl;

        std::vector<float> data(totalElements);

        auto readStart = std::chrono::high_resolution_clock::now();
        dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);
        auto readEnd = std::chrono::high_resolution_clock::now();
        double readTime = std::chrono::duration<double>(readEnd - readStart).count();
        std::cout << "[HDF5] Read Time: " << readTime << " seconds" << std::endl;

        return data;

    } catch (H5::FileIException& e) {
        std::cerr << "HDF5 File Error: " << e.getCDetailMsg() << std::endl;
        throw;
    } catch (H5::DataSetIException& e) {
        std::cerr << "HDF5 DataSet Error (dataset name wrong?): " << e.getCDetailMsg() << std::endl;
        throw;
    } catch (H5::DataSpaceIException& e) {
        std::cerr << "HDF5 DataSpace Error: " << e.getCDetailMsg() << std::endl;
        throw;
    }
}

// 保存每个"step"（此处只有1个）的小块min/max（去掉大块层，每个元素直接是小块interval）
void saveBigBlockMinMax(const std::string& filename,
                        const std::vector<std::vector<ScidxInterval<float>>>& all_steps_intervals)
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    size_t numSteps = all_steps_intervals.size();
    for (size_t step = 0; step < numSteps; ++step) {
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();
        for (const auto& interval : all_steps_intervals[step]) {
            minVal = std::min(minVal, interval.low);
            maxVal = std::max(maxVal, interval.high);
        }
        outFile.write(reinterpret_cast<const char*>(&minVal), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&maxVal), sizeof(float));
    }
    outFile.close();
}

// 树结构：bit-packing + zstd 压缩存储
void saveTreeStructureToByte(const std::vector<int>& fullTreeStructure, const std::string& treeID) {
    std::string filename = treeID + "-treeStructure";
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    size_t structureSize = fullTreeStructure.size();
    outFile.write(reinterpret_cast<const char*>(&structureSize), sizeof(size_t));

    std::vector<uint8_t> packedBytes;
    uint8_t currentByte = 0;
    int bitIndex = 0;
    for (int bit : fullTreeStructure) {
        currentByte |= (bit & 1) << bitIndex;
        bitIndex++;
        if (bitIndex == 8) {
            packedBytes.push_back(currentByte);
            currentByte = 0;
            bitIndex = 0;
        }
    }
    if (bitIndex > 0) {
        packedBytes.push_back(currentByte);
    }

    size_t bound = ZSTD_compressBound(packedBytes.size());
    std::vector<uint8_t> compressed(bound);
    size_t compressedSize = ZSTD_compress(
        compressed.data(), bound,
        packedBytes.data(), packedBytes.size(),
        3);
    compressed.resize(compressedSize);

    outFile.write(reinterpret_cast<const char*>(&compressedSize), sizeof(size_t));
    outFile.write(reinterpret_cast<const char*>(compressed.data()), compressedSize);

    outFile.close();
}


int main(int argc, char *argv[])
{
    // 单进程运行，不做任何多进程分工，只是满足scidx库内部对MPI环境
    // 已初始化的隐性依赖（库内部会调用MPI_Comm_rank等接口）
    MPI_Init(&argc, &argv);

    std::cout << "[DEBUG] Program started." << std::endl;

    std::string inputFileName;
    std::string variableName;   // HDF5 内部 dataset 路径, 例如 native_fields/velocity_x
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
        {
            inputFileName = argv[++i];
        }
        else if (arg == "--variable_name" && i + 1 < argc)
        {
            variableName = argv[++i];
        }
        else if (arg == "--dimensions" && i + 1 < argc)
        {
            nDim = std::stoul(argv[++i]);
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
        else if (arg == "--small_block_shape")
        {
            if (nDim)
            {
                if ((int)(i + nDim) < argc)
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
        }
        else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        }
    }

    if (inputFileName.empty() || variableName.empty() || nDim == 0 || smallBlockShape.size() != nDim)
    {
        std::cerr << "Error: missing or invalid arguments." << std::endl;
        std::cerr << "Usage: " << argv[0]
                  << " --input_file <path.hdf5> --variable_name <group/dataset> --dimensions 3"
                  << " --small_block_shape 4 4 4 --begin_step 0 --end_step 0"
                  << " --relative_error 1e-3 0" << std::endl;
        MPI_Finalize();
        return 1;
    }

    // ============================================================
    // 直接从 HDF5 读取整个 dataset（单 snapshot，不做多 step 循环）
    // begin_step / end_step 仅用于输出目录命名
    // ============================================================
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
        std::cerr << "Error: --dimensions " << nDim << " does not match actual HDF5 dataset rank "
                  << dataShape.size() << std::endl;
        MPI_Finalize();
        return 1;
    }

    size_t totalElements = varData.size();

    float Global_min_value = *std::min_element(varData.begin(), varData.end());
    float Global_max_value = *std::max_element(varData.begin(), varData.end());
    float error_bound = relative_error_bound * (Global_max_value - Global_min_value);

    std::cout << "[Data Range] Global_min_value = " << Global_min_value
               << ", Global_max_value = " << Global_max_value << std::endl;
    std::cout << "error_bound = " << error_bound << std::endl;

    // ============================================================
    // 计算小块布局、遍历所有点，直接归入小块并统计 min/max
    // （去掉大块层，直接对 small_block 建 interval）
    // ============================================================
    std::vector<size_t> smallBlockCountOnEachDim(nDim);
    size_t total_small_blocks = 1;
    for (size_t i = 0; i < nDim; i++) {
        smallBlockCountOnEachDim[i] = (dataShape[i] + smallBlockShape[i] - 1) / smallBlockShape[i];
        total_small_blocks *= smallBlockCountOnEachDim[i];
    }

    auto partitionStart = std::chrono::high_resolution_clock::now();

    std::vector<float> small_block_mins(total_small_blocks, std::numeric_limits<float>::max());
    std::vector<float> small_block_maxs(total_small_blocks, std::numeric_limits<float>::lowest());

    for (size_t p = 0; p < totalElements; p++)
    {
        std::vector<size_t> elem_global_id = positionToIndices(p, dataShape);
        std::vector<size_t> small_block_id(nDim);
        for (size_t i = 0; i < nDim; i++)
        {
            small_block_id[i] = elem_global_id[i] / smallBlockShape[i];
        }
        size_t small_block_position = indicesToPosition(smallBlockCountOnEachDim, small_block_id);

        small_block_mins[small_block_position] = std::min(small_block_mins[small_block_position], varData[p]);
        small_block_maxs[small_block_position] = std::max(small_block_maxs[small_block_position], varData[p]);
    }

    // 释放原始数据内存，不再需要
    std::vector<float>().swap(varData);

    std::vector<ScidxInterval<float>> intervals;
    intervals.reserve(total_small_blocks);
    for (size_t b = 0; b < total_small_blocks; b++) {
        ScidxInterval<float> interval;
        interval.low  = small_block_mins[b];
        interval.high = small_block_maxs[b];
        intervals.push_back(interval);
    }

    auto partitionEnd = std::chrono::high_resolution_clock::now();
    double partitionTime = std::chrono::duration<double>(partitionEnd - partitionStart).count();
    std::cout << "[Time] Partition + minmax + interval build: " << partitionTime << " seconds" << std::endl;

    // 只有一个"逻辑 step"
    std::vector<std::vector<ScidxInterval<float>>> all_steps_intervals;
    all_steps_intervals.push_back(intervals);

    // ============================================================
    // 输出目录
    // ============================================================
    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    std::string subDir = "/expanse/lustre/scratch/sdi/temp_project/nyx_index_compress/" + inputFileBaseName + "_" + safeVarName + "_"
                        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                        + std::to_string(extraValue) + "_hdf5_uniform_index/";

    createDirectory(subDir);

    // 存每个step（此处只有1个）的整体min/max
    saveBigBlockMinMax(subDir + "big_block_minmax", all_steps_intervals);

    // ============================================================
    // 建 AVL 区间树 + 压缩存储（单棵树，无 MPI，无 stagger）
    // ============================================================
    size_t numSteps = all_steps_intervals.size();

    auto buildStart = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < numSteps; ++s) {
        size_t actualStep = beginStepNum + s;
        std::string treeID = subDir + std::to_string(actualStep) + "-0";

        auto start_time1 = std::chrono::high_resolution_clock::now();

        ScidxAVLIntervalTree<float> avlIntervalTree;
        const std::vector<ScidxInterval<float>>& stepIntervals = all_steps_intervals[s];

        std::vector<SkippedNode<float>> skippedSameLowIntervals;
        std::vector<SkippedNode<float>> skippedZeroLowIntervals;

        for (size_t i = 0; i < stepIntervals.size(); i++) {
            if (stepIntervals[i].low == 0) {
                skippedZeroLowIntervals.push_back(SkippedNode<float>(stepIntervals[i], i));
                continue;
            }
            avlIntervalTree.insertNode(i, stepIntervals[i], skippedSameLowIntervals);
        }

        if (avlIntervalTree.getRoot() == nullptr) {
            ScidxInterval<float> dummyInterval;
            dummyInterval.low = 0; dummyInterval.high = 0;
            std::vector<SkippedNode<float>> dummySkipped;
            avlIntervalTree.insertNode(static_cast<size_t>(-1), dummyInterval, dummySkipped);
        }

        std::cout << "skippedSameLowIntervals size: " << skippedSameLowIntervals.size() << std::endl;
        std::cout << "skippedZeroLowIntervals size: " << skippedZeroLowIntervals.size() << std::endl;

        std::vector<SkippedNode<float>> skippedAllIntervals;
        skippedAllIntervals.insert(skippedAllIntervals.end(),
                                   skippedZeroLowIntervals.begin(), skippedZeroLowIntervals.end());
        skippedAllIntervals.insert(skippedAllIntervals.end(),
                                   skippedSameLowIntervals.begin(), skippedSameLowIntervals.end());
        skippedSameLowIntervals.clear();
        skippedZeroLowIntervals.clear();

        if (skippedAllIntervals.empty()) {
            ScidxInterval<float> dummyInterval;
            dummyInterval.low = 0; dummyInterval.high = 0;
            skippedAllIntervals.emplace_back(dummyInterval, static_cast<size_t>(-1));
        }

        auto start_time2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed2 = start_time2 - start_time1;
        std::cout << "[UNIFORM][Time1]: build avl tree " << elapsed2.count() << " seconds" << std::endl;

        std::vector<int> fullTreeStructure;
        std::vector<std::vector<ScidxAVLNode<float>*>> allLevels;
        levelOrderTraversal(avlIntervalTree.getRoot(), avlIntervalTree.getTreeHeight() + 1,
                            allLevels, fullTreeStructure);
        saveTreeStructureToByte(fullTreeStructure, treeID);
        computeOptimizedAVL(allLevels, error_bound, treeID, skippedAllIntervals);

        auto start_time3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed3 = start_time3 - start_time2;
        std::cout << "[UNIFORM][Time all compress and io]: compress avl tree "
                  << elapsed3.count() << " seconds" << std::endl;
    }

    auto buildEnd = std::chrono::high_resolution_clock::now();
    double buildTime = std::chrono::duration<double>(buildEnd - buildStart).count();

    std::cout << "\n========== TIME SUMMARY ==========" << std::endl;
    std::cout << "[Stage1] Partition + minmax + interval build: " << partitionTime << " s" << std::endl;
    std::cout << "[Stage2] UNIFORM build + compress:             " << buildTime << " s" << std::endl;
    std::cout << "[TOTAL]                                        " << (partitionTime + buildTime) << " s" << std::endl;
    std::cout << "===================================" << std::endl;

    MPI_Finalize();
    return 0;
}

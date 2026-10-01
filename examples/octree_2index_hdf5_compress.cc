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
#include <mpi.h>     // scidx库内部（如computeOptimizedAVL/Octree压缩函数）依赖MPI环境已初始化，
                     // 即使本程序单进程运行数据读取，也必须先MPI_Init，否则库内部调用
                     // MPI_Comm_rank会触发 "called before MPI_INIT" 崩溃
#include <H5Cpp.h>   // HDF5 C++ API
#include <scidx_octree_interval.h>  // 包含 OctreeNode, buildOctree, queryOctree
#include <scidx_octree.h>           // 包含 compressOctreeUniform, compressOctreeStagger
#include <scidx_octree_hilbert.h>   // 包含 compressOctreeUniformHilbert, compressOctreeStaggerHilbert
#include <scidx_octree_hilbert2.h>  // 包含 compressOctreeUniformHilbertNew, compressOctreeStaggerHilbertNew


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
// 从 HDF5 文件里读取一个 float32 dataset
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

// 保存 min/max（结构与原 Octree 版本一致：step -> bigBlock -> intervals，
// 此处每个 step 只有 1 个 bigBlock，因为 HDF5 是整体一次性读入）
void saveBigBlockMinMax(const std::string& filename,
                        const std::vector<std::vector<std::vector<ScidxInterval<float>>>>& all_steps_block_interval)
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    size_t numSteps = all_steps_block_interval.size();
    for (size_t step = 0; step < numSteps; ++step) {
        size_t numBigBlocks = all_steps_block_interval[step].size();
        for (size_t big = 0; big < numBigBlocks; ++big) {
            float minVal = std::numeric_limits<float>::max();
            float maxVal = std::numeric_limits<float>::lowest();
            for (const auto& interval : all_steps_block_interval[step][big]) {
                minVal = std::min(minVal, interval.low);
                maxVal = std::max(maxVal, interval.high);
            }
            outFile.write(reinterpret_cast<const char*>(&minVal), sizeof(float));
            outFile.write(reinterpret_cast<const char*>(&maxVal), sizeof(float));
        }
    }
    outFile.close();
}


int main(int argc, char *argv[])
{
    std::cout << "[DEBUG] Program started." << std::endl;

    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::string inputFileName;
    std::string variableName;   // HDF5 内部 dataset 路径, 例如 native_fields/velocity_x
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

    // 错位分块的半块大小
    std::vector<size_t> halfBlockShape(nDim);
    for (size_t i = 0; i < nDim; i++)
        halfBlockShape[i] = smallBlockShape[i] / 2;

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

    // ============================================================
    // 计算 均匀分块 + 错位分块 的布局
    // 错位分块沿用"新版"逻辑：开头半块与第一个整块合并成新 block0，
    // 后面的块整体往前挪一位（与 Octree/Hilbert 版本一致）
    // ============================================================
    std::vector<size_t> uniformBlockCountOnEachDim(nDim);
    size_t total_uniform_blocks = 1;
    for (size_t i = 0; i < nDim; i++) {
        uniformBlockCountOnEachDim[i] = (dataShape[i] + smallBlockShape[i] - 1) / smallBlockShape[i];
        total_uniform_blocks *= uniformBlockCountOnEachDim[i];
    }

    std::vector<size_t> staggerBlockCountOnEachDim(nDim);
    size_t total_stagger_blocks = 1;
    for (size_t i = 0; i < nDim; i++) {
        staggerBlockCountOnEachDim[i] = dataShape[i] / smallBlockShape[i];  // 去掉 +1，因为开头半块要合并进第一个整块
        total_stagger_blocks *= staggerBlockCountOnEachDim[i];
    }

    auto partitionStart = std::chrono::high_resolution_clock::now();

    std::vector<float> uniform_mins(total_uniform_blocks, std::numeric_limits<float>::max());
    std::vector<float> uniform_maxs(total_uniform_blocks, std::numeric_limits<float>::lowest());
    std::vector<float> stagger_mins(total_stagger_blocks, std::numeric_limits<float>::max());
    std::vector<float> stagger_maxs(total_stagger_blocks, std::numeric_limits<float>::lowest());

    // 一次遍历，同时完成均匀分块和错位分块的 min/max 统计
    for (size_t p = 0; p < totalElements; p++)
    {
        std::vector<size_t> elem_global_id = positionToIndices(p, dataShape);

        // 均匀块归属 + min/max
        std::vector<size_t> uniform_block_id(nDim);
        for (size_t i = 0; i < nDim; i++)
            uniform_block_id[i] = elem_global_id[i] / smallBlockShape[i];
        size_t uniform_pos = indicesToPosition(uniformBlockCountOnEachDim, uniform_block_id);
        uniform_mins[uniform_pos] = std::min(uniform_mins[uniform_pos], varData[p]);
        uniform_maxs[uniform_pos] = std::max(uniform_maxs[uniform_pos], varData[p]);

        // 错位块归属 + min/max（新版：开头半块合并进第一个整块）
        std::vector<size_t> stagger_block_id(nDim);
        for (size_t i = 0; i < nDim; i++)
        {
            if (elem_global_id[i] < halfBlockShape[i] + smallBlockShape[i])
                stagger_block_id[i] = 0;
            else
                stagger_block_id[i] = (elem_global_id[i] - halfBlockShape[i]) / smallBlockShape[i];
        }
        size_t stagger_pos = indicesToPosition(staggerBlockCountOnEachDim, stagger_block_id);
        stagger_mins[stagger_pos] = std::min(stagger_mins[stagger_pos], varData[p]);
        stagger_maxs[stagger_pos] = std::max(stagger_maxs[stagger_pos], varData[p]);
    }

    // 释放原始数据内存，不再需要
    std::vector<float>().swap(varData);

    std::vector<ScidxInterval<float>> uniformIntervals;
    uniformIntervals.reserve(total_uniform_blocks);
    for (size_t b = 0; b < total_uniform_blocks; b++) {
        ScidxInterval<float> interval;
        interval.low  = uniform_mins[b];
        interval.high = uniform_maxs[b];
        uniformIntervals.push_back(interval);
    }

    std::vector<ScidxInterval<float>> staggerIntervals;
    staggerIntervals.reserve(total_stagger_blocks);
    for (size_t b = 0; b < total_stagger_blocks; b++) {
        ScidxInterval<float> interval;
        interval.low  = stagger_mins[b];
        interval.high = stagger_maxs[b];
        staggerIntervals.push_back(interval);
    }

    auto partitionEnd = std::chrono::high_resolution_clock::now();
    double partitionTime = std::chrono::duration<double>(partitionEnd - partitionStart).count();
    std::cout << "[Stage2] Partition + minmax + interval build (uniform+stagger): "
              << partitionTime << " seconds" << std::endl;

    // 只有一个"逻辑 step"，每个 step 只有 1 个 bigBlock（HDF5 一次性整体读入）
    std::vector<std::vector<std::vector<ScidxInterval<float>>>> all_steps_uniform;
    all_steps_uniform.push_back({uniformIntervals});
    std::vector<std::vector<std::vector<ScidxInterval<float>>>> all_steps_stagger;
    all_steps_stagger.push_back({staggerIntervals});

    std::vector<size_t> uniformBlockCountOnEachDim_global = uniformBlockCountOnEachDim;
    std::vector<size_t> staggerBlockCountOnEachDim_global = staggerBlockCountOnEachDim;

    // ============================================================
    // 输出目录（分别给 uniform / stagger 两套 Octree/Hilbert 索引）
    // ============================================================
    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    std::string subDir_uniform = "/expanse/lustre/scratch/sdi/temp_project/octree_index_compressed/"
                        + inputFileBaseName + "_" + safeVarName + "_"
                        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                        + std::to_string(extraValue) + "_hdf5_octree_hilbert_uniform_index/";

    std::string subDir_stagger = "/expanse/lustre/scratch/sdi/temp_project/octree_index_compressed/"
                        + inputFileBaseName + "_" + safeVarName + "_"
                        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                        + std::to_string(extraValue) + "_hdf5_octree_hilbert_stagger_index/";

    if (mpi_rank == 0) {
        createDirectory(subDir_uniform);
        createDirectory(subDir_stagger);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ============================================================
    // 全局 min/max + error_bound（从 interval 里重新统计，逻辑与原 Octree 版本一致）
    // ============================================================
    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();

    for (const auto& step_blocks : all_steps_uniform)
        for (const auto& big_block : step_blocks)
            for (const auto& interval : big_block) {
                global_min = std::min(global_min, interval.low);
                global_max = std::max(global_max, interval.high);
            }
    for (const auto& step_blocks : all_steps_stagger)
        for (const auto& big_block : step_blocks)
            for (const auto& interval : big_block) {
                global_min = std::min(global_min, interval.low);
                global_max = std::max(global_max, interval.high);
            }

    float error_bound = static_cast<float>(relative_error_bound) * (global_max - global_min);
    std::cout << "[Data Range] global_min = " << global_min << ", global_max = " << global_max << std::endl;
    std::cout << "error_bound = " << error_bound << std::endl;

    saveBigBlockMinMax(subDir_uniform + "big_block_minmax", all_steps_uniform);
    saveBigBlockMinMax(subDir_stagger + "big_block_minmax", all_steps_stagger);

    size_t numSteps = all_steps_uniform.size();

    // ============================================================
    // 均匀分块 Octree 索引构建（[Hilbert2] 版本，导出 listLow/listHigh 供 STAGGER 查邻居用）
    // ============================================================
    if (mpi_rank == 0)
        std::cout << "\n=== Building UNIFORM Octree index ===" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    double build_start_time_uniform = MPI_Wtime();

    std::vector<std::vector<std::vector<float>>> listLowStorage(numSteps);
    std::vector<std::vector<std::vector<float>>> listHighStorage(numSteps);
    for (size_t s = 0; s < numSteps; s++) {
        listLowStorage[s].resize(all_steps_uniform[s].size());
        listHighStorage[s].resize(all_steps_uniform[s].size());
    }

    for (size_t s = 0; s < numSteps; ++s) {
        size_t actualStep = beginStepNum + s;
        size_t numBigBlocks = all_steps_uniform[s].size();

        for (size_t big = 0; big < numBigBlocks; ++big) {
            size_t task_id = s * numBigBlocks + big;
            if (task_id % mpi_size != mpi_rank) continue;

            auto start_time1 = std::chrono::high_resolution_clock::now();

            std::vector<ScidxInterval<float>> intervals = all_steps_uniform[s][big];
            std::string treeID = subDir_uniform + std::to_string(actualStep) + "-" + std::to_string(big);

            std::vector<float> blockMins(intervals.size()), blockMaxs(intervals.size());
            for (size_t i = 0; i < intervals.size(); i++) {
                blockMins[i] = intervals[i].low;
                blockMaxs[i] = intervals[i].high;
            }

            // 建立 Octree
            std::vector<OctreeNode<float>> octree = buildOctree(
                blockMins, blockMaxs, uniformBlockCountOnEachDim_global);

            auto start_time2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = start_time2 - start_time1;
            std::cout << "[Rank " << mpi_rank << "] [UNIFORM][Time1]: build octree "
                    << elapsed2.count() << " seconds" << std::endl;

            // [Hilbert2] 压缩并保存，同时输出 listLow、listHigh 供 STAGGER 用
            compressOctreeUniformHilbertNew(octree, error_bound, treeID,
                                        uniformBlockCountOnEachDim_global,
                                        &listLowStorage[s][big],
                                        &listHighStorage[s][big]);

            auto start_time3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed3 = start_time3 - start_time2;
            std::cout << "[Rank " << mpi_rank << "] [UNIFORM][Time all compress and io]: "
                    << elapsed3.count() << " seconds" << std::endl;
        }
    }

    double build_end_time_uniform = MPI_Wtime();
    double local_uniform = build_end_time_uniform - build_start_time_uniform;

    double max_uniform, sum_uniform;
    MPI_Reduce(&local_uniform, &max_uniform, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_uniform, &sum_uniform, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::cout << "Total UNIFORM Octree build + compression time (max across ranks): "
                  << max_uniform << " seconds." << std::endl;
        std::cout << "Average UNIFORM Octree build + compression time: "
                  << sum_uniform / mpi_size << " seconds." << std::endl;
    }

    // ============================================================
    // 错位分块 Octree 索引构建（[Hilbert2] 版本，查 UNIFORM 的 listLow/listHigh 做 8 邻居平均预测）
    // ============================================================
    if (mpi_rank == 0)
        std::cout << "\n=== Building STAGGER Octree index ===" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    double build_start_time_stagger = MPI_Wtime();

    for (size_t s = 0; s < numSteps; ++s) {
        size_t actualStep = beginStepNum + s;
        size_t numBigBlocks = all_steps_stagger[s].size();

        for (size_t big = 0; big < numBigBlocks; ++big) {
            size_t task_id = s * numBigBlocks + big;
            if (task_id % mpi_size != mpi_rank) continue;

            auto start_time1 = std::chrono::high_resolution_clock::now();

            std::vector<ScidxInterval<float>> intervals = all_steps_stagger[s][big];
            std::string treeID = subDir_stagger + std::to_string(actualStep) + "-" + std::to_string(big);

            size_t expectedBlocks = 1;
            for (auto d : staggerBlockCountOnEachDim_global) expectedBlocks *= d;
            std::cout << "[DEBUG STAGGER] intervals.size()=" << intervals.size()
                    << " expectedBlocks=" << expectedBlocks << std::endl;

            std::vector<float> blockMins(intervals.size()), blockMaxs(intervals.size());
            for (size_t i = 0; i < intervals.size(); i++) {
                blockMins[i] = intervals[i].low;
                blockMaxs[i] = intervals[i].high;
            }

            // 建立 Octree
            std::vector<OctreeNode<float>> octree = buildOctree(
                blockMins, blockMaxs, staggerBlockCountOnEachDim_global);

            auto start_time2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = start_time2 - start_time1;
            std::cout << "[Rank " << mpi_rank << "] [STAGGER][Time1]: build octree "
                      << elapsed2.count() << " seconds" << std::endl;

            // [Hilbert2] STAGGER 查 UNIFORM 的 listLow/listHigh 做 8 邻居平均预测
            compressOctreeStaggerHilbertNew(octree, error_bound, treeID,
                                        staggerBlockCountOnEachDim_global,
                                        uniformBlockCountOnEachDim_global,
                                        listLowStorage[s][big],
                                        listHighStorage[s][big]);

            auto start_time3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed3 = start_time3 - start_time2;
            std::cout << "[Rank " << mpi_rank << "] [STAGGER][Time all compress and io]: "
                      << elapsed3.count() << " seconds" << std::endl;
        }
    }

    double build_end_time_stagger = MPI_Wtime();
    double local_stagger = build_end_time_stagger - build_start_time_stagger;

    double max_stagger, sum_stagger;
    MPI_Reduce(&local_stagger, &max_stagger, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stagger, &sum_stagger, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::cout << "Total STAGGER Octree build + compression time (max across ranks): "
                  << max_stagger << " seconds." << std::endl;
        std::cout << "Average STAGGER Octree build + compression time: "
                  << sum_stagger / mpi_size << " seconds." << std::endl;

        std::cout << "\n========== TIME SUMMARY ==========" << std::endl;
        std::cout << "[Stage 2] Partition + minmax + interval:   " << partitionTime << " s" << std::endl;
        std::cout << "[Stage 3] UNIFORM Octree build + compress: " << max_uniform    << " s" << std::endl;
        std::cout << "[Stage 4] STAGGER Octree build + compress: " << max_stagger    << " s" << std::endl;
        std::cout << "[TOTAL]                                    "
                  << (partitionTime + max_uniform + max_stagger)
                  << " s" << std::endl;
        std::cout << "==================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
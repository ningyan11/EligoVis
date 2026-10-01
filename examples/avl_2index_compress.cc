#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <random>
#include <adios2.h> 
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
#include <mpi.h>
#include <zstd.h>


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

void saveBigBlockMinMax(const std::string& filename, 
                        const std::vector<std::vector<std::vector<ScidxInterval<double>>>>& all_steps_block_interval_second) 
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    size_t numSteps = all_steps_block_interval_second.size();
    for (size_t step = 0; step < numSteps; ++step) {
        size_t numBigBlocks = all_steps_block_interval_second[step].size();
        for (size_t big = 0; big < numBigBlocks; ++big) {
            double minVal = std::numeric_limits<double>::max();
            double maxVal = std::numeric_limits<double>::lowest();
            for (const auto& interval : all_steps_block_interval_second[step][big]) {
                minVal = std::min(minVal, interval.low);
                maxVal = std::max(maxVal, interval.high);
            }
            outFile.write(reinterpret_cast<const char*>(&minVal), sizeof(double));
            outFile.write(reinterpret_cast<const char*>(&maxVal), sizeof(double));
        }
    }
    outFile.close();
}

/*void saveTreeStructureToByte(const std::vector<int>& fullTreeStructure, const std::string& treeID) {
    std::string filename = treeID + "-treeStructure";
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    size_t structureSize = fullTreeStructure.size();
    outFile.write(reinterpret_cast<const char*>(&structureSize), sizeof(size_t));
    uint8_t currentByte = 0;
    int bitIndex = 0;
    for (int bit : fullTreeStructure) {
        currentByte |= (bit & 1) << bitIndex;
        bitIndex++;
        if (bitIndex == 8) {
            outFile.put(static_cast<char>(currentByte));
            currentByte = 0;
            bitIndex = 0;
        }
    }
    if (bitIndex > 0) {
        outFile.put(static_cast<char>(currentByte));
    }
    outFile.close();
}*/


//修改树结构的压缩，增加zstd
void saveTreeStructureToByte(const std::vector<int>& fullTreeStructure, const std::string& treeID) {
    std::string filename = treeID + "-treeStructure";
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // 存 structureSize
    size_t structureSize = fullTreeStructure.size();
    outFile.write(reinterpret_cast<const char*>(&structureSize), sizeof(size_t));

    // bit-packing 到缓冲区
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

    // Zstd 压缩
    size_t bound = ZSTD_compressBound(packedBytes.size());
    std::vector<uint8_t> compressed(bound);
    size_t compressedSize = ZSTD_compress(
        compressed.data(), bound,
        packedBytes.data(), packedBytes.size(),
        3);
    compressed.resize(compressedSize);

    // 存压缩后大小和数据
    outFile.write(reinterpret_cast<const char*>(&compressedSize), sizeof(size_t));
    outFile.write(reinterpret_cast<const char*>(compressed.data()), compressedSize);

    outFile.close();
}

int main(int argc, char *argv[])
{
    std::cout << "[DEBUG] Program started. " << std::endl;

    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::string inputFileName;
    std::string variableName;
    std::string variableType;
    size_t nDim = 0;

    size_t beginStepNum = 0;
    size_t endStepNum = 0;
    std::vector<size_t> targetSteps; 

    // block_shape: 读进来但不使用，仅为兼容脚本参数
    std::vector<size_t> blockShape;
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

    // 错位分块的半块大小
    std::vector<size_t> halfBlockShape(nDim);
    for (size_t i = 0; i < nDim; i++)
        halfBlockShape[i] = smallBlockShape[i] / 2;

    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);

    std::vector<std::vector<std::vector<ScidxInterval<double>>>> all_steps_uniform;
    std::vector<std::vector<std::vector<ScidxInterval<double>>>> all_steps_stagger;

    size_t step = 0;
    std::set<size_t> targetSet(targetSteps.begin(), targetSteps.end());

    // ============================================================
    // 新增：阶段1、2 累计计时变量
    // ============================================================
    double total_adios_read_time = 0.0;  // 阶段1: ADIOS 读取数据到内存
    double total_partition_time  = 0.0;  // 阶段2: 遍历+分块+min/max+构建interval列表

    std::vector<size_t> uniformBlockCountOnEachDim_global(nDim);
    std::vector<size_t> staggerBlockCountOnEachDim_global(nDim);
    bool dimsSaved = false;

    while (reader_engine.BeginStep() == adios2::StepStatus::OK)
    {
        if (!targetSet.empty() && targetSet.find(step) == targetSet.end()) 
        {
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

            // ============================================================
            // 新增：阶段1 - ADIOS 读取数据到内存（纯IO）
            // ============================================================
            double t_adios_start = MPI_Wtime();
            reader_engine.Get(var, varData.data(), adios2::Mode::Sync);
            double t_adios_end = MPI_Wtime();
            total_adios_read_time += (t_adios_end - t_adios_start);
            std::cout << "[Step " << step << "] [Stage1] ADIOS read: "
                      << (t_adios_end - t_adios_start) << " s" << std::endl;

            // ============================================================
            // 新增：阶段2 - 遍历所有元素，同时完成：
            //   计算坐标、判断块归属、更新min/max、构建interval列表
            // ============================================================
            double t_partition_start = MPI_Wtime();

            // 均匀分块参数
            std::vector<size_t> uniformBlockCountOnEachDim(nDim);
            size_t total_uniform_blocks = 1;
            for (size_t i = 0; i < nDim; i++)
            {
                uniformBlockCountOnEachDim[i] = (var.Shape()[i] + smallBlockShape[i] - 1) / smallBlockShape[i];
                total_uniform_blocks *= uniformBlockCountOnEachDim[i];
            }

            // 错位分块参数
            std::vector<size_t> staggerBlockCountOnEachDim(nDim);
            size_t total_stagger_blocks = 1;
            for (size_t i = 0; i < nDim; i++)
            {
                staggerBlockCountOnEachDim[i] = var.Shape()[i] / smallBlockShape[i] + 1;
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

            // 一次遍历，同时处理均匀块和错位块
            for (size_t p = 0; p < varElements; p++)
            {
                std::vector<size_t> global_id = positionToIndices(p, var.Shape());

                // 均匀块归属 + min/max
                std::vector<size_t> uniform_block_id(nDim);
                for (size_t i = 0; i < nDim; i++)
                    uniform_block_id[i] = global_id[i] / smallBlockShape[i];
                size_t uniform_pos = indicesToPosition(uniformBlockCountOnEachDim, uniform_block_id);
                uniform_mins[uniform_pos] = std::min(uniform_mins[uniform_pos], varData[p]);
                uniform_maxs[uniform_pos] = std::max(uniform_maxs[uniform_pos], varData[p]);

                // 错位块归属 + min/max
                std::vector<size_t> stagger_block_id(nDim);
                for (size_t i = 0; i < nDim; i++)
                {
                    if (global_id[i] < halfBlockShape[i])
                        stagger_block_id[i] = 0;
                    else
                        stagger_block_id[i] = 1 + (global_id[i] - halfBlockShape[i]) / smallBlockShape[i];
                }
                size_t stagger_pos = indicesToPosition(staggerBlockCountOnEachDim, stagger_block_id);
                stagger_mins[stagger_pos] = std::min(stagger_mins[stagger_pos], varData[p]);
                stagger_maxs[stagger_pos] = std::max(stagger_maxs[stagger_pos], varData[p]);
            }

            // 构建 interval 列表
            std::vector<ScidxInterval<double>> uniformIntervals;
            uniformIntervals.reserve(total_uniform_blocks);
            for (size_t b = 0; b < total_uniform_blocks; b++) {
                ScidxInterval<double> interval;
                interval.low  = uniform_mins[b];
                interval.high = uniform_maxs[b];
                uniformIntervals.push_back(interval);
            }

            std::vector<ScidxInterval<double>> staggerIntervals;
            staggerIntervals.reserve(total_stagger_blocks);
            for (size_t b = 0; b < total_stagger_blocks; b++) {
                ScidxInterval<double> interval;
                interval.low  = stagger_mins[b];
                interval.high = stagger_maxs[b];
                staggerIntervals.push_back(interval);
            }

            double t_partition_end = MPI_Wtime();
            total_partition_time += (t_partition_end - t_partition_start);
            std::cout << "[Step " << step << "] [Stage2] Partition + minmax + interval build: "
                      << (t_partition_end - t_partition_start) << " s" << std::endl;

            all_steps_uniform.push_back({uniformIntervals});
            all_steps_stagger.push_back({staggerIntervals});

            varData.clear();
            varData.shrink_to_fit();
        }

        reader_engine.EndStep();
        step++;
    }
    reader_engine.Close();

    // 新增：打印阶段1、2 汇总
    if (mpi_rank == 0) {
        std::cout << "[Stage1 Total] ADIOS read time (all steps): "
                  << total_adios_read_time << " s" << std::endl;
        std::cout << "[Stage2 Total] Partition + minmax + interval (all steps): "
                  << total_partition_time << " s" << std::endl;
    }

    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    std::string subDir_uniform = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + safeVarName + "_"
                               + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                               + std::to_string(extraValue) + "_newour_stagger1_index_mapping_upgrad/";

    std::string subDir_stagger = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + safeVarName + "_"
                               + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                               + std::to_string(extraValue) + "_newour_stagger2_index_mapping_upgrad/";

    if (mpi_rank == 0) {
        createDirectory(subDir_uniform);
        createDirectory(subDir_stagger);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double global_min = std::numeric_limits<double>::max();
    double global_max = std::numeric_limits<double>::lowest();

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

    double error_bound = relative_error_bound * (global_max - global_min);
    std::cout << "error_bound = " << error_bound << std::endl;

    saveBigBlockMinMax(subDir_uniform + "big_block_minmax", all_steps_uniform);
    saveBigBlockMinMax(subDir_stagger + "big_block_minmax", all_steps_stagger);

    size_t numSteps = all_steps_uniform.size();

    // ============================================================
    // 均匀分块 AVL 树索引构建（与原始代码完全一致的计时方式）
    // ============================================================
    if (mpi_rank == 0)
        std::cout << "\n=== Building UNIFORM index ===" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    double build_start_time_uniform = MPI_Wtime();

    // 新增：提前搭好listHighStorage的形状
    std::vector<std::vector<std::vector<double>>> listHighStorage(numSteps);
    for (size_t s = 0; s < numSteps; s++) {
        listHighStorage[s].resize(all_steps_uniform[s].size());
    }

    for (size_t s = 0; s < numSteps; ++s) {
        size_t actualStep = beginStepNum + s;
        size_t numBigBlocks = all_steps_uniform[s].size();

        for (size_t big = 0; big < numBigBlocks; ++big) {
            size_t task_id = s * numBigBlocks + big;
            if (task_id % mpi_size != mpi_rank) continue;

            auto start_time1 = std::chrono::high_resolution_clock::now();

            ScidxAVLIntervalTree<double> avlIntervalTree;
            std::vector<ScidxInterval<double>> intervals = all_steps_uniform[s][big];
            std::string treeID = subDir_uniform + std::to_string(actualStep) + "-" + std::to_string(big);

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

            std::cout << "skippedSameLowIntervals size: " << skippedSameLowIntervals.size() << std::endl;
            std::cout << "skippedZeroLowIntervals size: " << skippedZeroLowIntervals.size() << std::endl;

            std::vector<SkippedNode<double>> skippedAllIntervals;
            skippedAllIntervals.insert(skippedAllIntervals.end(),
                                       skippedZeroLowIntervals.begin(), skippedZeroLowIntervals.end());
            skippedAllIntervals.insert(skippedAllIntervals.end(),
                                       skippedSameLowIntervals.begin(), skippedSameLowIntervals.end());
            skippedSameLowIntervals.clear();
            skippedZeroLowIntervals.clear();

            if (skippedAllIntervals.empty()) {
                ScidxInterval<double> dummyInterval;
                dummyInterval.low = 0; dummyInterval.high = 0;
                skippedAllIntervals.emplace_back(dummyInterval, static_cast<size_t>(-1));
            }

            auto start_time2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = start_time2 - start_time1;
            std::cout << "[Rank " << mpi_rank << "] [UNIFORM][Time1]: build avl tree "
                      << elapsed2.count() << " seconds" << std::endl;

            std::vector<int> fullTreeStructure;
            std::vector<std::vector<ScidxAVLNode<double>*>> allLevels;
            levelOrderTraversal(avlIntervalTree.getRoot(), avlIntervalTree.getTreeHeight() + 1,
                                allLevels, fullTreeStructure);
            saveTreeStructureToByte(fullTreeStructure, treeID);
            computeOptimizedAVL(allLevels, error_bound, treeID, skippedAllIntervals);

            /*computeOptimizedAVL(allLevels, error_bound, treeID, skippedAllIntervals,
                             &listHighStorage[s][big]); */

            auto start_time3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed3 = start_time3 - start_time2;
            std::cout << "[Rank " << mpi_rank << "] [UNIFORM][Time all compress and io]: compress avl tree "
                      << elapsed3.count() << " seconds" << std::endl;
        }
    }

    double build_end_time_uniform = MPI_Wtime();
    double local_uniform = build_end_time_uniform - build_start_time_uniform;

    double max_uniform, sum_uniform;
    MPI_Reduce(&local_uniform, &max_uniform, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_uniform, &sum_uniform, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::cout << "Total UNIFORM build + compression time (max across ranks): "
                  << max_uniform << " seconds." << std::endl;
        std::cout << "Average UNIFORM build + compression time: "
                  << sum_uniform / mpi_size << " seconds." << std::endl;
    }

    // ============================================================
    // 错位分块 AVL 树索引构建（与原始代码完全一致的计时方式）
    // ============================================================
    if (mpi_rank == 0)
        std::cout << "\n=== Building STAGGER index ===" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    double build_start_time_stagger = MPI_Wtime();

    for (size_t s = 0; s < numSteps; ++s) {
        size_t actualStep = beginStepNum + s;
        size_t numBigBlocks = all_steps_stagger[s].size();

        for (size_t big = 0; big < numBigBlocks; ++big) {
            size_t task_id = s * numBigBlocks + big;
            if (task_id % mpi_size != mpi_rank) continue;

            auto start_time1 = std::chrono::high_resolution_clock::now();

            ScidxAVLIntervalTree<double> avlIntervalTree;
            std::vector<ScidxInterval<double>> intervals = all_steps_stagger[s][big];
            std::string treeID = subDir_stagger + std::to_string(actualStep) + "-" + std::to_string(big);

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

            std::cout << "skippedSameLowIntervals size: " << skippedSameLowIntervals.size() << std::endl;
            std::cout << "skippedZeroLowIntervals size: " << skippedZeroLowIntervals.size() << std::endl;

            std::vector<SkippedNode<double>> skippedAllIntervals;
            skippedAllIntervals.insert(skippedAllIntervals.end(),
                                       skippedZeroLowIntervals.begin(), skippedZeroLowIntervals.end());
            skippedAllIntervals.insert(skippedAllIntervals.end(),
                                       skippedSameLowIntervals.begin(), skippedSameLowIntervals.end());
            skippedSameLowIntervals.clear();
            skippedZeroLowIntervals.clear();

            if (skippedAllIntervals.empty()) {
                ScidxInterval<double> dummyInterval;
                dummyInterval.low = 0; dummyInterval.high = 0;
                skippedAllIntervals.emplace_back(dummyInterval, static_cast<size_t>(-1));
            }

            auto start_time2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = start_time2 - start_time1;
            std::cout << "[Rank " << mpi_rank << "] [STAGGER][Time1]: build avl tree "
                      << elapsed2.count() << " seconds" << std::endl;

            std::vector<int> fullTreeStructure;
            std::vector<std::vector<ScidxAVLNode<double>*>> allLevels;
            levelOrderTraversal(avlIntervalTree.getRoot(), avlIntervalTree.getTreeHeight() + 1,
                                allLevels, fullTreeStructure);
            saveTreeStructureToByte(fullTreeStructure, treeID);
            computeOptimizedAVL(allLevels, error_bound, treeID, skippedAllIntervals);

            /*computeOptimizedAVLForStaggerWithCrossPrediction(
            allLevels, error_bound, treeID, skippedAllIntervals,
            listHighStorage[s][big],
            uniformBlockCountOnEachDim_global, staggerBlockCountOnEachDim_global, nDim
        );*/


            auto start_time3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed3 = start_time3 - start_time2;
            std::cout << "[Rank " << mpi_rank << "] [STAGGER][Time all compress and io]: compress avl tree "
                      << elapsed3.count() << " seconds" << std::endl;
        }
    }

    double build_end_time_stagger = MPI_Wtime();
    double local_stagger = build_end_time_stagger - build_start_time_stagger;

    double max_stagger, sum_stagger;
    MPI_Reduce(&local_stagger, &max_stagger, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stagger, &sum_stagger, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::cout << "Total STAGGER build + compression time (max across ranks): "
                  << max_stagger << " seconds." << std::endl;
        std::cout << "Average STAGGER build + compression time: "
                  << sum_stagger / mpi_size << " seconds." << std::endl;

        std::cout << "\n========== TIME SUMMARY ==========" << std::endl;
        std::cout << "[Stage 1] ADIOS read:                    " << total_adios_read_time << " s" << std::endl;
        std::cout << "[Stage 2] Partition + minmax + interval: " << total_partition_time  << " s" << std::endl;
        std::cout << "[Stage 3] UNIFORM build + compress:      " << max_uniform           << " s" << std::endl;
        std::cout << "[Stage 4] STAGGER build + compress:      " << max_stagger           << " s" << std::endl;
        std::cout << "[TOTAL]                                  "
                  << (total_adios_read_time + total_partition_time + max_uniform + max_stagger)
                  << " s" << std::endl;
        std::cout << "==================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}




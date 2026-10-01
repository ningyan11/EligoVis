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

void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);  // 创建目录
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
}

//将一维数组中的值的position，多维数据的形状，得到对应的多维中的位置
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


// 计算所有大块的 min/max，并存储到文件
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
    //std::cout << "Saved big block min/max values to " << filename << std::endl;
}

// 将 fullTreeStructure 保存到文件(直接int存储)
void saveTreeStructureToFile(const std::string& treeID, const std::vector<int>& fullTreeStructure) {
    std::string filename = treeID + "_structure.dat"; // 生成文件名
    std::ofstream outFile(filename, std::ios::binary);

    if (!outFile) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // 先写入数组大小
    size_t size = fullTreeStructure.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

    // 写入数组内容
    outFile.write(reinterpret_cast<const char*>(fullTreeStructure.data()), size * sizeof(int));

    outFile.close();
    //std::cout << "Tree structure saved to " << filename << std::endl;
}

//保存TreeStructure（8个合成byte存储）
void saveTreeStructureToByte(const std::vector<int>& fullTreeStructure, const std::string& treeID) {
    std::string filename = treeID + "-treeStructure";
    std::ofstream outFile(filename, std::ios::binary);
    
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // **先存储结构大小 (size_t)，保证解码时知道有多少位**
    size_t structureSize = fullTreeStructure.size();
    outFile.write(reinterpret_cast<const char*>(&structureSize), sizeof(size_t));

    uint8_t currentByte = 0;
    int bitIndex = 0;

    for (int bit : fullTreeStructure) {
        currentByte |= (bit & 1) << bitIndex;
        bitIndex++;

        if (bitIndex == 8) { // **存满 8 bit 就写入文件**
            outFile.put(static_cast<char>(currentByte));
            currentByte = 0;
            bitIndex = 0;
        }
    }

    // **处理最后剩余的 bits**
    if (bitIndex > 0) {
        outFile.put(static_cast<char>(currentByte));
    }

    outFile.close();
}

/*// 导出 countMatrix 到 CSV 文件
void exportCountMatrixToCSV(const std::vector<std::vector<int>>& countMatrix, const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        // 遍历 countMatrix，并将其写入 CSV 文件
        for (const auto& row : countMatrix) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) {
                    file << ",";  // 添加逗号分隔
                }
            }
            file << "\n";  // 每一行之后换行
        }
        file.close();
        std::cout << "Count matrix exported to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}*/

/*// 计算每个 low bin 和 high bin 的索引，返回 std::pair
std::pair<int, int> getBinIndices(double low, double high, double lowMin, double lowMax, double highMin, double highMax, int lowBinCount, int highBinCount) {
    int lowBin = std::floor((low - lowMin) / (lowMax - lowMin) * lowBinCount);
    int highBin = std::floor((high - highMin) / (highMax - highMin) * highBinCount);

    // 防止索引越界
    if (lowBin >= lowBinCount) lowBin = lowBinCount - 1;
    if (lowBin < 0) lowBin = 0;

    if (highBin >= highBinCount) highBin = highBinCount - 1;
    if (highBin < 0) highBin = 0;

    return std::make_pair(lowBin, highBin);  // 返回 std::pair
}*/

/*// 计算条件概率和期望值并保存到 CSV 文件
template <typename T>
void calculateConditionalProbability(const std::vector<ScidxInterval<T>>& intervals, int lowBinCount, int highBinCount, const std::string& csvFile) {
    // 1. 计算 low 和 high 的最小值和最大值
    T lowMin = intervals[0].low, lowMax = intervals[0].low;
    T highMin = intervals[0].high, highMax = intervals[0].high;

    for (const auto& interval : intervals) {
        if (interval.low < lowMin) lowMin = interval.low;
        if (interval.low > lowMax) lowMax = interval.low;
        if (interval.high < highMin) highMin = interval.high;
        if (interval.high > highMax) highMax = interval.high;
    }

    // 2. 初始化 2D 条件概率矩阵，大小为 [lowBinCount][highBinCount]
    std::vector<std::vector<int>> countMatrix(lowBinCount, std::vector<int>(highBinCount, 0));
    std::vector<int> lowBinTotalCount(lowBinCount, 0);

    // 3. 遍历每个区间，计算其 low 和 high 所在的 bin
    for (const auto& interval : intervals) {
        std::pair<int, int> binIndices = getBinIndices(interval.low, interval.high, lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount);
        int lowBin = binIndices.first;
        int highBin = binIndices.second;
        std::cout << "Interval: low = " << interval.low << ", high = " << interval.high
          << ", lowBin = " << binIndices.first << ", highBin = " << binIndices.second << std::endl;

        countMatrix[lowBin][highBin]++;
        lowBinTotalCount[lowBin]++;
    }

    std::string filename = "count_matrix.csv";
    std::cout << "Exporting count matrix to file: " << filename << std::endl;
    exportCountMatrixToCSV(countMatrix, filename);    

    // 4. 计算条件概率和期望值
    std::vector<std::pair<int, double>> nonZeroBins;  // 用于存储有非零值的 bin

    for (int i = 0; i < lowBinCount; ++i) {
        if (lowBinTotalCount[i] == 0) continue;  // 如果该 low bin 没有元素，跳过

        double expectedValue = 0.0;
        for (int j = 0; j < highBinCount; ++j) {
            double conditionalProbability = static_cast<double>(countMatrix[i][j]) / lowBinTotalCount[i];
            expectedValue += conditionalProbability * j;
        }

        if (expectedValue > 0) {
            nonZeroBins.emplace_back(i, expectedValue);
        }
    }

    // 5. 保存 lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount 以及映射结果到 CSV 文件
    std::ofstream file(csvFile);
    if (file.is_open()) {
        // 写入前几行的统计信息
        file << "lowMin,lowMax,highMin,highMax,lowBinCount,highBinCount\n";
        file << lowMin << "," << lowMax << "," << highMin << "," << highMax << "," << lowBinCount << "," << highBinCount << "\n";

        // 写入 low bin -> high bin 的映射
        file << "Low Bin,Expected High Bin\n";
        for (const auto& bin : nonZeroBins) {
            file << bin.first << "," << bin.second << "\n";
        }

        file.close();
        std::cout << "CSV file saved as " << csvFile << std::endl;
    } else {
        std::cerr << "Unable to open file: " << csvFile << std::endl;
    }   

}*/

int main(int argc, char *argv[])
{
    std::cout << "[DEBUG] Program started. " << std::endl;
    //omp_set_nested(1); // 启用嵌套并行

    //auto totalStart = std::chrono::high_resolution_clock::now(); 
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

    std::vector<size_t> blockShape(nDim), smallBlockShape(nDim);

    double relative_error_bound = 1E-3;  // 默认相对误差

    size_t extraValue =0;

    // 解析命令行参数
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
        else if (arg == "--block_shape" && i + nDim < argc)
        {
            for (size_t j = 0; j < nDim; j++)
            {
                blockShape.push_back(std::stoul(argv[++i]));
            }
        }
        else if (arg == "--begin_step")
        {
            if (i + 1 < argc)
            {
                beginStepNum = atoi(argv[i + 1]);
            }
            else
            {
                std::cerr << "--begin_step option requires one argument." << std::endl;
                return 1;
            }
        }
        else if (arg == "--end_step")
        {
            if (i + 1 < argc)
            {
                endStepNum = atoi(argv[i + 1]);
            }
            else
            {
                std::cerr << "--end_step option requires one argument." << std::endl;
                return 1;
            }
        }
        else if (arg == "--target_steps")
        {
            while (i + 1 < argc && std::isdigit(argv[i + 1][0])) {
                targetSteps.push_back(std::stoul(argv[++i]));
            }
        }

        else if (arg == "--small_block_shape")
        {
            if (nDim)
            {
                if ((int)(i + nDim) < argc)
                {
                    for (size_t j = i + 1; j < i + 1 + nDim; j++)
                    {
                        smallBlockShape.push_back(atoi(argv[j]));
                    }
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

    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);

    std::vector<std::vector<std::vector<ScidxInterval<double>>>> all_steps_block_interval_second;

    std::vector<std::vector<double>> all_steps_block_mins;
    std::vector<std::vector<double>> all_steps_block_maxs;

    size_t step = 0;

    //auto readBegin_time = std::chrono::high_resolution_clock::now();

    std::set<size_t> targetSet(targetSteps.begin(), targetSteps.end());

    while (reader_engine.BeginStep() == adios2::StepStatus::OK)
    {
        size_t total_blocks = 1;
        //固定step才读取
        if (!targetSet.empty() && targetSet.find(step) == targetSet.end()) 
        {
            reader_engine.EndStep();
            step++;
            continue;
        }
        auto var = reader_io.InquireVariable<double>(variableName);
        variableType= reader_io.VariableType(variableName);
       
        size_t varElements = 1;
        for (size_t i = 0; i < nDim; i++)
        {
            varElements *= var.Shape()[i];
        }

        std::vector<size_t> blockCountOnEachDim;

        for (size_t i = 0; i < nDim; i++)
        {
            blockCountOnEachDim.push_back(var.Shape()[i]/blockShape[i]);
            //总的大块数量
            total_blocks *= var.Shape()[i]/blockShape[i];
        }

        if (variableType == "double")
        {
            std::vector<double> varData(varElements);
            //从adios中取得所有数据
            reader_engine.Get(var, varData.data(), adios2::Mode::Sync);
            
            std::vector<std::vector<double>> blocks(total_blocks);

            //处理每一个点，将对应的点放入对应的大块中去
            for (size_t p = 0; p < varElements; p++)
            {
                std::vector<size_t> elem_global_id = positionToIndices(p, var.Shape());
                std::vector<size_t> block_global_id(nDim);
                for (size_t i = 0; i < nDim; i++)
                {
                    block_global_id[i] = elem_global_id[i]/blockShape[i];
                }   
                             
                size_t block_position = indicesToPosition(blockCountOnEachDim, block_global_id);
                

                // Ensure block_position is within bounds
                if (block_position >= total_blocks) {
                    std::cerr << "Error: block_position out of bounds. block_position = " << block_position << ", total_blocks = " << total_blocks << std::endl;
                    reader_engine.EndStep();
                    return 1;
                }
                blocks[block_position].push_back(varData[p]);
            }

            std::vector<double> block_mins(total_blocks, std::numeric_limits<double>::max());
            std::vector<double> block_maxs(total_blocks, std::numeric_limits<double>::min());

            // 计算每个大块的最小值和最大值
            for (size_t b = 0; b < total_blocks; b++)
            {
                if (!blocks[b].empty())
                {
                    auto minmax = std::minmax_element(blocks[b].begin(), blocks[b].end());
                    block_mins[b] = *minmax.first;
                    block_maxs[b] = *minmax.second;
                }
            }

            //将每个step的大块最小值和最大值放入all_steps_block_mins，all_steps_block_maxs
            all_steps_block_mins.push_back(block_mins);
            all_steps_block_maxs.push_back(block_maxs);

            // 用于存储一个step中多个大块中的小块的interval
            std::vector<std::vector<ScidxInterval<double>>> one_steps_block_interval_second;

            //进行每个大块中的处理
            for (size_t b = 0; b < total_blocks; b++)
            {
                //std::cout << "blocks of big " << b << std::endl;
                //大块对应的一维id,对应的块
                std::vector<size_t> block_idx_on_each_dim = positionToIndices(b, blockCountOnEachDim);
                std::vector<size_t> blockStart(nDim);
                std::vector<size_t> blockCount(nDim);
                /*std::cout << "block " << b << " start: ";
                for (size_t i = 0; i < nDim; i++)
                {
                    blockStart[i] = block_idx_on_each_dim[i]*blockShape[i];
                    std::cout << blockStart[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "block " << b << " count: ";*/
                for (size_t i = 0; i < nDim; i++)
                {
                    if (block_idx_on_each_dim[i] == blockCountOnEachDim[i]-1)
                    {
                        blockCount[i] = var.Shape()[i]-block_idx_on_each_dim[i]*blockShape[i];
                    }
                    else
                    {
                        blockCount[i] = blockShape[i];
                    }
                    //std::cout << blockCount[i] << " ";
                }
                //std::cout << std::endl;

                //change for the second layer
                size_t blockSize = blocks[b].size();
                std::vector<double> blockData = blocks[b];

                std::vector<size_t> smallBlockCountOnEachDim(nDim);
                size_t total_small_blocks = 1;
                for (size_t i = 0; i < nDim; i++)
                {
                    smallBlockCountOnEachDim[i] = (blockCount[i] + smallBlockShape[i] - 1) / smallBlockShape[i];
                    total_small_blocks *= smallBlockCountOnEachDim[i];
                }

                std::vector<double> small_block_mins(total_small_blocks, std::numeric_limits<double>::max());
                std::vector<double> small_block_maxs(total_small_blocks, std::numeric_limits<double>::min());     
                
                //std::vector<std::vector<double>> smallBlocks(total_small_blocks);
                //大块中小块的处理，blockSize大块中全部的数据
                for (size_t p = 0; p < blockSize; p++)
                {
                    std::vector<size_t> elem_local_id = positionToIndices(p, blockCount);
                    std::vector<size_t> small_block_local_id(nDim);
                    for (size_t i = 0; i < nDim; i++)
                    {
                        small_block_local_id[i] = elem_local_id[i] / smallBlockShape[i];
                    }
                    size_t small_block_position = indicesToPosition(smallBlockCountOnEachDim, small_block_local_id);
                    
                    //某个大块中对应的小块的position：small_block_position（好多个点会在同一个块的position
                    //smallBlocks[small_block_position].push_back(blockData[p]);

                    //计算同一个大块对应的同一个小块small_block_position，全部点的最大值，最小值
                    small_block_mins[small_block_position] = std::min(small_block_mins[small_block_position], blockData[p]);
                    small_block_maxs[small_block_position] = std::max(small_block_maxs[small_block_position], blockData[p]);
                }
                    
                //一个step,一个大块中全部小块的interval（min,max转换成interval）
                std::vector<ScidxInterval<double>> blockIntervals;   
                // 每一个step,每一个大块，每一个小块。将每个小块的区间插入到interval
                for (size_t sb = 0; sb < total_small_blocks; sb++)
                    {
                        ScidxInterval<double> interval;
                        interval.low = small_block_mins[sb];
                        interval.high = small_block_maxs[sb];
                        blockIntervals.push_back(interval);
                    }

                //同一个step中全部大块对应全部小块构建的interval
                one_steps_block_interval_second.push_back(blockIntervals); 
                    
                // 释放 small_block_mins 和 small_block_maxs 内存
                small_block_mins.clear();
                small_block_mins.shrink_to_fit();
                small_block_maxs.clear();
                small_block_maxs.shrink_to_fit();  
            }

            //全部step，对应的interval
            all_steps_block_interval_second.push_back(one_steps_block_interval_second);  

            // 释放 blocks 内存
            for (auto& block : blocks) {
                block.clear();
                block.shrink_to_fit();
            }

            // 释放 varData 内存
            varData.clear();
            varData.shrink_to_fit();     
        }

        reader_engine.EndStep();
        step++;
    }
    reader_engine.Close();

    //auto readEnd_time = std::chrono::high_resolution_clock::now();

    //std::chrono::duration<double> elapsed1 = readEnd_time - totalStart;

    //std::cout << "time to read all data and save as blocks " << elapsed1.count() << " seconds" << std::endl;

    //std::cout << "[Rank " << mpi_rank << "] time to read all data and save as blocks " << elapsed1.count() << " seconds" << std::endl;

    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string subDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" +safeVarName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_newour_index_ddefault/";

    //std::string subDir = baseDir + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_index/";

    //避免并行
    if (mpi_rank == 0) {
        createDirectory(subDir);
    }
    MPI_Barrier(MPI_COMM_WORLD); // 等待目录创建完成

    double global_min = std::numeric_limits<double>::max();
    double global_max = std::numeric_limits<double>::lowest();

    for (const auto& step_blocks : all_steps_block_interval_second) {
        for (const auto& big_block : step_blocks) {
            for (const auto& interval : big_block) {
                global_min = std::min(global_min, interval.low);
                global_max = std::max(global_max, interval.high);
            }
        }
    }

    double error_bound = relative_error_bound * (global_max - global_min);
    std::cout << "error_bound = " << error_bound << std::endl;

    // 存储所有大块的 min/max
    std::string filename = subDir + "big_block_minmax";
    //按step的顺序依次存储
    saveBigBlockMinMax(filename, all_steps_block_interval_second);

    //double error_bound = 1E-3;
    size_t numSteps = all_steps_block_interval_second.size();

    //double start_time = omp_get_wtime();

    MPI_Barrier(MPI_COMM_WORLD);  // 保证计时公平性
    double build_start_time = MPI_Wtime();

    //#pragma omp parallel for schedule(dynamic)
    for (size_t step = 0; step < numSteps; ++step) {
        size_t actualStep = beginStepNum + step; // 计算真实的 step
        size_t numBigBlocks = all_steps_block_interval_second[step].size();

        //#pragma omp parallel for schedule(dynamic)
        for (size_t big = 0; big < numBigBlocks; ++big) {
            size_t task_id = step * numBigBlocks + big;
            if (task_id % mpi_size != mpi_rank) continue; 
    
            auto start_time1 = std::chrono::high_resolution_clock::now();

           ScidxAVLIntervalTree<double> avlIntervalTree;
           std::vector<ScidxInterval<double>> intervals = all_steps_block_interval_second[step][big];
           

           std::string treeID = subDir + std::to_string(actualStep) + "-" + std::to_string(big);

           /*std::string csvFileName = treeID + "-low_high_bin_mapping.csv";

           calculateConditionalProbability(intervals, 100, 100, csvFileName);*/

           //因为min值相同而未被插入的interval
           std::vector<SkippedNode<double>> skippedSameLowIntervals;

           //std::vector<ScidxInterval<double>> zeroLowIntervalList;

           std::vector<SkippedNode<double>> skippedZeroLowIntervals;

            for (size_t i = 0; i < intervals.size(); i++) {
                //std::cout << i << " [" << intervals[i].low << " " << intervals[i].high << "]" << std::endl;
                //这里处理了，所以树中不会存在low等于0的节点
                if (intervals[i].low == 0) {
                   
                    skippedZeroLowIntervals.push_back(SkippedNode<double>(intervals[i], i));

                    continue;
                }

                avlIntervalTree.insertNode(i, intervals[i], skippedSameLowIntervals);
            }
            //std::cout << "AVL Interval Tree after insertions:" << std::endl;

            //avlIntervalTree.display();

            //整个树全部low都为0的特殊修复
            if (avlIntervalTree.getRoot() == nullptr) {
                ScidxInterval<double> dummyInterval;
                dummyInterval.low = 0;
                dummyInterval.high = 0;
            
                std::vector<SkippedNode<double>> dummySkipped;
                //id用-1来表示
                size_t dummyId = static_cast<size_t>(-1);  
            
                avlIntervalTree.insertNode(dummyId, dummyInterval, dummySkipped);
            }
            
            

            // 打印 skippedSameLowIntervals 的 id
            std::cout << "skippedSameLowIntervals size: " << skippedSameLowIntervals.size() << std::endl;
            std::cout << "skippedZeroLowIntervals size: " << skippedZeroLowIntervals.size() << std::endl;
          
            /*for (const auto& node : skippedSameLowIntervals) {
                std::cout << " skippedSameLowIntervals id: " << node.id << std::endl;
            }

            // 打印 skippedZeroLowIntervals 的 id
              for (const auto& node : skippedZeroLowIntervals) {
                std::cout << "skippedZeroLowIntervals id: " << node.id << std::endl;
            }*/

            //just for test
            /*if (step == 0 && big == 0) {
                std::cout << "[Step 0, Big Block 0] Displaying AVL Interval Tree:" << std::endl;
                avlIntervalTree.display();
            }*/

            std::vector<SkippedNode<double>> skippedAllIntervals;

        
            skippedAllIntervals.insert(skippedAllIntervals.end(),
                                    skippedZeroLowIntervals.begin(),
                                    skippedZeroLowIntervals.end());

            skippedAllIntervals.insert(skippedAllIntervals.end(),
                                    skippedSameLowIntervals.begin(),
                                    skippedSameLowIntervals.end());

            skippedSameLowIntervals.clear();
            skippedZeroLowIntervals.clear();

            if (skippedAllIntervals.empty()) {
                // 构造一个虚拟 skipped 节点，ID 用 -1，min 和 max 为 0
                ScidxInterval<double> dummyInterval;
                dummyInterval.low = 0;
                dummyInterval.high = 0;
            
                skippedAllIntervals.emplace_back(dummyInterval, static_cast<size_t>(-1));
            }
            

            auto start_time2 = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed2 = start_time2 - start_time1;

            //std::cout << "build avl tree " << elapsed2.count() << " seconds" << std::endl;

            std::cout << "[Rank " << mpi_rank << "] [Time1]: build avl tree " << elapsed2.count() << " seconds" << std::endl;

            int treeHeight = avlIntervalTree.getTreeHeight();
            //std::cout << "Tree height after insertions: " << treeHeight << std::endl;

            //记录整个tree的结构, 0/1表示，用以恢复树结构
            std::vector<int> fullTreeStructure; 
            
            //记录整个树的层级节点，用以后续压缩
            std::vector<std::vector<ScidxAVLNode<double>*>> allLevels;
            

            levelOrderTraversal(avlIntervalTree.getRoot(), avlIntervalTree.getTreeHeight() + 1, allLevels, fullTreeStructure);

            

            saveTreeStructureToByte(fullTreeStructure, treeID);

            //当前tree的压缩
            computeOptimizedAVL(allLevels, error_bound, treeID, skippedAllIntervals);

            auto start_time3 = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed3 = start_time3 - start_time2;

            //std::cout << "compress avl tree " << elapsed3.count() << " seconds" << std::endl;

            std::cout << "[Rank " << mpi_rank << "] [Time all compress and io]: compress avl tree " << elapsed3.count() << " seconds" << std::endl;

        }
    }
     //double end_time = omp_get_wtime();
     //std::cout << "Total AVL tree build and compress time = " << (end_time - start_time) << " seconds." << std::endl;
    double build_end_time = MPI_Wtime();
    double local_build_time = build_end_time - build_start_time;

    double max_build_time, sum_build_time;
    MPI_Reduce(&local_build_time, &max_build_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_build_time, &sum_build_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::cout << "Total build + compression time (max across ranks): "
                << max_build_time << " seconds." << std::endl;
        std::cout << "Average build + compression time: "
                << sum_build_time / mpi_size << " seconds." << std::endl;
    }
     
    MPI_Finalize(); 
    return 0;
}

            


    









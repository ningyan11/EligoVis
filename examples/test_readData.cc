
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
#include <utility>  // for std::pair
#include <fstream> 

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());


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

// Function to generate a random interval
template <typename T>
ScidxInterval<T> generateRandomInterval(T maxLow, T maxHigh) {
    std::uniform_real_distribution<T> distLow(0, maxLow);
    std::uniform_real_distribution<T> distHigh(distLow(gen), maxHigh);

    ScidxInterval<T> interval;
    interval.low = distLow(gen);
    interval.high = distHigh(gen);

    return interval;
}

template <typename T>
bool containsDuplicateValues(const std::vector<T>& vec) {
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        auto count = std::count(vec.begin(), vec.end(), *it);
        if (count > 1) {
            return true;
        }
    }
    return false;
}

void listAllVariables(adios2::IO &io) {
    std::cout << "Listing all variables in the file:" << std::endl;
    for (const auto &variablePair : io.AvailableVariables()) {
        std::cout << "Variable Name: " << variablePair.first << std::endl;
        for (const auto &attributePair : variablePair.second) {
            std::cout << "  Attribute: " << attributePair.first << " = " << attributePair.second << std::endl;
        }
    }
}

template <typename T>
ScidxAVLNode<T>* attachSubTreesBFS(const std::vector<bool>& fullTreeVectorOfMap, const std::vector<ScidxAVLNode<T>*>& compressedSubTrees);

std::pair<int, int> getBinIndices(double low, double high, double lowMin, double lowMax, double highMin, double highMax, int lowBinCount, int highBinCount);

void writeIntervalsToCSV(const std::vector<ScidxInterval<double>>& intervals, std::ofstream& outfile);


template <typename T>
void calculateConditionalProbability(const std::vector<ScidxInterval<T>>& intervals, int lowBinCount, int highBinCount, const std::string& csvFile);

void exportCountMatrixToCSV(const std::vector<std::vector<int>>& countMatrix, const std::string& filename);


int main(int argc, char *argv[]) {
    
    std::string inputFileName;
    std::string variableName;
    std::string variableType;
    size_t nDim = 0;
    size_t beginStepNum = 0;
    size_t endStepNum = 0;
    std::vector<size_t> blockShape(nDim), smallBlockShape(nDim);

    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--input_file")
        {
            if (i+1 < argc)
            {
                inputFileName = argv[i+1];
            }
            else
            {
                std::cerr << "--input_file option requires one argument." << std::endl;
                return 1;
            }            
        }
        else if (arg == "--variable_name")
        {
            if (i+1 < argc)
            {
                variableName = argv[i+1];
            }
            else
            {
                std::cerr << "--variable_name option requires one argument." << std::endl;
                return 1;
            }             
        }

        else if (arg == "--dimensions")
        {
            if (i + 1 < argc)
            {
                nDim = atoi(argv[i + 1]);
            }
            else
            {
                std::cerr << "--dimensions option requires one argument." << std::endl;
                return 1;
            }
        }
        else if (arg == "--big_block_shape")
        {
            if (nDim)
            {
                if ((int)(i + nDim) < argc)
                {
                    for (size_t j = i + 1; j < i + 1 + nDim; j++)
                    {
                        blockShape.push_back(atoi(argv[j]));
                    }
                }
                else
                {
                    std::cerr << "--block_shape option requires [# of dimensions] argument." << std::endl;
                    return 1;
                }
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
    }


    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);

    size_t steps = reader_engine.Steps();
    std::cout << "total steps: " << steps << std::endl;


    // 输出文件名
    std::string filename = std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_intervals4.csv";
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return 1;
    }

    adios2::IO writer_io = adios.DeclareIO("WriterIO");
    adios2::Engine writer_engine = writer_io.Open("BigBlock.bp", adios2::Mode::Write); 

    writer_engine.BeginStep();


    // 用于存储每个大块的小块区间
    std::vector<std::vector<ScidxInterval<double>>> all_intervals; 
    //vector用于存储每个step中,每个大块的最小值和最大值
    std::vector<std::vector<double>> all_steps_block_mins;
    std::vector<std::vector<double>> all_steps_block_maxs;

    std::vector<std::vector<std::vector<ScidxInterval<double>>>> all_steps_block_interval_second;

    size_t step = 0;
    while (reader_engine.BeginStep() == adios2::StepStatus::OK)
    {   
        
        if (step < beginStepNum) {
            reader_engine.EndStep();
            step++;
            continue;
        }
        std::cout << "Current step aaa: " << step << std::endl;
        size_t total_blocks = 1;
        
        auto var = reader_io.InquireVariable(variableName);
        variableType= reader_io.VariableType(variableName);

        size_t varElements = 1;
        std::cout <<  " variable name: " << variableName << ", type: " << variableType << std::endl;
        
        //varElements总共数据个数
        for (size_t i = 0; i < nDim; i++)
        {
            std::cout << var.Shape()[i] << " ";
            varElements *= var.Shape()[i];
        }
        std::cout << std::endl;


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
            std::vector<double> block_maxs(total_blocks, std::numeric_limits<double>::lowest());

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

            std::cout << "total_blocks " << total_blocks << std::endl;

            // 用于存储一个step中多个大块中的小块的interval
            std::vector<std::vector<ScidxInterval<double>>> one_steps_block_interval_second;

            //进行每个大块中的处理
            for (size_t b = 0; b < total_blocks; b++)
            {
                std::cout << "blocks of big " << b << std::endl;
                //大块对应的一维id,对应的块
                std::vector<size_t> block_idx_on_each_dim = positionToIndices(b, blockCountOnEachDim);
                std::vector<size_t> blockStart(nDim);
                std::vector<size_t> blockCount(nDim);
                std::cout << "block " << b << " start: ";
                for (size_t i = 0; i < nDim; i++)
                {
                    blockStart[i] = block_idx_on_each_dim[i]*blockShape[i];
                    std::cout << blockStart[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "block " << b << " count: ";
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
                    std::cout << blockCount[i] << " ";
                }
                std::cout << std::endl;


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
                std::vector<double> small_block_maxs(total_small_blocks, std::numeric_limits<double>::lowest());     
                
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
        std::cout << "Completed step " << std::endl;
        step++;  
        std::cout << "Current step bbbb: " << step << std::endl;  

        if (step > endStepNum) {
            break;
        }

    }


    


    // 确定有多少个 big block
    size_t numBigs = all_steps_block_interval_second[0].size();
    
    // 创建合并后的vector（合并多个step中，同一个大块对应的全部interval
    std::vector<std::vector<ScidxInterval<double>>> mergedBigs(numBigs);

    // 遍历每个 step
    for (const auto& step : all_steps_block_interval_second) {
        // 遍历每个 big block
        for (size_t i = 0; i < numBigs; ++i) {
            mergedBigs[i].insert(mergedBigs[i].end(), step[i].begin(), step[i].end());
        }
    }


    //计算多个step中同一个大块的最大最小值
    //all_steps_block_mins/maxs,每个大块的最小值和最大值
    size_t maxLength = all_steps_block_mins[0].size();
    std::vector<double> minValuesBig(maxLength, std::numeric_limits<double>::max());
    std::vector<double> maxValuesBig(maxLength, std::numeric_limits<double>::lowest());
    

    // min
    for (const auto& innerVector : all_steps_block_mins) {
        for (size_t i = 0; i < innerVector.size(); ++i) {
            minValuesBig[i] = std::min(minValuesBig[i], innerVector[i]);
        }
    }

    // max
    for (const auto& innerVector : all_steps_block_maxs) {
        for (size_t i = 0; i < innerVector.size(); ++i) {
            maxValuesBig[i] = std::max(maxValuesBig[i], innerVector[i]);
            
        }
    }


    // 将每个大块的最小值和最大值写入 ADIOS
    adios2::Variable<double> blockMinVariable = writer_io.DefineVariable<double>("firstLayer_low", {});
    adios2::Variable<double> blockMaxVariable = writer_io.DefineVariable<double>("firstLayer_high", {});
    writer_engine.Put(blockMinVariable, minValuesBig.data(), adios2::Mode::Sync);
    writer_engine.Put(blockMaxVariable, maxValuesBig.data(), adios2::Mode::Sync);
    writer_engine.Close();
   

   
    for (size_t k = 0; k < mergedBigs.size(); ++k) {
       
        std::vector<ScidxInterval<double>> intervals = mergedBigs[k];
       
        writeIntervalsToCSV(intervals, outfile);
    }
    outfile.close();
    return 0;


}

// 计算每个 low bin 和 high bin 的索引，返回 std::pair
std::pair<int, int> getBinIndices(double low, double high, double lowMin, double lowMax, double highMin, double highMax, int lowBinCount, int highBinCount) {
    int lowBin = std::floor((low - lowMin) / (lowMax - lowMin) * lowBinCount);
    int highBin = std::floor((high - highMin) / (highMax - highMin) * highBinCount);

    // 防止索引越界
    if (lowBin >= lowBinCount) lowBin = lowBinCount - 1;
    if (lowBin < 0) lowBin = 0;

    if (highBin >= highBinCount) highBin = highBinCount - 1;
    if (highBin < 0) highBin = 0;

    return std::make_pair(lowBin, highBin);  // 返回 std::pair
}

// 计算条件概率和期望值并保存到 CSV 文件
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

  
    

}

// 导出 countMatrix 到 CSV 文件
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
}


void writeIntervalsToCSV(const std::vector<ScidxInterval<double>>& intervals, std::ofstream& outfile) {
    if (outfile.is_open()) {
        // 遍历每个 interval，并将其 low 和 high 值写入到文件中
        for (const auto& interval : intervals) {
            outfile << interval.low << "," << interval.high << "\n";
        }
    } else {
        std::cerr << "Unable to write to the file." << std::endl;
    }
}
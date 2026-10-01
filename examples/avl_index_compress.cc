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
    std::cout << "Saved big block min/max values to " << filename << std::endl;
}


// 将 fullTreeStructure 保存到文件
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
    std::cout << "Tree structure saved to " << filename << std::endl;
}





int main(int argc, char *argv[])
{
    std::string inputFileName;
    std::string variableName;
    std::string variableType;
    size_t nDim = 0;

    size_t beginStepNum = 0;
    size_t endStepNum = 0;

    std::vector<size_t> blockShape(nDim), smallBlockShape(nDim);


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
    }

    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);

    std::vector<std::vector<std::vector<ScidxInterval<double>>>> all_steps_block_interval_second;

    std::vector<std::vector<double>> all_steps_block_mins;
    std::vector<std::vector<double>> all_steps_block_maxs;

    size_t step = 0;

    while (reader_engine.BeginStep() == adios2::StepStatus::OK)
    {
        size_t total_blocks = 1;
        if (step < beginStepNum) {
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
        if (step > endStepNum) {
            break;
        }
    }
    reader_engine.Close();

    std::string baseDir = "/home/nyan/scidx/scidx/";
 
    std::string subDir = baseDir + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_index/";

    createDirectory(subDir);


    // 存储所有大块的 min/max
    std::string filename = subDir + "big_block_minmax";
    //按step的顺序依次存储
    saveBigBlockMinMax(filename, all_steps_block_interval_second);

    double error_bound = 1E-3;
    size_t numSteps = all_steps_block_interval_second.size();
    for (size_t step = 0; step < numSteps; ++step) {
        size_t actualStep = beginStepNum + step; // 计算真实的 step
        size_t numBigBlocks = all_steps_block_interval_second[step].size();
        for (size_t big = 0; big < numBigBlocks; ++big) {
            ScidxAVLIntervalTree<double> avlIntervalTree;
           std::vector<ScidxInterval<double>> intervals = all_steps_block_interval_second[step][big];
           std::vector<ScidxInterval<double>> zeroLowIntervalList;
           std::vector<SkippedNode<double>> skippedIntervals;

            for (size_t i = 0; i < intervals.size(); i++) {
                //std::cout << i << " [" << intervals[i].low << " " << intervals[i].high << "]" << std::endl;
                if (intervals[i].low == 0) {
                    zeroLowIntervalList.push_back(intervals[i]);
                    continue;
                }

                avlIntervalTree.insertNode(i, intervals[i], skippedIntervals);
            }
            std::cout << "AVL Interval Tree after insertions:" << std::endl;
            avlIntervalTree.display();

            int treeHeight = avlIntervalTree.getTreeHeight();
            std::cout << "Tree height after insertions: " << treeHeight << std::endl;

            std::string treeID = subDir + std::to_string(actualStep) + "-" + std::to_string(big);



            //记录整个tree的结构, 0/1表示，用以恢复树结构
            std::vector<int> fullTreeStructure; 
            
            //记录整个树的层级节点，用以后续压缩
            std::vector<std::vector<ScidxAVLNode<double>*>> allLevels;
            

            levelOrderTraversal(avlIntervalTree.getRoot(), avlIntervalTree.getTreeHeight() + 1, allLevels, fullTreeStructure);

            //当前tree的压缩
            compressNaiveAVL(allLevels, error_bound, treeID);

        }
    }
     return 0;
}

            




    








#include <vector>
#include <queue>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>

#include <random>

#include <scidx.h>
#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <zstd.h>

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());

// Function to generate a random interval
template <typename T>
ScidxrbInterval<T> generateRandomInterval(T maxLow, T maxHigh) {
    std::uniform_real_distribution<T> distLow(0, maxLow);
    std::uniform_real_distribution<T> distHigh(distLow(gen), maxHigh);

    ScidxrbInterval<T> rbInterval;
    rbInterval.low = distLow(gen);
    rbInterval.high = distHigh(gen);

    return rbInterval;
}

void printIntervalTreeArray(std::vector<int>& arr);
size_t convertIntArray2ByteArray_fast_1b(const std::vector<int>& intArray, std::vector<unsigned char>& result);
void saveToFile(const std::vector<std::vector<float>>& data, const std::string& filename);



int main(int argc, char *argv[]) {

    char *inputFileName = NULL;
    size_t nDim = 0;
    std::vector<size_t> dataShape;
    std::vector<size_t> blockShape;
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
                std::cerr << "--input option requires one argument." << std::endl;
                return 1;
            }            
        }
        else if (arg == "--dimensions")
        {
            if (i+1 < argc)
            {
                std::stringstream ss_dim(argv[i+1]);
                ss_dim >> nDim;
            }
            else
            {
                std::cerr << "--dimensions option requires one argument." << std::endl;
                return 1;
            } 
        }
        else if (arg == "--data_shape")
        {
            if (nDim)
            {
                if ((int)(i+nDim) < argc)
                {
                    for (size_t j = i+1; j < i+1+nDim; j++)
                    {
                        dataShape.push_back(atoi(argv[j]));
                    }
                    
                }
                else
                {
                    std::cerr << "--data_shape option requires [# of dimensions] argument." << std::endl;
                    return 1;
                }  
            }
            else
            {
                std::cerr << "# of dimensions must be greater than 0." << std::endl;
                return 1;                
            }
        }
        else if (arg == "--block_shape")
        {
            if (nDim)
            {
                if ((int)(i+nDim) < argc)
                {
                    for (size_t j = i+1; j < i+1+nDim; j++)
                    {
                        blockShape.push_back(atoi(argv[j]));
                    }
                    
                }
                else
                {
                    std::cerr << "--blockShape option requires [# of dimensions] argument." << std::endl;
                    return 1;
                }  
            }
        }
        
    }

    size_t nElem;
    int status;
    float *dataBuffer = scidx_readFloatData(inputFileName, &nElem, &status);

    std::cout << "read in " << nElem << " data elements." << std::endl;

    std::vector<float> data{dataBuffer, dataBuffer+nElem};

    std::vector<std::vector<float>> blockMinMax = obtainBlockMinMax(data, dataShape, blockShape);
    
    saveToFile(blockMinMax, "outputOfPoints.txt");
	
    std::vector<ScidxrbInterval<float>> intervals;

    float global_min = 0;
    float global_max = 0;
    for (size_t i = 0; i < blockMinMax.size(); i++)
    {
        if (blockMinMax[i][0] < global_min)
        {
            global_min = blockMinMax[i][0];
        }
        if (blockMinMax[i][1] > global_max)
        {
            global_max = blockMinMax[i][1];
        }        
        
        ScidxrbInterval<float> interval;
        interval.low = blockMinMax[i][0];
        interval.high = blockMinMax[i][1];
        intervals.push_back(interval);
    }
    

    ScidxRedBlackIntervalTree<float> rbIntervalTree;

    for (size_t i = 0; i < intervals.size(); i++)
    {
        std::cout << intervals[i].low << " " << intervals[i].high << std::endl;
        rbIntervalTree.insert(intervals[i], i);
        
    }

    std::cout << "RB Interval Tree after insertions:" << std::endl;
    rbIntervalTree.display();


    //put the intervalTree into the int array
    std::vector<int> resultArray;
    ScidxrbNode<float>* rbIntervalTreeRoot = rbIntervalTree.getRoot();
    convertTreeToArray(rbIntervalTreeRoot, resultArray);
    printIntervalTreeArray(resultArray);

    std::vector<unsigned char> byteArray;
    size_t byteLength = convertIntArray2ByteArray_fast_1b(resultArray, byteArray);
    std::cout << "Byte Array Length: " << byteLength << std::endl;

    int levelsToTraverse = 20;

    std::vector<std::vector<std::vector<ScidxrbNode<float>*>>> allSubTrees;
    std::vector<std::vector<ScidxrbNode<float>*>> firstSubTreeNodesInLevels;

    levelOrderTraversal(rbIntervalTree.getRoot(), levelsToTraverse, firstSubTreeNodesInLevels);
    allSubTrees.push_back(firstSubTreeNodesInLevels);

    std::vector<ScidxrbNode<float>*>& lastLevelOfFirstSubTree = firstSubTreeNodesInLevels.back();

    std::vector<ScidxrbNode<float>*> rootsOfNewSubTrees;

    rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfFirstSubTree.begin(), lastLevelOfFirstSubTree.end());

    while (!rootsOfNewSubTrees.empty())
    {
        
        ScidxrbNode<float>* node = rootsOfNewSubTrees.front();
        rootsOfNewSubTrees.erase(rootsOfNewSubTrees.begin());  

        if (node->left == nullptr && node->right == nullptr)
        {
            continue;
        }
        

        std::vector<std::vector<ScidxrbNode<float>*>> currentSubTreeNodesInLevels;
        levelOrderTraversal(node, levelsToTraverse, currentSubTreeNodesInLevels);   
        allSubTrees.push_back(currentSubTreeNodesInLevels);
        std::vector<ScidxrbNode<float>*>& lastLevelOfCurrentSubTree = currentSubTreeNodesInLevels.back();

        rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfCurrentSubTree.begin(), lastLevelOfCurrentSubTree.end());

    }

    for (size_t i = 0; i < allSubTrees.size(); i++)
    {
        std::cout << "Subtree #" << i << ":" << std::endl;
        for (size_t j = 0; j < allSubTrees[i].size(); j++)
        {
            std::cout << "    level #" << j << ":" << std::endl;
            for (size_t k = 0; k < allSubTrees[i][j].size(); k++)
            {
                std::cout << "        " << "[" << allSubTrees[i][j][k]->rbInterval.low << ", " << allSubTrees[i][j][k]->rbInterval.high << "]" << std::endl;
            }
            
        }
        
    }


    float error_bound = 1E3;
    std::vector<int> mergedCompressedTypesLow;

    /* for (size_t i = 0; i < allSubTrees.size(); i++)
    {
        std::vector<std::vector<ScidxNode<float>*>> curSubTree = allSubTrees[i];
         for (size_t m = 0; m < curAllCompressedType.size(); ++m) {
            for (size_t n = 0; n < curAllCompressedType[m].size(); ++n) {
                std::cout << curAllCompressedType[m][n] << " ";
            }
        std::cout << std::endl;
        }

        if (!curAllCompressedType.empty()){
            mergedCompressedTypesLow.insert(mergedCompressedTypesLow.end(), curAllCompressedType[0].begin(), curAllCompressedType[0].end());        
        }
        
    } 

    //mainBuffer for all encode subtrees
    std::vector<unsigned char> mainBuffer;

    //size of every encoded subtree
    std::vector<size_t> encodedSizes;

    
    //init huffmanttree for all data
    unsigned char *treeBuffer;
    size_t treeBufferSize;
    //init
    free(treeBuffer);

   

    // write size to file
    std::ofstream outFile("encoded_sizes.bin", std::ios::binary);
    for (size_t size : encodedSizes) {
        outFile.write(reinterpret_cast<char*>(&size), sizeof(size_t));
    }
    outFile.close();
    
    */

    std::vector<std::vector<ScidxrbNode<float>*>> firstrbSubTree = allSubTrees[0];

    compressTree(firstrbSubTree, error_bound);

    /*std::vector<int> firstIntArray_MaxHigh = firstCompressedType[1];
    std::cout << "Values in firstIntArray_MaxHigh: ";
    for (int value : firstIntArray_MaxHigh) {
        std::cout << value << " ";
    }
    std::cout << std::endl;*/



    /*const int numberQueryOfIntervals = 20;
    const float maxLow = global_min;
    const float maxHigh = global_max;

    for (size_t i = 0; i < numberQueryOfIntervals; i++)
    {
        ScidxInterval<float> queryInterval = generateRandomInterval(maxLow, maxHigh);
        std::cout << "query interval: [" << queryInterval.low << ", " << queryInterval.high << "]" << std::endl;
        std::vector<ScidxNode<float>*> result = rbIntervalTree.query(queryInterval);
        std::cout << "overlapped intervals: " << std::endl;
        for (size_t j = 0; j < result.size(); j++)
        {
            std::cout << "    [" << result[j]->interval.low << ", " << result[j]->interval.high << "] (id: " << result[j]->id << ")" << std::endl;
        }
        
    }*/
    
    
    return 0;
}


void printIntervalTreeArray(std::vector<int>& arr) {
    std::cout << "Interval Tree Array: ";
    
    for (int value : arr) {
        std::cout << value << " ";
    }
    
    std::cout << std::endl;
}

size_t convertIntArray2ByteArray_fast_1b(const std::vector<int>& intArray, std::vector<unsigned char>& result) {
    size_t byteLength = 0;
    size_t intArrayLength = intArray.size();
    
    if (intArrayLength % 8 == 0)
        byteLength = intArrayLength / 8;
    else
        byteLength = intArrayLength / 8 + 1;

    result.resize(byteLength, 0); // Resize and initialize result vector

    size_t n = 0;
    int tmp, type;
    
    for (size_t i = 0; i < byteLength; ++i) {
        tmp = 0;

        for (size_t j = 0; j < 8 && n < intArrayLength; ++j) {
            type = intArray[n];

            if (type == 1)
                tmp |= (1 << (7 - j));

            ++n;
        }

        result[i] = static_cast<unsigned char>(tmp);
    }

    return byteLength;
}


void saveToFile(const std::vector<std::vector<float>>& data, const std::string& filename) {
    std::ofstream outfile(filename);

    if (!outfile) {
        std::cerr << "无法打开文件：" << filename << std::endl;
        return;
    }

    for (const auto& row : data) {
        for (const auto& value : row) {
            outfile << value << " ";
        }
        outfile << std::endl;
    }

    std::cout << "数据已成功保存到文件：" << filename << std::endl;
}



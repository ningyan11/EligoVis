#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <adios2.h> 
#include <scidx.h> 
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
// Random number generator
std::random_device rd;
std::mt19937 gen(rd());

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
ScidxInterval<T> generateRandomIntervalWithPercentage(T maxLow, T maxHigh) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // 计算总体数据范围的10%
    T one_percent_range = (maxHigh - maxLow) * 0.1;
    
    std::uniform_real_distribution<T> distLow(maxLow, maxHigh);
    std::uniform_real_distribution<T> distHigh(maxLow, maxLow + one_percent_range);
    
    ScidxInterval<T> interval;
    interval.low = distLow(gen);
    interval.high = distHigh(gen);
    
    return interval;
}

size_t convertIntArray2ByteArray_fast_1b(const std::vector<int>& intArray, std::vector<unsigned char>& result);
//ScidxRBNode<double>* decompressTree(adios2::Engine& engine, adios2::IO& io, int step, std::vector<int>& firstVector, float error_bound, std::vector<unsigned char> huffmanOutLow, std::vector<unsigned char> huffmanOutHigh, std::vector<unsigned char> huffmanOutMaxHigh);

// ScidxRBNode<double>* attachSubTreesBFS(const std::vector<int>& fullTreeVectorOfMap, const std::vector<ScidxRBNode<double>*>& compressedSubTrees); // BOOL change
ScidxRBNode<double>* attachSubTreesBFS(const std::vector<bool>& fullTreeVectorOfMap, const std::vector<ScidxRBNode<double>*>& compressedSubTrees);

void queryAndConnectSubTree(
    ScidxRedBlackIntervalTree<double>& tree,
    const std::vector<ScidxRBNode<double>*>& compressedSubTrees,
    // const std::vector<int>& allSubTreeMap, // BOOL change
    const std::vector<bool>& allSubTreeMap,
    const ScidxInterval<double>& queryInterval,
    int levelsToTraverse,
    int times,
    std::vector<ScidxRBNode<double>*>& queryResults,
    float error_bound
);

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

int main(int argc, char *argv[]) {
    std::string inputFileName;
    std::string variableName;
    std::string variableType;

    std::vector<size_t> blockShape;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input_file") {
            if (i+1 < argc) {
                inputFileName = argv[i+1];
            } else {
                std::cerr << "--input_file option requires one argument." << std::endl;
                return 1;
            }            
        } else if (arg == "--variable_name") {
            if (i+1 < argc) {
                variableName = argv[i+1];
            } else {
                std::cerr << "--variable_name option requires one argument." << std::endl;
                return 1;
            }             
        }
    }

    std::cout << inputFileName << "," << variableName << std::endl;

    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);
    size_t steps = reader_engine.Steps();
    
    auto var = reader_io.InquireVariable(variableName);
    variableType = reader_io.VariableType(variableName);

    std::cout << "variableType: " << variableType << std::endl;
    std::cout << "steps " << steps << std::endl;
            
    ScidxRedBlackIntervalTree<double> rbIntervalTree;

        size_t blockID = 0;
        std::vector<double> block_mins;
        std::vector<double> block_maxs;

        
        for (size_t step = 1; step < steps; step++) {
            auto blocksInfo = reader_engine.AllStepsBlocksInfo(var).at(step);
            for (const auto &info : blocksInfo) {
                size_t blockSize = 1;

                
                for (size_t i = 0; i < info.Count.size(); i++) {
                    blockSize *= info.Count[i];
                }
                          
                std::vector<double> blockData(blockSize);
                var.SetSelection({info.Start, info.Count});
                var.SetStepSelection({step, 1});
                reader_engine.Get(var, blockData.data(), adios2::Mode::Sync);
                auto minmax = minmax_element(blockData.begin(), blockData.end());
                block_mins.push_back(*minmax.first);
                block_maxs.push_back(*minmax.second);
                blockID++;
            }
        }

        std::cout << blockID << "," << variableName << std::endl;
        std::vector<ScidxInterval<double>> intervals;

       

        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::min();
        for (size_t i = 0; i < block_mins.size(); i++) {
            if (block_mins[i] < global_min) {
                global_min = block_mins[i];
            }
            if (block_maxs[i] > global_max) {
                global_max = block_maxs[i];
            }
            
            ScidxInterval<double> interval;
            interval.low = block_mins[i];
            interval.high = block_maxs[i];
            intervals.push_back(interval);
        }



        std::vector<ScidxInterval<double>> zeroLowIntervalList;

        std::cout << inputFileName << "," << variableName << std::endl;

        std::cout << intervals.size() << "," << variableName << std::endl;

        for (size_t i = 0; i < intervals.size(); i++) {
            std::cout << i << " [" << intervals[i].low << " " << intervals[i].high << "]" << std::endl;
            if (intervals[i].low == 0) {
                zeroLowIntervalList.push_back(intervals[i]);
                continue;
            }
            
            rbIntervalTree.insert(intervals[i], i);
        }

        std::cout << "Red-Black Interval Tree after insertions:" << std::endl;
        // rbIntervalTree.display();

     

        std::cout << "zeroLowIntervalList size" << zeroLowIntervalList.size() << std::endl;

        /*for (size_t i = 0; i < zeroLowIntervalList.size(); i++) {
            std::cout << " [" << zeroLowIntervalList[i].low << " " << zeroLowIntervalList[i].high << "]" << std::endl;
        }*/

    std::cout << "After build the tree " << std::endl;

    // 创建 IO 对象
    std::cout << "Declaring IO object" << std::endl;
    adios2::IO iO = adios.DeclareIO("WriteData");
    std::string filename = "output.bp";
    std::cout << "Opening file " << filename << std::endl;
    adios2::Engine bpWriter = iO.Open(filename, adios2::Mode::Write);
    std::cout << "File opened successfully" << std::endl;

    int levelsToTraverse = 5;
    std::vector<bool> allSubTreeMap = getAllSubTreesMap(rbIntervalTree, levelsToTraverse); // BOOL change
    std::cout << "Finished getAllSubTreesMap" << std::endl;

    std::vector<std::vector<std::vector<ScidxRBNode<double>*>>> allSubTrees;
    std::vector<std::vector<ScidxRBNode<double>*>> firstSubTreeNodesInLevels;
    std::vector<size_t> sizesOfBigMap;
    std::vector<int> combinedVector;
    std::vector<int> firstVectorOfMap;

    levelOrderTraversal(rbIntervalTree.getRoot(), levelsToTraverse, firstSubTreeNodesInLevels, firstVectorOfMap);
    std::cout << "Finished levelOrderTraversal" << std::endl;
    
    allSubTrees.push_back(firstSubTreeNodesInLevels);
    sizesOfBigMap.push_back(firstVectorOfMap.size());
    combinedVector.insert(combinedVector.end(), firstVectorOfMap.begin(), firstVectorOfMap.end());

    std::vector<ScidxRBNode<double>*>& lastLevelOfFirstSubTree = firstSubTreeNodesInLevels.back();
    std::vector<ScidxRBNode<double>*> rootsOfNewSubTrees;
    rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfFirstSubTree.begin(), lastLevelOfFirstSubTree.end());

    while (!rootsOfNewSubTrees.empty()) {
        ScidxRBNode<double>* node = rootsOfNewSubTrees.front();
        rootsOfNewSubTrees.erase(rootsOfNewSubTrees.begin());  

        if (node->left == nullptr && node->right == nullptr) {
            continue;
        }
        
        std::vector<std::vector<ScidxRBNode<double>*>> currentSubTreeNodesInLevels;
        std::vector<int> currentSubTreeVectorOfMap;

        levelOrderTraversal(node, levelsToTraverse, currentSubTreeNodesInLevels, currentSubTreeVectorOfMap);
        
        sizesOfBigMap.push_back(currentSubTreeVectorOfMap.size());
        combinedVector.insert(combinedVector.end(), currentSubTreeVectorOfMap.begin(), currentSubTreeVectorOfMap.end());
        
        allSubTrees.push_back(currentSubTreeNodesInLevels);

        //level of subtree is smaller than levelsToTraverse
        if (currentSubTreeNodesInLevels.size() < levelsToTraverse) {
            continue;
        }

        std::vector<ScidxRBNode<double>*>& lastLevelOfCurrentSubTree = currentSubTreeNodesInLevels.back();
        rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfCurrentSubTree.begin(), lastLevelOfCurrentSubTree.end());
    }

    for (size_t i = 0; i < allSubTrees.size(); i++) {
        std::cout << "Subtree #" << i << ":" << std::endl;
        for (size_t j = 0; j < allSubTrees[i].size(); j++) {
            std::cout << "    level #" << j << ":" << std::endl;
            for (size_t k = 0; k < allSubTrees[i][j].size(); k++) {
                std::cout << "        " << "[" << allSubTrees[i][j][k]->interval.low << ", " << allSubTrees[i][j][k]->interval.high << "]" << std::endl;
            }
        }
    }

    float error_bound = 1E3;
    std::vector<int> mergedCompressedTypesLow;

    std::vector<std::vector<int>> eachMapOfSubTree; 
    size_t startIndex = 0; 
    for (size_t size : sizesOfBigMap) {
        if (startIndex + size <= combinedVector.size()) {
            std::vector<int> subVector(combinedVector.begin() + startIndex, combinedVector.begin() + startIndex + size);
            eachMapOfSubTree.push_back(subVector);
            startIndex += size; 
        } else {
            std::cerr << "Error: The sizes in sizesOfBigMap exceed the size of combinedVector." << std::endl;
            break;
        }
    }

    if (allSubTrees.size() != eachMapOfSubTree.size()) {
        std::cerr << "Error: Mismatch in sizes of the vectors." << std::endl;
        return 1;
    }

    std::vector<ScidxRBNode<double>*> compressedSubTrees; 

    std::vector<std::vector<int>> allTreeTypesLow;
    std::vector<std::vector<int>> allTreeTypesHigh;
    std::vector<std::vector<int>> allTreeTypesMaxHigh;

    bpWriter.BeginStep();
    
    //对单个子树进行type计算，并将越界数据分别存入adios
    for (size_t i = 0; i < allSubTrees.size(); ++i) {
        std::vector<std::vector<ScidxRBNode<double>*>> currentSubTree = allSubTrees[i];
        computeTypeBuffer<double>(currentSubTree, error_bound, i, bpWriter, iO, allTreeTypesLow, allTreeTypesHigh, allTreeTypesMaxHigh);
    }

    std::vector<scidx::HuffmanTree*> huffmanTrees = fullHuffman(allTreeTypesLow, allTreeTypesHigh, allTreeTypesMaxHigh, bpWriter, iO);

    for (size_t i = 0; i < allTreeTypesLow.size(); ++i) {
        std::vector<int> currentTypesLow = allTreeTypesLow[i];
        std::vector<int> currentTypesHigh = allTreeTypesHigh[i];
        std::vector<int> currentTypesMaxHigh = allTreeTypesMaxHigh[i];
        compressTree<double>(currentTypesLow, currentTypesHigh, currentTypesMaxHigh, error_bound, bpWriter, iO, i, huffmanTrees[0], huffmanTrees[1], huffmanTrees[2]);
    }
    
    bpWriter.EndStep();
    bpWriter.Close(); // 关闭写入引擎

    // 创建 IO 对象
    adios2::IO iORead = adios.DeclareIO("ReadData");

    // 打开引擎
    adios2::Engine bpReader = iORead.Open(filename, adios2::Mode::Read);

    bpReader.BeginStep();
    adios2::Variable<unsigned char> fullHuffmanTreeHigh = iORead.InquireVariable<unsigned char>("fullHuffmanTreeHigh");
    adios2::Variable<unsigned char> fullHuffmanTreeLow = iORead.InquireVariable<unsigned char>("fullHuffmanTreeLow");
    adios2::Variable<unsigned char> fullHuffmanTreeMaxHigh = iORead.InquireVariable<unsigned char>("fullHuffmanTreeMaxHigh");

    size_t varSizeHigh = fullHuffmanTreeHigh.Shape()[0];
    size_t varSizeLow = fullHuffmanTreeLow.Shape()[0];
    size_t varSizeMaxHigh = fullHuffmanTreeMaxHigh.Shape()[0];

    std::vector<unsigned char> huffmanOutLow(varSizeLow);
    std::vector<unsigned char> huffmanOutHigh(varSizeHigh);
    std::vector<unsigned char> huffmanOutMaxHigh(varSizeMaxHigh);

    bpReader.Get(fullHuffmanTreeHigh, huffmanOutHigh.data(), adios2::Mode::Sync);  
    bpReader.Get(fullHuffmanTreeLow, huffmanOutLow.data(), adios2::Mode::Sync);
    bpReader.Get(fullHuffmanTreeMaxHigh, huffmanOutMaxHigh.data(), adios2::Mode::Sync);

    // 获取变量的大小
    for (size_t i = 0; i < allSubTrees.size(); ++i) {
        std::vector<int> currentMap = eachMapOfSubTree[i];
        ScidxRBNode<double>* compressedRoot = decompressTree<double>(bpReader, iORead, i, currentMap, error_bound, huffmanOutLow, huffmanOutHigh, huffmanOutMaxHigh);
        compressedSubTrees.push_back(compressedRoot);
    }
    bpReader.EndStep();
    bpReader.Close();

    std::cout << "Displaying All Compressed SubTrees:" << std::endl;
    for (auto& subtreeRoot : compressedSubTrees) {
        ScidxRedBlackIntervalTree<double> tempTree;
        tempTree.setRoot(subtreeRoot);
        tempTree.display();
        std::cout << "----------" << std::endl;
    }

    ScidxRBNode<double>* finalRoot = attachSubTreesBFS(allSubTreeMap, compressedSubTrees);
    ScidxRedBlackIntervalTree<double> finalTree;
    finalTree.setRoot(finalRoot);
    std::cout << "reconstruct Tree" << std::endl;
    finalTree.display();

    //std::vector<ScidxRBNode<double>*> result = rbIntervalTree.query(queryIntervalForTree);

    ScidxRBNode<double>* root = compressedSubTrees[0];
    ScidxRedBlackIntervalTree<double> firstRbIntervalTree;
    firstRbIntervalTree.setRoot(compressedSubTrees[0]);

    std::vector<ScidxRBNode<double>*> resultsOfQuery;
    //queryAndConnectSubTree(firstRbIntervalTree, compressedSubTrees, allSubTreeMap, queryIntervalForTree, levelsToTraverse, 1, resultsOfQuery, error_bound);

    return 0;
}

void queryAndConnectSubTree(
    ScidxRedBlackIntervalTree<double>& tree,
    const std::vector<ScidxRBNode<double>*>& compressedSubTrees,
    const std::vector<bool>& allSubTreeMap, // BOOL change
    const ScidxInterval<double>& queryInterval,
    int levelsToTraverse,
    int times,
    std::vector<ScidxRBNode<double>*>& queryResults,
    float error_bound
) {
    auto queryResult = tree.queryWithMap(queryInterval, times, levelsToTraverse, error_bound);
    queryResults = queryResult.first;

    // 终止条件1: queryResult.second为空
    if (queryResult.second.empty()) {
        //std::cout << "Query result is empty, terminating recursion." << std::endl;
        return;
    }

    // 获取最后一层
    auto lastLayerIter = std::prev(queryResult.second.end());

    // 终止条件2: lastLayerIter指向queryResult.second的开始
    if (lastLayerIter == queryResult.second.begin() && queryResult.second.size() == 1) {
        std::cout << "Only one layer exists, terminating recursion." << std::endl;
        return;
    }

    int maxDepth = lastLayerIter->first; // 获取最大深度
    const auto& nodesAndIndicesAtLastLayer = lastLayerIter->second; // 获取最后一层的节点和它们的索引

    // 终止条件3: nodesAndIndicesAtLastLayer为空
    if (nodesAndIndicesAtLastLayer.empty()) {
        std::cout << "The last layer is empty, terminating recursion." << std::endl;
        return;
    }

    std::cout << "The depth of the last layer in this tree: " << maxDepth << std::endl;

    for (const auto& nodeIndexPair : nodesAndIndicesAtLastLayer) {
        ScidxRBNode<double>* node = nodeIndexPair.first; // 节点指针
        unsigned long long index = nodeIndexPair.second; // 节点在其层中的索引

        // 计算实际在allSubTreeMap中的位置
        unsigned long long realIndex = index - std::pow(2, maxDepth - 1);
        if (realIndex < allSubTreeMap.size() && allSubTreeMap[realIndex] == true) { // BOOL change
            int countOfOne = std::count(allSubTreeMap.begin(), allSubTreeMap.begin() + realIndex + 1, 1);

            // 终止条件4: compressedSubTrees中值取完
            if (compressedSubTrees.empty() || countOfOne > compressedSubTrees.size()) {
                std::cout << "Compressed sub-trees are exhausted, terminating recursion." << std::endl;
                return;
            }

            ScidxRBNode<double>* currentSubTreeInAll = compressedSubTrees[countOfOne - 1];

            // 衔接子树
            if (currentSubTreeInAll->left) {
                node->left = currentSubTreeInAll->left;
            }
            if (currentSubTreeInAll->right) {
                node->right = currentSubTreeInAll->right;
            }

            // 继续递归
            queryAndConnectSubTree(tree, compressedSubTrees, allSubTreeMap, queryInterval, levelsToTraverse, times + 1, queryResults, error_bound);
        }
    }
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

// ScidxRBNode<double>* attachSubTreesBFS(const std::vector<int>& fullTreeVectorOfMap, const std::vector<ScidxRBNode<double>*>& compressedSubTrees) { // BOOL change
ScidxRBNode<double>* attachSubTreesBFS(const std::vector<bool>& fullTreeVectorOfMap, const std::vector<ScidxRBNode<double>*>& compressedSubTrees) {
    if (compressedSubTrees.empty() || fullTreeVectorOfMap.empty()) return nullptr;

    ScidxRBNode<double>* root = compressedSubTrees[0];

    size_t mapIndex = 0; 
    size_t subTreeIndex = 1; 

    while (mapIndex < fullTreeVectorOfMap.size()) {
        std::vector<ScidxRBNode<double>*> currentLastLevelNodes = getLastLevelNodesIncludingNull(root);

        for (ScidxRBNode<double>* node : currentLastLevelNodes) {
            // if (node != nullptr && fullTreeVectorOfMap[mapIndex] == 1 && subTreeIndex < compressedSubTrees.size()) { // BOOL change
            if (node != nullptr && fullTreeVectorOfMap[mapIndex] == true && subTreeIndex < compressedSubTrees.size()) {
                // attach subTree
                ScidxRBNode<double>* subTreeRoot = compressedSubTrees[subTreeIndex++];

                if (subTreeRoot->left != nullptr) {
                    node->left = subTreeRoot->left;
                }
                if (subTreeRoot->right != nullptr) {
                    node->right = subTreeRoot->right;
                }
            }
            mapIndex++;
        }

        if (mapIndex >= fullTreeVectorOfMap.size()) break;
    }
    return root;
}

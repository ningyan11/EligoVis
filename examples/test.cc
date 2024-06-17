
#include <vector>
#include <queue>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>

#include <random>

#include <scidx_block_min_max.h>

#include <scidx.h>

#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <adios2.h>

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
void compressTree(std::vector<int> currentTypesLow, std::vector<int> currentTypesHigh, std::vector<int> currentTypesMaxHigh, float error_bound, adios2::Engine& engine, adios2::IO& io, int step, scidx::HuffmanTree*  fullHuffmanTreeLow, scidx::HuffmanTree*  fullHuffmanTreeHigh, scidx::HuffmanTree*  fullHuffmanTreeMaxHigh);

ScidxRBNode<float>* decompressTree(adios2::Engine& engine, adios2::IO& io, int step, std::vector<int>& firstVector, float error_bound, std::vector<unsigned char> huffmanOutLow, std::vector<unsigned char> huffmanOutHigh, std::vector<unsigned char> huffmanOutMaxHigh);


void computeTypeBuffer(
    std::vector<std::vector<ScidxRBNode<float>*>>& singleSubTree, 
    float error_bound, size_t i, adios2::Engine& engine, adios2::IO& io,
    std::vector<std::vector<int>>& allTreeTypesLow, 
    std::vector<std::vector<int>>& allTreeTypesHigh, 
    std::vector<std::vector<int>>& allTreeTypesMaxHigh
);

std::vector<scidx::HuffmanTree*> fullHuffman(const std::vector<std::vector<int>>& allTreeTypesLow,
                                          const std::vector<std::vector<int>>& allTreeTypesHigh,
                                          const std::vector<std::vector<int>>& allTreeTypesMaxHigh,
                                          adios2::Engine& engine, adios2::IO& io);

ScidxRBNode<float>* attachSubTreesBFS(const std::vector<int>& fullTreeVectorOfMap, const std::vector<ScidxRBNode<float>*>& compressedSubTrees);
void queryAndConnectSubTree(
    ScidxRedBlackIntervalTree<float>& tree,
    const std::vector<ScidxRBNode<float>*>& compressedSubTrees,
    const std::vector<int>& allSubTreeMap,
    const ScidxInterval<float>& queryInterval,
    int levelsToTraverse,
    int times,
    std::vector<ScidxRBNode<float>*>& queryResults,
    float error_bound
) ;


int main(int argc, char *argv[]) {

    // 在程序开始时清空文件
    std::ofstream originalDataFile1("OutlayerMax.txt", std::ios_base::trunc);
    originalDataFile1.close();
    std::ofstream originalDataFile2("OutlayerMaxHighLeafNode.txt", std::ios_base::trunc);
    originalDataFile2.close();
    std::ofstream originalDataFile3("original_data.txt", std::ios_base::trunc);
    originalDataFile3.close();

    // 创建 ADIOS2 对象
    adios2::ADIOS adios;

    // 创建 IO 对象
    adios2::IO iO = adios.DeclareIO("WriteData");

    std::string filename = "output.bp";
    adios2::Engine bpWriter = iO.Open(filename, adios2::Mode::Write);


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

    //std::vector<std::vector<float>> sortedBlockMinMax = sortResultsByMax(blockMinMax);
    
	
    std::vector<ScidxInterval<float>> intervals;

    std::vector<ScidxInterval<float>> avlIntervals;

    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::min();
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
        
        ScidxInterval<float> interval;
        ScidxInterval<float> avlInterval;

        interval.low = blockMinMax[i][0];
        interval.high = blockMinMax[i][1];

        avlInterval.low = blockMinMax[i][0];
        avlInterval.high = blockMinMax[i][1];

        intervals.push_back(interval);
        avlIntervals.push_back(avlInterval);
    }
    

    ScidxRedBlackIntervalTree<float> rbIntervalTree;

    for (size_t i = 0; i < intervals.size(); i++)
    {
        std::cout << intervals[i].low << " " << intervals[i].high << std::endl;
        rbIntervalTree.insert(intervals[i], i);
        
    }

    std::cout << "RB Interval Tree after insertions:" << std::endl;
    rbIntervalTree.display();

    int levelsToTraverse = 5;

    std::vector<int> allSubTreeMap = getAllSubTreesMap(rbIntervalTree, levelsToTraverse);

    std::vector<std::vector<std::vector<ScidxRBNode<float>*>>> allSubTrees;
    std::vector<std::vector<ScidxRBNode<float>*>> firstSubTreeNodesInLevels;
    std::vector<size_t> sizesOfBigMap;
    std::vector<int> combinedVector;
    std::vector<int> firstVectorOfMap;
    
    levelOrderTraversal(rbIntervalTree.getRoot(), levelsToTraverse, firstSubTreeNodesInLevels, firstVectorOfMap);
    allSubTrees.push_back(firstSubTreeNodesInLevels);
    sizesOfBigMap.push_back(firstVectorOfMap.size());
    combinedVector.insert(combinedVector.end(), firstVectorOfMap.begin(), firstVectorOfMap.end());


    std::vector<ScidxRBNode<float>*>& lastLevelOfFirstSubTree = firstSubTreeNodesInLevels.back();

    std::vector<ScidxRBNode<float>*> rootsOfNewSubTrees;

    rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfFirstSubTree.begin(), lastLevelOfFirstSubTree.end());

    while (!rootsOfNewSubTrees.empty())
    {
        
        ScidxRBNode<float>* node = rootsOfNewSubTrees.front();
        rootsOfNewSubTrees.erase(rootsOfNewSubTrees.begin());  

        if (node->left == nullptr && node->right == nullptr)
        {
            continue;
        }
        

        std::vector<std::vector<ScidxRBNode<float>*>> currentSubTreeNodesInLevels;
        std::vector<int> currentSubTreeVectorOfMap;

        levelOrderTraversal(node, levelsToTraverse, currentSubTreeNodesInLevels, currentSubTreeVectorOfMap); 
        sizesOfBigMap.push_back(currentSubTreeVectorOfMap.size());
        combinedVector.insert(combinedVector.end(), currentSubTreeVectorOfMap.begin(), currentSubTreeVectorOfMap.end());
 
        allSubTrees.push_back(currentSubTreeNodesInLevels);

        //level of subtree is smaller than levelsToTraverse
        if(currentSubTreeNodesInLevels.size() < levelsToTraverse){
            continue;
        }

        std::vector<ScidxRBNode<float>*>& lastLevelOfCurrentSubTree = currentSubTreeNodesInLevels.back();

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
                std::cout << "        " << "[" << allSubTrees[i][j][k]->interval.low << ", " << allSubTrees[i][j][k]->interval.high << "]" << std::endl;
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

    //reconstruct the first subTree and display it
    /*std::vector<std::vector<ScidxRBNode<float>*>> firstrbSubTree = allSubTrees[0];
    size_t firstStartIndex = 0;
    size_t firstVectorSize = sizesOfBigMap[0];
    std::vector<int> firstVector(combinedVector.begin() + firstStartIndex, combinedVector.begin() + firstStartIndex + firstVectorSize);
    ScidxRBNode<float>*  fistSubTree = compressTree(firstrbSubTree, error_bound, firstVector);
    ScidxRedBlackIntervalTree<float> tree;
    tree.setRoot(fistSubTree);
    tree.display(); */





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

    std::vector<ScidxRBNode<float>*> compressedSubTrees; 

    std::vector<std::vector<int>> allTreeTypesLow;
    std::vector<std::vector<int>> allTreeTypesHigh;
    std::vector<std::vector<int>> allTreeTypesMaxHigh;

    bpWriter.BeginStep();
    
    //对单个子树进行type计算，并将越界数据分别存入adios
    for (size_t i = 0; i < allSubTrees.size(); ++i) {
        std::vector<std::vector<ScidxRBNode<float>*>> currentSubTree = allSubTrees[i];
        computeTypeBuffer(currentSubTree, error_bound, i, bpWriter, iO, allTreeTypesLow, allTreeTypesHigh, allTreeTypesMaxHigh);
    }


    std::vector<scidx::HuffmanTree*> huffmanTrees = fullHuffman(allTreeTypesLow, allTreeTypesHigh, allTreeTypesMaxHigh, bpWriter, iO);

    for (size_t i = 0; i < allTreeTypesLow.size(); ++i) {
        std::vector<int> currentTypesLow = allTreeTypesLow[i];
        std::vector<int> currentTypesHigh = allTreeTypesHigh[i];
        std::vector<int> currentTypesMaxHigh = allTreeTypesMaxHigh[i];
        compressTree(currentTypesLow, currentTypesHigh, currentTypesMaxHigh, error_bound,  bpWriter, iO, i, huffmanTrees[0], huffmanTrees[1], huffmanTrees[2]);

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
        ScidxRBNode<float>* compressedRoot = decompressTree(bpReader, iORead, i,currentMap, error_bound, huffmanOutLow, huffmanOutHigh, huffmanOutMaxHigh);
        compressedSubTrees.push_back(compressedRoot);
    }
    bpReader.EndStep();
    bpReader.Close();

   
std::cout << "Displaying All Compressed SubTrees:" << std::endl;
    for (auto& subtreeRoot : compressedSubTrees) {
        
        ScidxRedBlackIntervalTree<float> tempTree;
        tempTree.setRoot(subtreeRoot); 
        tempTree.display(); 
        std::cout << "----------" << std::endl; 
    }

    
    ScidxRBNode<float>* finalRoot = attachSubTreesBFS(allSubTreeMap, compressedSubTrees);
    ScidxRedBlackIntervalTree<float> finalTree; 
    finalTree.setRoot(finalRoot);
    std::cout << "reconstruct Tree" << std::endl;
    finalTree.display();


    const int numberQueryOfIntervals = 1000;
    const float maxLow = global_min;
    const float maxHigh = global_max;

   for (size_t i = 0; i < numberQueryOfIntervals; i++)
    {
        ScidxInterval<float> queryIntervalForTree = generateRandomIntervalWithPercentage(maxLow, maxHigh);
        //std::cout << "query interval: [" << queryIntervalForTree.low << ", " << queryIntervalForTree.high << "]" << std::endl;
        std::vector<ScidxRBNode<float>*> result = rbIntervalTree.query(queryIntervalForTree);
        /*std::cout << "overlapped intervals: " << std::endl;
        for (size_t j = 0; j < result.size(); j++)
        {
            std::cout << "    [" << result[j]->interval.low << ", " << result[j]->interval.high << "] (id: " << result[j]->id << ")" << std::endl;
        }
        */
        std::size_t originalResultCount = result.size();
            
    // random decompress
    //ScidxInterval<float> queryIntervalForTree = generateRandomInterval(maxLow, maxHigh);
    //std::cout << "query interval: [" << queryIntervalForTree.low << ", " << queryIntervalForTree.high << "]" << std::endl;
        
    ScidxRBNode<float>* root = compressedSubTrees[0];
    ScidxRedBlackIntervalTree<float> firstRbIntervalTree;
    firstRbIntervalTree.setRoot(compressedSubTrees[0]);

    std::vector<ScidxRBNode<float>*> resultsOfQuery;
    queryAndConnectSubTree(firstRbIntervalTree, compressedSubTrees, allSubTreeMap, queryIntervalForTree, levelsToTraverse, 1, resultsOfQuery, error_bound );
    /*for (size_t k = 0; k < resultsOfQuery.size(); k++)
        {
            std::cout << "    [" << resultsOfQuery[k]->interval.low << ", " << resultsOfQuery[k]->interval.high << "] (id: " << resultsOfQuery[k]->id << ")" << std::endl;
    }

    firstRbIntervalTree.display();
    */
    std::size_t compressedResultCount = resultsOfQuery.size();
    std::size_t falsePositives = compressedResultCount - originalResultCount;
    double fpr = static_cast<double>(falsePositives) / compressedResultCount;
    //std::cout << "False Positive Rate (FPR): " << fpr << std::endl;

}
ScidxAVLIntervalTree<float> avlIntervalTree;

    for (size_t i = 0; i < intervals.size(); i++)
    {
        //std::cout << intervals[i].low << " " << intervals[i].high << std::endl;
        avlIntervalTree.insertNode(i, intervals[i]);
    }
    std::cout << "AVL Interval Tree after insertions:" << std::endl;
    avlIntervalTree.display();

    return 0;
}

void queryAndConnectSubTree(
    ScidxRedBlackIntervalTree<float>& tree,
    const std::vector<ScidxRBNode<float>*>& compressedSubTrees,
    const std::vector<int>& allSubTreeMap,
    const ScidxInterval<float>& queryInterval,
    int levelsToTraverse,
    int times,
    std::vector<ScidxRBNode<float>*>& queryResults,
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
        ScidxRBNode<float>* node = nodeIndexPair.first; // 节点指针
        unsigned long long index = nodeIndexPair.second; // 节点在其层中的索引

        // 计算实际在allSubTreeMap中的位置
        unsigned long long realIndex = index - std::pow(2, maxDepth - 1);
        if (realIndex < allSubTreeMap.size() && allSubTreeMap[realIndex] == 1) {
            int countOfOne = std::count(allSubTreeMap.begin(), allSubTreeMap.begin() + realIndex + 1, 1) ;


        // 终止条件4: compressedSubTrees中值取完
        if (compressedSubTrees.empty() || countOfOne > compressedSubTrees.size()) {
            std::cout << "Compressed sub-trees are exhausted, terminating recursion." << std::endl;
            return;
        }


            ScidxRBNode<float>* currentSubTreeInAll = compressedSubTrees[countOfOne - 1];

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

ScidxRBNode<float>* attachSubTreesBFS(const std::vector<int>& fullTreeVectorOfMap, const std::vector<ScidxRBNode<float>*>& compressedSubTrees) {
    if (compressedSubTrees.empty() || fullTreeVectorOfMap.empty()) return nullptr;

    ScidxRBNode<float>* root = compressedSubTrees[0];

    size_t mapIndex = 0; 
    size_t subTreeIndex = 1; 

    while (mapIndex < fullTreeVectorOfMap.size()) {
       
        std::vector<ScidxRBNode<float>*> currentLastLevelNodes = getLastLevelNodesIncludingNull(root);   

        for (ScidxRBNode<float>* node : currentLastLevelNodes) {
      
            if (node != nullptr && fullTreeVectorOfMap[mapIndex] == 1 && subTreeIndex < compressedSubTrees.size()) {

                // attach subTree
                ScidxRBNode<float>* subTreeRoot = compressedSubTrees[subTreeIndex++];
            
               if(subTreeRoot->left != nullptr){
                    node->left = subTreeRoot->left;
               }
               if(subTreeRoot->right != nullptr){
              
                    node->right = subTreeRoot->right;
               }
            
            }
            mapIndex++; 
        }  
        
        if (mapIndex >= fullTreeVectorOfMap.size()) break; 
    }
    return root;
}
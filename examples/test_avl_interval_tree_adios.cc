
#include <vector>
#include <queue>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>

#include <random>

#include <adios2.h> 
#include <scidx_avl.h> 
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
bool containsDuplicateValues(const std::vector<T>& vec) {
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        auto count = std::count(vec.begin(), vec.end(), *it);
        if (count > 1) {
            return true;
        }
    }
    return false;
}

template <typename T>
ScidxAVLNode<T>* attachSubTreesBFS(const std::vector<bool>& fullTreeVectorOfMap, const std::vector<ScidxAVLNode<T>*>& compressedSubTrees);


int main(int argc, char *argv[]) {

    std::string inputFileName;
    std::string variableName;
    std::string variableType;

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
        
    }

    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::ReadRandomAccess);
    size_t steps = reader_engine.Steps();
    std::cout << "total steps: " << steps << std::endl;
    std::cout << "total steps: " << variableName << std::endl;
    
    auto var = reader_io.InquireVariable(variableName);
    variableType= reader_io.VariableType(variableName);

    std::cout << "variable name: " << variableName << ", variable type: " << variableType << std::endl;

    ScidxAVLIntervalTree<double> avlIntervalTree;

    if (variableType == "double")
    { 
        size_t totalDataSize = 0;
        size_t blockID = 0;
        std::vector<double> block_mins;
        std::vector<double> block_maxs;
        for (size_t step = 1; step < steps; step++)
        {
            
            auto blocksInfo = reader_engine.AllStepsBlocksInfo(var).at(step);

            std::cout << "Step: " << step << ", Number of blocks: " << blocksInfo.size() << std::endl;
            

           for (const auto &info : blocksInfo)
            {
                
                size_t blockSize = 1;   
                for (size_t i = 0; i < blocksInfo[i].Count.size(); i++)
                {
                    blockSize *= blocksInfo[i].Count[i];
                }  
                          
                std::vector<double> blockData(blockSize);
                var.SetSelection({info.Start, info.Count});
                var.SetStepSelection({step, 1});
                reader_engine.Get(var, blockData.data(), adios2::Mode::Sync); 
                auto minmax = minmax_element(blockData.begin(), blockData.end());
                block_mins.push_back(*minmax.first);
                block_maxs.push_back(*minmax.second);
                totalDataSize += sizeof(*minmax.first) + sizeof(*minmax.second);
                blockID++;    
            }

        }

        std::cout << "Total size of all max and min values: " << totalDataSize << " bytes" << std::endl;


        std::cout << "read end " << variableType << std::endl;

        std::vector<ScidxInterval<double>> intervals;

        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::min();
        for (size_t i = 0; i < block_mins.size(); i++)
        {
            if (block_mins[i] < global_min)
            {
                global_min = block_mins[i];
            }
            if (block_maxs[i] > global_max)
            {
                global_max = block_maxs[i];
            }        
            
            ScidxInterval<double> interval;
            interval.low = block_mins[i];
            interval.high = block_maxs[i];
            intervals.push_back(interval);
        }
        

        std::vector<ScidxInterval<double>> zeroLowIntervalList;

        std::cout << "intervals size," << intervals.size() << std::endl;

        for (size_t i = 0; i < intervals.size(); i++)
        {
            //std::cout << i << " [" << intervals[i].low << " " << intervals[i].high << "]" << std::endl;
            if (intervals[i].low == 0)
            {
                zeroLowIntervalList.push_back(intervals[i]);
                continue;
            }
            
            avlIntervalTree.insertNode(i, intervals[i]);
        }
        std::cout << "AVL Interval Tree after insertions:" << std::endl;
        //avlIntervalTree.display();

        /*for (size_t i = 0; i < zeroLowIntervalList.size(); i++)
        {
            std::cout << " [" << zeroLowIntervalList[i].low << " " << zeroLowIntervalList[i].high << "]" << std::endl;
        }*/   
    }


    adios2::IO iO = adios.DeclareIO("WriteData");
    std::string filename = "avl_output.bp";
    adios2::Engine bpWriter = iO.Open(filename, adios2::Mode::Write);

    int levelsToTraverse = 10;
    std::vector<bool> allSubTreeMap = getAllSubTreesMap(avlIntervalTree, levelsToTraverse);

    std::vector<std::vector<std::vector<ScidxAVLNode<double>*>>> allSubTrees;
    std::vector<std::vector<ScidxAVLNode<double>*>> firstSubTreeNodesInLevels;
    std::vector<size_t> sizesOfBigMap;
    std::vector<int> combinedVector;
    std::vector<int> firstVectorOfMap;

    levelOrderTraversal(avlIntervalTree.getRoot(), levelsToTraverse, firstSubTreeNodesInLevels, firstVectorOfMap);
    

    allSubTrees.push_back(firstSubTreeNodesInLevels);
    sizesOfBigMap.push_back(firstVectorOfMap.size());
    combinedVector.insert(combinedVector.end(), firstVectorOfMap.begin(), firstVectorOfMap.end());


    std::vector<ScidxAVLNode<double>*>& lastLevelOfFirstSubTree = firstSubTreeNodesInLevels.back();
    std::vector<ScidxAVLNode<double>*> rootsOfNewSubTrees;
    rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfFirstSubTree.begin(), lastLevelOfFirstSubTree.end());

    while (!rootsOfNewSubTrees.empty()) {
        
        ScidxAVLNode<double>* node = rootsOfNewSubTrees.front();
        rootsOfNewSubTrees.erase(rootsOfNewSubTrees.begin());  

        if (node->left == nullptr && node->right == nullptr) {
            continue;
        }
        
        std::vector<std::vector<ScidxAVLNode<double>*>> currentSubTreeNodesInLevels;
        std::vector<int> currentSubTreeVectorOfMap;

        //TODO: why do levelOrderTraversal again?
        levelOrderTraversal(node, levelsToTraverse, currentSubTreeNodesInLevels, currentSubTreeVectorOfMap);
        
        sizesOfBigMap.push_back(currentSubTreeVectorOfMap.size());
        combinedVector.insert(combinedVector.end(), currentSubTreeVectorOfMap.begin(), currentSubTreeVectorOfMap.end());
        
        allSubTrees.push_back(currentSubTreeNodesInLevels);


        //level of subtree is smaller than levelsToTraverse
        if (currentSubTreeNodesInLevels.size() < levelsToTraverse) {
            continue;
        }

        std::vector<ScidxAVLNode<double>*>& lastLevelOfCurrentSubTree = currentSubTreeNodesInLevels.back();
        rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfCurrentSubTree.begin(), lastLevelOfCurrentSubTree.end());
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

    std::vector<ScidxAVLNode<double>*> compressedSubTrees; 

    std::vector<std::vector<int>> allTreeTypesLow;
    std::vector<std::vector<int>> allTreeTypesHigh;
    std::vector<std::vector<size_t>> allTreeId;
    std::vector<std::vector<int>> allTreeTypesMaxHigh;


    //begin to write the compressed data into adios
    bpWriter.BeginStep();

    //compress subtree
    for (size_t i = 0; i < allSubTrees.size(); ++i) {
        std::vector<std::vector<ScidxAVLNode<double>*>> currentSubTree = allSubTrees[i];
        computeTypeBufferAVL<double>(currentSubTree, error_bound, i, bpWriter, iO, allTreeTypesLow, allTreeTypesHigh, allTreeId, allTreeTypesMaxHigh);
    }

    std::cout << "Finished computeTypeBuffer" << std::endl;


    std::vector<scidx::HuffmanTree*> huffmanTrees = fullHuffman(allTreeTypesLow, allTreeTypesHigh, allTreeTypesMaxHigh, bpWriter, iO);

    for (size_t i = 0; i < allTreeTypesLow.size(); ++i) {
        std::vector<int> currentTypesLow = allTreeTypesLow[i];
        std::vector<int> currentTypesHigh = allTreeTypesHigh[i];
        std::vector<size_t> currentIds = allTreeId[i];
        std::vector<int> currentTypesMaxHigh = allTreeTypesMaxHigh[i];
        compressTree<double>(currentTypesLow, currentTypesHigh, currentIds, currentTypesMaxHigh, error_bound, bpWriter, iO, i, huffmanTrees[0], huffmanTrees[1], huffmanTrees[2]);
    }

    bpWriter.EndStep();
    bpWriter.Close();

    reader_engine.Close();






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
        ScidxAVLNode<double>* compressedRoot = decompressTreeAVL<double>(bpReader, iORead, i, currentMap, error_bound, huffmanOutLow, huffmanOutHigh, huffmanOutMaxHigh);
        compressedSubTrees.push_back(compressedRoot);
    }
    bpReader.EndStep();
    bpReader.Close();

    //std::cout << "Displaying All Compressed SubTrees:" << std::endl;
    for (auto& subtreeRoot : compressedSubTrees) {
        ScidxAVLIntervalTree<double> tempTree;
        tempTree.setRoot(subtreeRoot);
        //tempTree.display();
        //mstd::cout << "----------" << std::endl;
    }

    ScidxAVLNode<double>* finalRoot = attachSubTreesBFS(allSubTreeMap, compressedSubTrees);
    ScidxAVLIntervalTree<double> finalTree;
    finalTree.setRoot(finalRoot);
    std::cout << "reconstruct AVL Tree successfully" << std::endl;
    //finalTree.display();



    return 0;
}

template <typename T>
ScidxAVLNode<T>* attachSubTreesBFS(const std::vector<bool>& fullTreeVectorOfMap, const std::vector<ScidxAVLNode<T>*>& compressedSubTrees) {
    if (compressedSubTrees.empty() || fullTreeVectorOfMap.empty()) return nullptr;

    ScidxAVLNode<T>* root = compressedSubTrees[0];

    size_t mapIndex = 0; 
    size_t subTreeIndex = 1; 

    while (mapIndex < fullTreeVectorOfMap.size()) {
        std::vector<ScidxAVLNode<T>*> currentLastLevelNodes = getLastLevelNodesIncludingNull(root);

        for (ScidxAVLNode<T>* node : currentLastLevelNodes) {
            if (node != nullptr && fullTreeVectorOfMap[mapIndex] == true && subTreeIndex < compressedSubTrees.size()) {
                // attach subTree
                ScidxAVLNode<T>* subTreeRoot = compressedSubTrees[subTreeIndex++];

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


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

std::vector<bool> decodeSubTreesMap(const std::vector<uint8_t>& encodedData, int remainingBits);

template <typename T>
ScidxAVLNode<T>* attachSubTreesBFS(const std::vector<bool>& fullTreeVectorOfMap, const std::vector<ScidxAVLNode<T>*>& compressedSubTrees);


int main(int argc, char *argv[]) {
    
    //TODO：先设为1，检测成功后进行遍历
    size_t k = 0;

    std::string filename = "avl_output.bp";
    adios2::ADIOS adios;

    // 创建 IO 对象
    adios2::IO iORead = adios.DeclareIO("ReadData");
    adios2::Engine bpReader = iORead.Open(filename, adios2::Mode::Read);

    bpReader.BeginStep();

    std::string newFullHuffmanTreeHigh = std::to_string(k) + "_" + "Hu_h";
    std::string newFullHuffmanTreeLow = std::to_string(k) + "_" + "Hu_l";
    std::string newFullHuffmanTreeMaxHigh = std::to_string(k) + "_" + "Hu_mh";
  
    adios2::Variable<unsigned char> fullHuffmanTreeHigh = iORead.InquireVariable<unsigned char>(newFullHuffmanTreeHigh);
    adios2::Variable<unsigned char> fullHuffmanTreeLow = iORead.InquireVariable<unsigned char>(newFullHuffmanTreeLow);
    adios2::Variable<unsigned char> fullHuffmanTreeMaxHigh = iORead.InquireVariable<unsigned char>(newFullHuffmanTreeMaxHigh);


    size_t varSizeHigh = fullHuffmanTreeHigh.Shape()[0];
    size_t varSizeLow = fullHuffmanTreeLow.Shape()[0];
    size_t varSizeMaxHigh = fullHuffmanTreeMaxHigh.Shape()[0];

    std::cout << "varSizeLow = " << varSizeLow << std::endl;

    std::vector<unsigned char> huffmanOutLow(varSizeLow);
    std::vector<unsigned char> huffmanOutHigh(varSizeHigh);
    std::vector<unsigned char> huffmanOutMaxHigh(varSizeMaxHigh);

    std::cout << "Before reading, huffmanOutLow.size() = " << huffmanOutLow.size() << std::endl;


    //从adios中读取出fullHuffmanTree
    bpReader.Get(fullHuffmanTreeHigh, huffmanOutHigh.data(), adios2::Mode::Sync);  
    bpReader.Get(fullHuffmanTreeLow, huffmanOutLow.data(), adios2::Mode::Sync);
    bpReader.Get(fullHuffmanTreeMaxHigh, huffmanOutMaxHigh.data(), adios2::Mode::Sync);

    std::cout << "After reading, huffmanOutLow.size() = " << huffmanOutLow.size() << std::endl;

     //这里先将第k个tree，对应的subTree的个数存入adios（后面可能需要去掉）
    std::string varNamesubTreeize = std::to_string(k) + "_NumSub";
    adios2::Variable<size_t> subTreeize = iORead.InquireVariable<size_t>(varNamesubTreeize);
    size_t subTreeSizeOut;
    bpReader.Get(subTreeize, subTreeSizeOut, adios2::Mode::Sync);  

    //每个subTree的map是连续写到combinedVector中去的，然后sizesOfBigMap记录了每个子树的大小
    std::string varNamesubSizeOfMap = std::to_string(k) + "_SizeSubMap";
    std::string varNamesubMap = std::to_string(k) + "_SubMap";

    adios2::Variable<size_t> subSizeOfMap = iORead.InquireVariable<size_t>(varNamesubSizeOfMap);
    adios2::Variable<int> subFullMap = iORead.InquireVariable<int>(varNamesubMap);

    
    std::vector<size_t> subSizeOfMapOut(subSizeOfMap.Shape()[0]);
    std::vector<int> subFullMapOut(subFullMap.Shape()[0]);
    
    bpReader.Get(subSizeOfMap, subSizeOfMapOut.data(), adios2::Mode::Sync); 
    bpReader.Get(subFullMap, subFullMapOut.data(), adios2::Mode::Sync); 

    


    //将subTrees结合的map,根据大小进行切分
    std::vector<std::vector<int>> nestedVector;
    size_t currentIndex = 0;

    for (size_t size : subSizeOfMapOut) {
        // Ensure we don't go out of bounds
        if (currentIndex + size <= subFullMapOut.size()) {
            std::vector<int> subVector(subFullMapOut.begin() + currentIndex, subFullMapOut.begin() + currentIndex + size);
            nestedVector.push_back(subVector);
            currentIndex += size;
        } else {
            // Handle case where the size exceeds the remaining elements in subFullMapOut
            std::cerr << "Error: Size exceeds the remaining elements in subFullMapOut." << std::endl;
            break;
        }
    }

    float error_bound = 1E3;

    //解压后子树的vector
    std::vector<ScidxAVLNode<double>*> compressedSubTrees; 
    compressedSubTrees.reserve(subTreeSizeOut);

    
    //对子树进行遍历解压
    for (size_t i = 0; i < subTreeSizeOut ; ++i) {
            std::vector<int>& currentMap = nestedVector[i];
            ScidxAVLNode<double>* compressedRoot = decompressTreeAVL<double>(k, bpReader, iORead, i, currentMap, error_bound, huffmanOutLow, huffmanOutHigh, huffmanOutMaxHigh);
            compressedSubTrees.push_back(compressedRoot);
    }
     
    std::cout << "-nnnnn-----" << std::endl;


    //std::cout << "Displaying All Compressed SubTrees:" << std::endl;
    for (auto& subtreeRoot : compressedSubTrees) {
        ScidxAVLIntervalTree<double> tempTree;
        tempTree.setRoot(subtreeRoot);
        tempTree.display();
        std::cout << "----------" << std::endl;
    }


    //整个tree的map
    std::string varNameAllSubTreeMap = std::to_string(k) + "_AllMap";
    std::string varNameAllSubTreeBits = std::to_string(k) + "_Bits";
    adios2::Variable<uint8_t> allSubTreeMapVar = iORead.InquireVariable<uint8_t>(varNameAllSubTreeMap);
    adios2::Variable<int> allSubTreeBitVar = iORead.InquireVariable<int>(varNameAllSubTreeBits);
    std::vector<uint8_t> allSubTreeMapOut(allSubTreeMapVar.Shape()[0]);
    int remainingBitsOut;

    std::cout << "jxnsncn" << std::endl;
    bpReader.Get(allSubTreeMapVar, allSubTreeMapOut.data(), adios2::Mode::Sync);
    bpReader.Get(allSubTreeBitVar, remainingBitsOut, adios2::Mode::Sync); 

    // Print remainingBitsOut
    std::cout << "Remaining Bits: " << remainingBitsOut << std::endl;

    std::vector<bool> decodedAllTreeMap = decodeSubTreesMap(allSubTreeMapOut, remainingBitsOut);

    ScidxAVLNode<double>* finalRoot = attachSubTreesBFS<double>(decodedAllTreeMap, compressedSubTrees);
    ScidxAVLIntervalTree<double> finalTree;
    finalTree.setRoot(finalRoot);
    std::cout << "reconstruct AVL Tree successfully" << std::endl;
    finalTree.display();

    bpReader.EndStep();
    bpReader.Close();

    return 0;

}


//还原整个树的map(byte到bool)
std::vector<bool> decodeSubTreesMap(const std::vector<uint8_t>& encodedData, int remainingBits) {
    std::vector<bool> decoded;

    int totalBits = (encodedData.size() - 1) * 8 + remainingBits; // Calculate total valid bits
    decoded.reserve(totalBits); // Pre-allocate space for efficiency

    int bitIndex = 0;
    for (size_t i = 0; i < encodedData.size(); ++i) {
        uint8_t byte = encodedData[i];
        int bitsToProcess = (i == encodedData.size() - 1) ? remainingBits : 8; // Last byte only processes remaining bits

        for (int j = 0; j < bitsToProcess && bitIndex < totalBits; ++j) {
            bool bit = (byte >> j) & 1;
            decoded.push_back(bit);
            bitIndex++;
        }
    }

    return decoded;
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
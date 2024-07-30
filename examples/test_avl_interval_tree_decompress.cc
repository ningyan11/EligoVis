
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

//ScidxRBNode<double>* attachSubTreesBFS(const std::vector<int>& fullTreeVectorOfMap, const std::vector<ScidxRBNode<double>*>& compressedSubTrees);


int main(int argc, char *argv[]) {
    
    /*std::string filename = "avl_output.bp";
    adios2::ADIOS adios;

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
        ScidxRBNode<double>* compressedRoot = decompressTreeAVL<double>(bpReader, iORead, i, currentMap, error_bound, huffmanOutLow, huffmanOutHigh, huffmanOutMaxHigh);
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

}

template <typename T>
ScidxAVLNode<T>* attachSubTreesBFS(const std::vector<int>& fullTreeVectorOfMap, const std::vector<ScidxAVLNode<T>*>& compressedSubTrees) {
    if (compressedSubTrees.empty() || fullTreeVectorOfMap.empty()) return nullptr;

    ScidxAVLNode<T>* root = compressedSubTrees[0];

    size_t mapIndex = 0; 
    size_t subTreeIndex = 1; 

    while (mapIndex < fullTreeVectorOfMap.size()) {
        std::vector<ScidxAVLNode<T>*> currentLastLevelNodes = getLastLevelNodesIncludingNull(root);

        for (ScidxAVLNode<T>* node : currentLastLevelNodes) {
            if (node != nullptr && fullTreeVectorOfMap[mapIndex] == 1 && subTreeIndex < compressedSubTrees.size()) {
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
    */
}
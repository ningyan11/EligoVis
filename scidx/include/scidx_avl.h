#ifndef _SCIDX_AVL_H
#define _SCIDX_AVL_H

#include <scidx_avl_interval_tree.h>
#include <scidx_Huffman.h>
#include <adios2.h>
#include <vector>

#ifdef _WIN32
#define PATH_SEPARATOR ';'
#else
#define PATH_SEPARATOR ':'
#endif

#ifdef __cplusplus
extern "C" {
#endif

std::vector<scidx::HuffmanTree*> fullHuffman(const std::vector<std::vector<int>>& allTreeTypesLow,
                                             const std::vector<std::vector<int>>& allTreeTypesHigh,
                                             const std::vector<std::vector<int>>& allTreeTypesMaxHigh,
                                             adios2::Engine& engine, adios2::IO& io);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

template<typename T>
void compressTree(size_t k, std::vector<int> currentTypesLow, std::vector<int> currentTypesHigh, std::vector<size_t> currentIds, std::vector<int> currentTypesMaxHigh, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t step, scidx::HuffmanTree* fullHuffmanTreeLow, scidx::HuffmanTree* fullHuffmanTreeHigh, scidx::HuffmanTree* fullHuffmanTreeMaxHigh);

template<typename T>
ScidxAVLNode<T>* decompressTreeAVL(size_t k, adios2::Engine& engine, adios2::IO& io, size_t step, std::vector<int>& firstVector, T error_bound, std::vector<unsigned char> huffmanOutLow, std::vector<unsigned char> huffmanOutHigh, std::vector<unsigned char> huffmanOutMaxHigh);

template<typename T>
void computeTypeBufferAVL(
    size_t k,
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleSubTree, 
    T error_bound, size_t i, adios2::Engine& engine, adios2::IO& io,
    std::vector<std::vector<int>>& allTreeTypesLow, 
    std::vector<std::vector<int>>& allTreeTypesHigh,
    std::vector<std::vector<size_t>>& allTreeId, 
    std::vector<std::vector<int>>& allTreeTypesMaxHigh
);

template<typename T>
void compressNaiveAVL(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
    T error_bound, 
    std::string treeID
);

template<typename T>
void computeOptimizedAVL(std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
                         T error_bound, std::string treeID, std::vector<SkippedNode<T>> skippedAllIntervals);


template<typename T>
void computeOptimizedAVLNew(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree,
    T error_bound,
    std::string treeID,
    std::vector<SkippedNode<T>> skippedAllIntervals,
    std::vector<T>* outListHigh = nullptr
);

template<typename T>
void computeOptimizedAVLForStaggerWithCrossPrediction(
    std::vector<std::vector<ScidxAVLNode<T>*>>& staggerTree,
    T error_bound,
    std::string treeID,
    std::vector<SkippedNode<T>> skippedAllIntervals,
    const std::vector<T>& listHigh,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    size_t nDim
);

template<typename T>
void computeOptimizedTestAVL(std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
                                                  T error_bound, std::string treeID);

template<typename T>
void computeOptimizedSZ3(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
    T error_bound, 
    std::string treeID, std::vector<SkippedNode<T>> skippedAllIntervals
);

template <typename T>
void computeOptimizedZFP(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree,
    T error_bound,
    const std::string& treeID,
    std::vector<SkippedNode<T>> skippedAllIntervals
);


inline std::pair<size_t, size_t> stmCompress_SZ3(double *input, size_t numElements, int blockSize, std::string subDir, double error_bound );


                                            
                                            


std::vector<scidx::HuffmanTree*> fullHuffmanAVL(size_t k, const std::vector<std::vector<int>>& allTreeTypesLow,
                                          const std::vector<std::vector<int>>& allTreeTypesHigh,
                                          const std::vector<std::vector<int>>& allTreeTypesMaxHigh,
                                          adios2::Engine& engine, adios2::IO& io);



template<typename T>
std::tuple<ScidxAVLNode<T>*, std::vector<T>, std::vector<T>, std::vector<size_t>>  decompressOptimizedAVL(const std::string& treeID ,  T error_bound);


template<typename T>
ScidxAVLNode<T>* decompressOptimizedTestAVL(const std::string& treeID ,  T error_bound);

#endif /* __cplusplus */

#endif /* _SCIDX_AVL_H */

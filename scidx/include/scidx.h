/**
 * Scientific Compact Data Index (scidx)
 * Authors: Ning Yan, Lipeng Wan, Sheng Di
 *
**/

#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <scidx_rw.h>
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <adios2.h>



#ifndef _SCIDX_H
#define _SCIDX_H

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
void compressTree(std::vector<int> currentTypesLow, std::vector<int> currentTypesHigh, std::vector<int> currentTypesMaxHigh, T error_bound, adios2::Engine& engine, adios2::IO& io, int step, scidx::HuffmanTree* fullHuffmanTreeLow, scidx::HuffmanTree* fullHuffmanTreeHigh, scidx::HuffmanTree* fullHuffmanTreeMaxHigh);

template<typename T>
ScidxRBNode<T>* decompressTree(adios2::Engine& engine, adios2::IO& io, int step, std::vector<int>& firstVector, T error_bound, std::vector<unsigned char> huffmanOutLow, std::vector<unsigned char> huffmanOutHigh, std::vector<unsigned char> huffmanOutMaxHigh);

template<typename T>
void computeTypeBuffer(
    std::vector<std::vector<ScidxRBNode<T>*>>& singleSubTree, 
    T error_bound, size_t i, adios2::Engine& engine, adios2::IO& io,
    std::vector<std::vector<int>>& allTreeTypesLow, 
    std::vector<std::vector<int>>& allTreeTypesHigh, 
    std::vector<std::vector<int>>& allTreeTypesMaxHigh
);

#endif /* __cplusplus */

#endif /* ----- #ifndef _SCIDX_H  ----- */

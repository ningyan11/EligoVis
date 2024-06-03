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

void compressTree(std::vector<std::vector<ScidxRBNode<float>*>>& singleSubTree, float error_bound, adios2::Engine& engine, adios2::IO& io, int step, scidx::HuffmanTree*  fullHuffmanTreeLow, scidx::HuffmanTree*  fullHuffmanTreeHigh, scidx::HuffmanTree*  fullHuffmanTreeMaxHigh);

ScidxRBNode<float>* decompressTree(adios2::Engine& engine, adios2::IO& io, int step, std::vector<int>& firstVector, float error_bound, std::vector<unsigned char> huffmanOutLow, std::vector<unsigned char> huffmanOutHigh, std::vector<unsigned char> huffmanOutMaxHigh);

void computeTypeBuffer(std::vector<std::vector<ScidxRBNode<float>*>>& singleSubTree, float error_bound, std::vector<int>& allTreeTypesLow, std::vector<int>& allTreeTypesHigh, std::vector<int>& allTreeTypesMaxhigh);

scidx::HuffmanTree* fullHuffman(std::vector<int>& allTreeTypes, adios2::Engine& engine, adios2::IO& io, const std::string& variableName);


#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SCIDX_H  ----- */

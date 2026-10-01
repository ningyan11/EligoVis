#ifndef _SCIDX_OCTREE_H
#define _SCIDX_OCTREE_H

#include <vector>
#include <string>
#include <scidx_octree_interval.h>
#include <scidx_Huffman.h>
#include <adios2.h>


#ifdef _WIN32
#define PATH_SEPARATOR ';'
#else
#define PATH_SEPARATOR ':'
#endif




template<typename T>
void compressOctreeUniform(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& blockCountOnEachDim,
    std::vector<T>* outListHigh);

template<typename T>
void compressOctreeStagger(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<T>& listHigh);

template<typename T>
std::vector<OctreeNode<T>> decompressOctreeUniform(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& blockCountOnEachDim);

template<typename T>
std::vector<OctreeNode<T>> decompressOctreeStagger(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<T>& listHigh);

#endif /* _SCIDX_OCTREE_H */
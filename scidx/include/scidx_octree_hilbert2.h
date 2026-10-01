#ifndef _SCIDX_OCTREE_HILBERT2_H
#define _SCIDX_OCTREE_HILBERT2_H

#include <vector>
#include <string>
#include <scidx_octree_interval.h>   // OctreeNode, buildOctree

#ifdef _WIN32
#define PATH_SEPARATOR ';'
#else
#define PATH_SEPARATOR ':'
#endif

// ============================================================
// 设计说明：
// - UNIFORM 树：min、max 都按 Hilbert 曲线顺序做链式预测（跟前一个
//   Hilbert 序邻居比较），压缩/解压时同时对外提供 listLow、listHigh
//   两份"按普通行主序(x+y*Dx+z*Dx*Dy)索引"的重建值数组。
// - STAGGER 树：叶子按普通行主序存储和遍历，不再走链式预测；min 和
//   max 都通过"该叶子对应的最多8个 UNIFORM 邻居"，分别查 listLow、
//   listHigh 取平均值作为预测值，再对预测残差做逐元素独立量化。
// ============================================================

template<typename T>
void compressOctreeUniformHilbertNew(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& blockCountOnEachDim,
    std::vector<T>* outListLow,
    std::vector<T>* outListHigh);

template<typename T>
void compressOctreeStaggerHilbertNew(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<T>& listLow,
    const std::vector<T>& listHigh);

template<typename T>
std::vector<OctreeNode<T>> decompressOctreeUniformHilbertNew(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& blockCountOnEachDim,
    std::vector<T>* outListLow,
    std::vector<T>* outListHigh);

template<typename T>
std::vector<OctreeNode<T>> decompressOctreeStaggerHilbertNew(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<T>& listLow,
    const std::vector<T>& listHigh);

#endif /* _SCIDX_OCTREE_HILBERT2_H */
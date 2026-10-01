#ifndef _SCIDX_OCTREE_INTERVAL_STANDARD_H
#define _SCIDX_OCTREE_INTERVAL_STANDARD_H

#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <scidx_defines.h>

#ifdef _WIN32
#define PATH_SEPARATOR ';'
#else
#define PATH_SEPARATOR ':'
#endif

// ============================================================
// 标准隐式完全八叉树（Morton/Z-order 内部存储）
//   子节点数组下标 = 8*父节点下标 + 1 + k  (k=0..7)，纯整数算术，O(1)
//   跟“行主序重算坐标”版本相比：
//     - 建树更慢（每个叶子多一次 Morton 位交织编码开销，实测约2倍）
//     - 查询更快（省掉每层的除法和 octreeLevelOffset 重新累加，实测约1.9倍）
//
// 外部接口跟原 scidx_octree_interval.h 完全一致：
//   - blockMins/blockMaxs 仍按行主序 flatId = x + y*Dx + z*Dx*Dy 传入
//   - queryOctree 返回的 id 仍是同一套行主序 flatId
//   下游所有代码（压缩/STAGGER/Marching Cubes）不需要改动。
// ============================================================

// Octree 节点
template<typename T>
struct OctreeNode {
    T minVal;
    T maxVal;
};

// ---------- Morton (Z-order) 编码 / 解码，最多支持每维 21 bit（2^21，远超常见分块规模） ----------
namespace scidx_octree_standard_detail {

inline uint64_t splitBy3(uint32_t a) {
    uint64_t x = a & 0x1fffffULL;
    x = (x | x << 32) & 0x1f00000000ffffULL;
    x = (x | x << 16) & 0x1f0000ff0000ffULL;
    x = (x | x << 8)  & 0x100f00f00f00f00fULL;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2)  & 0x1249249249249249ULL;
    return x;
}
inline uint64_t morton3D_encode(uint32_t x, uint32_t y, uint32_t z) {
    return splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2);
}
inline uint32_t compact3(uint64_t x) {
    x &= 0x1249249249249249ULL;
    x = (x | (x >> 2))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x >> 4))  & 0x100f00f00f00f00fULL;
    x = (x | (x >> 8))  & 0x1f0000ff0000ffULL;
    x = (x | (x >> 16)) & 0x1f00000000ffffULL;
    x = (x | (x >> 32)) & 0x1fffffULL;
    return (uint32_t)x;
}
inline void morton3D_decode(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z) {
    x = compact3(code);
    y = compact3(code >> 1);
    z = compact3(code >> 2);
}

inline void decodeFlatId3D(size_t flatId, const std::vector<size_t>& dim, size_t& x, size_t& y, size_t& z) {
    x = flatId % dim[0];
    size_t rem = flatId / dim[0];
    y = rem % dim[1];
    z = rem / dim[1];
}
inline size_t encodeFlatId3D(size_t x, size_t y, size_t z, const std::vector<size_t>& dim) {
    return x + y * dim[0] + z * dim[0] * dim[1];
}

} // namespace scidx_octree_standard_detail

// 建立 Octree（标准隐式8叉树，Morton序存储）
template<typename T>
inline std::vector<OctreeNode<T>> buildOctree(
    const std::vector<T>& blockMins,
    const std::vector<T>& blockMaxs,
    const std::vector<size_t>& blockCountOnEachDim)
{
    using namespace scidx_octree_standard_detail;

    size_t dimSize = blockCountOnEachDim[0];
    size_t depth = 0, d = dimSize;
    while (d > 1) { d /= 2; depth++; }

    size_t totalNodes = 0, levelSize = 1;
    std::vector<size_t> levelOffsetArr(depth + 1);
    for (size_t i = 0; i <= depth; i++) {
        levelOffsetArr[i] = totalNodes;
        totalNodes += levelSize;
        levelSize *= 8;
    }
    size_t leafOffset = levelOffsetArr[depth];

    std::vector<OctreeNode<T>> tree(totalNodes);

    // 叶子层：输入仍是行主序 flatId，散射到 Morton 序的数组位置
    for (size_t i = 0; i < blockMins.size(); i++) {
        size_t x, y, z;
        decodeFlatId3D(i, blockCountOnEachDim, x, y, z);
        uint64_t m = morton3D_encode((uint32_t)x, (uint32_t)y, (uint32_t)z);
        tree[leafOffset + m].minVal = blockMins[i];
        tree[leafOffset + m].maxVal = blockMaxs[i];
    }

    // 自底向上聚合：子节点在数组里天然连续（8*父+1 .. 8*父+8），纯算术定位
    for (int level = (int)depth - 1; level >= 0; level--) {
        size_t parentOffset = levelOffsetArr[level];
        size_t childOffset  = levelOffsetArr[level + 1];
        size_t levelCount = (size_t)1 << (3 * level); // 8^level

        for (size_t p = 0; p < levelCount; p++) {
            T minV = std::numeric_limits<T>::max();
            T maxV = std::numeric_limits<T>::lowest();
            size_t firstChild = childOffset + p * 8;
            for (size_t k = 0; k < 8; k++) {
                minV = std::min(minV, tree[firstChild + k].minVal);
                maxV = std::max(maxV, tree[firstChild + k].maxVal);
            }
            tree[parentOffset + p].minVal = minV;
            tree[parentOffset + p].maxVal = maxV;
        }
    }
    return tree;
}

// 递归查询：子节点下标用纯算术 8*nodeIdx+1+k 定位，不需要每层重算坐标/除法
template<typename T>
inline void queryOctreeRecursive(
    const std::vector<OctreeNode<T>>& tree,
    T queryLow, T queryHigh, T error_bound,
    size_t nodeIdx, size_t nodeLevel, size_t depth,
    uint64_t mortonPrefix,
    std::vector<size_t>& resultIds,
    const std::vector<size_t>& blockCountOnEachDim)
{
    using namespace scidx_octree_standard_detail;

    const OctreeNode<T>& node = tree[nodeIdx];
    if (node.maxVal + error_bound < queryLow || node.minVal - error_bound > queryHigh) return;

    if (nodeLevel == depth) {
        uint32_t x, y, z;
        morton3D_decode(mortonPrefix, x, y, z);
        // 转回行主序 flatId 输出，保证跟压缩/下游代码完全对应，不需要改任何调用方
        size_t flatId = encodeFlatId3D(x, y, z, blockCountOnEachDim);
        resultIds.push_back(flatId);
        return;
    }

    size_t firstChild = nodeIdx * 8 + 1;
    for (size_t k = 0; k < 8; k++) {
        uint64_t childMorton = (mortonPrefix << 3) | k;
        queryOctreeRecursive(tree, queryLow, queryHigh, error_bound,
                              firstChild + k, nodeLevel + 1, depth,
                              childMorton, resultIds, blockCountOnEachDim);
    }
}

// 查询入口
template<typename T>
inline std::vector<size_t> queryOctree(
    const std::vector<OctreeNode<T>>& octree,
    T queryLow, T queryHigh,
    const std::vector<size_t>& blockCountOnEachDim,
    T error_bound)
{
    size_t dimSize = blockCountOnEachDim[0];
    size_t depth = 0, d = dimSize;
    while (d > 1) { d /= 2; depth++; }

    std::vector<size_t> resultIds;
    queryOctreeRecursive(octree, queryLow, queryHigh, error_bound,
                          0, 0, depth, 0ULL, resultIds, blockCountOnEachDim);
    return resultIds;
}

#endif /* _SCIDX_OCTREE_INTERVAL_STANDARD_H */

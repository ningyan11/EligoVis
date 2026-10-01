#ifndef _SCIDX_OCTREE_HILBERT_H
#define _SCIDX_OCTREE_HILBERT_H

#include <vector>
#include <string>
#include <cstdint>
#include <scidx_octree_interval.h>   // OctreeNode, buildOctree

#ifdef _WIN32
#define PATH_SEPARATOR ';'
#else
#define PATH_SEPARATOR ':'
#endif

// ============================================================
// 3D Hilbert 曲线坐标 <-> 一维索引的相互转换
// 基于 Skilling(2004) 的通用 n 维 Hilbert 曲线算法（transpose 变换）。
// 要求 dimSize = 2^bits（跟 buildOctree 对 blockCountOnEachDim 的要求一致）。
// ============================================================
inline uint64_t hilbertXYZToD(int bits, uint64_t x, uint64_t y, uint64_t z) {
    if (bits <= 0) return 0;

    uint64_t X[3] = {x, y, z};
    const int n = 3;
    uint64_t M = 1ULL << (bits - 1);

    // Inverse undo
    for (uint64_t Q = M; Q > 1; Q >>= 1) {
        uint64_t P = Q - 1;
        for (int i = 0; i < n; i++) {
            if (X[i] & Q) {
                X[0] ^= P;
            } else {
                uint64_t t = (X[0] ^ X[i]) & P;
                X[0] ^= t;
                X[i] ^= t;
            }
        }
    }

    // Gray encode
    for (int i = 1; i < n; i++) X[i] ^= X[i - 1];
    uint64_t t2 = 0;
    for (uint64_t Q = M; Q > 1; Q >>= 1) {
        if (X[n - 1] & Q) t2 ^= (Q - 1);
    }
    for (int i = 0; i < n; i++) X[i] ^= t2;

    // 把 transpose 形式的 X[0..2] 按位交错，拼成一个整数 Hilbert 距离
    // 顺序：从最高位到最低位，每一位依次取 X[0],X[1],X[2] 的对应位
    uint64_t h = 0;
    for (int b = bits - 1; b >= 0; b--) {
        for (int i = 0; i < n; i++) {
            h = (h << 1) | ((X[i] >> b) & 1ULL);
        }
    }
    return h;
}

inline void hilbertDToXYZ(int bits, uint64_t d, uint64_t& x, uint64_t& y, uint64_t& z) {
    if (bits <= 0) { x = y = z = 0; return; }

    const int n = 3;
    uint64_t X[3] = {0, 0, 0};

    // 按 hilbertXYZToD 里打包的顺序，把 d 的每一位还原回 X[0..2]
    int totalBits = bits * n;
    for (int idx = 0; idx < totalBits; idx++) {
        int bitPos = totalBits - 1 - idx;
        uint64_t bit = (d >> bitPos) & 1ULL;
        int b = bits - 1 - (idx / n);
        int i = idx % n;
        if (bit) X[i] |= (1ULL << b);
    }

    // Gray decode
    uint64_t t = X[n - 1] >> 1;
    for (int i = n - 1; i > 0; i--) X[i] ^= X[i - 1];
    X[0] ^= t;

    // Undo excess work（AxestoTranspose 的逆操作）
    uint64_t M = 1ULL << (bits - 1);
    for (uint64_t Q = 2; Q != M; Q <<= 1) {
        uint64_t P = Q - 1;
        for (int i = n - 1; i >= 0; i--) {
            if (X[i] & Q) {
                X[0] ^= P;
            } else {
                uint64_t t2 = (X[0] ^ X[i]) & P;
                X[0] ^= t2;
                X[i] ^= t2;
            }
        }
    }

    x = X[0];
    y = X[1];
    z = X[2];
}

// 从 dimSize（要求是2的幂）算出 bits
inline int hilbertBitsFromDimSize(size_t dimSize) {
    int bits = 0;
    size_t d = dimSize;
    while (d > 1) { d >>= 1; bits++; }
    return bits;
}

// ============================================================
// 压缩 / 解压接口（min、max 全部改为沿 Hilbert 曲线顺序做链式预测）
// ============================================================
template<typename T>
void compressOctreeUniformHilbert(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& blockCountOnEachDim);

template<typename T>
void compressOctreeStaggerHilbert(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& staggerBlockCountOnEachDim);

template<typename T>
std::vector<OctreeNode<T>> decompressOctreeUniformHilbert(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& blockCountOnEachDim);

template<typename T>
std::vector<OctreeNode<T>> decompressOctreeStaggerHilbert(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& staggerBlockCountOnEachDim);

#endif /* _SCIDX_OCTREE_HILBERT_H */
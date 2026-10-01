#include "scidx_octree_hilbert2.h"
#include <scidx_Huffman.h>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <mpi.h>
#include <zstd.h>

using namespace scidx;

// ============================================================
// 非模板函数，来自 scidx_avl.cc，跨文件复用
// ============================================================
HuffmanTree* processHuffmanTreeNaiveAVL(std::vector<int> treeType, std::string& fileName);
std::vector<unsigned char> singleHuffmanEncodeZstdAVL(std::vector<int> type, HuffmanTree *huffmanTree);
std::vector<unsigned char> loadHuffmanTree(const std::string& filename);
std::vector<unsigned char> readCompressedData(const std::string& filename);
std::vector<int> singleHuffmanDecodeZstdAVL(std::vector<unsigned char> compressedByZstdData, size_t typeSize, unsigned char *s);



namespace {

// ---------- 通用二进制文件读写（outlier数据） ----------
template<typename T>
std::vector<T> readHilbert2BinaryVector(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open " << filename << std::endl;
        return {};
    }
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    std::vector<T> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    return data;
}

template<typename T>
void writeHilbert2BinaryVector(const std::string& filename, const std::vector<T>& data) {
    std::ofstream outFile(filename, std::ios::binary);
    size_t size = data.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    outFile.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
    outFile.close();
}

// ---------- 3D Hilbert 曲线坐标 <-> 一维索引（Skilling 通用算法，未改动） ----------
uint64_t hilbert2XYZToD(int bits, uint64_t x, uint64_t y, uint64_t z) {
    if (bits <= 0) return 0;
    uint64_t X[3] = {x, y, z};
    const int n = 3;
    uint64_t M = 1ULL << (bits - 1);
    for (uint64_t Q = M; Q > 1; Q >>= 1) {
        uint64_t P = Q - 1;
        for (int i = 0; i < n; i++) {
            if (X[i] & Q) X[0] ^= P;
            else { uint64_t t = (X[0] ^ X[i]) & P; X[0] ^= t; X[i] ^= t; }
        }
    }
    for (int i = 1; i < n; i++) X[i] ^= X[i - 1];
    uint64_t t2 = 0;
    for (uint64_t Q = M; Q > 1; Q >>= 1) if (X[n - 1] & Q) t2 ^= (Q - 1);
    for (int i = 0; i < n; i++) X[i] ^= t2;
    uint64_t h = 0;
    for (int b = bits - 1; b >= 0; b--)
        for (int i = 0; i < n; i++) h = (h << 1) | ((X[i] >> b) & 1ULL);
    return h;
}

void hilbert2DToXYZ(int bits, uint64_t d, uint64_t& x, uint64_t& y, uint64_t& z) {
    if (bits <= 0) { x = y = z = 0; return; }
    const int n = 3;
    uint64_t X[3] = {0, 0, 0};
    int totalBits = bits * n;
    for (int idx = 0; idx < totalBits; idx++) {
        int bitPos = totalBits - 1 - idx;
        uint64_t bit = (d >> bitPos) & 1ULL;
        int b = bits - 1 - (idx / n);
        int i = idx % n;
        if (bit) X[i] |= (1ULL << b);
    }
    uint64_t t = X[n - 1] >> 1;
    for (int i = n - 1; i > 0; i--) X[i] ^= X[i - 1];
    X[0] ^= t;
    uint64_t N = 1ULL << bits;
    for (uint64_t Q = 2; Q != N; Q <<= 1) {
        uint64_t P = Q - 1;
        for (int i = n - 1; i >= 0; i--) {
            if (X[i] & Q) X[0] ^= P;
            else { uint64_t t2 = (X[0] ^ X[i]) & P; X[0] ^= t2; X[i] ^= t2; }
        }
    }
    x = X[0]; y = X[1]; z = X[2];
}

int hilbert2BitsFromDimSize(size_t dimSize) {
    int bits = 0; size_t d = dimSize;
    while (d > 1) { d >>= 1; bits++; }
    return bits;
}

// ---------- Morton (Z-order) 编码：必须跟 scidx_octree_interval.h 里 buildOctree
// 内部用的完全是同一套实现，否则读出来的叶子位置会对不上 ----------
uint64_t splitBy3(uint32_t a) {
    uint64_t x = a & 0x1fffffULL;
    x = (x | x << 32) & 0x1f00000000ffffULL;
    x = (x | x << 16) & 0x1f0000ff0000ffULL;
    x = (x | x << 8)  & 0x100f00f00f00f00fULL;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2)  & 0x1249249249249249ULL;
    return x;
}
uint64_t morton3D_encode(uint32_t x, uint32_t y, uint32_t z) {
    return splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2);
}

// ---------- 行主序(x+y*Dx+z*Dx*Dy) flatId <-> 3D坐标 ----------
inline void decodeFlatId3D(size_t flatId, const std::vector<size_t>& dim, size_t& x, size_t& y, size_t& z) {
    x = flatId % dim[0];
    size_t rem = flatId / dim[0];
    y = rem % dim[1];
    z = rem / dim[1];
}
inline size_t encodeFlatId3D(size_t x, size_t y, size_t z, const std::vector<size_t>& dim) {
    return x + y * dim[0] + z * dim[0] * dim[1];
}

// STAGGER叶子坐标 -> 最多8个关联UNIFORM块的flatId（未改动，逻辑本身没问题）
inline void relatedUniformIds3D(
    size_t sx, size_t sy, size_t sz,
    const std::vector<size_t>& uniformDim,
    size_t* relatedIds, size_t& relatedCount)
{
    relatedCount = 0;
    for (size_t mask = 0; mask < 8; mask++) {
        size_t offX = mask & 1, offY = (mask >> 1) & 1, offZ = (mask >> 2) & 1;
        size_t ux, uy, uz; bool valid = true;
        if (sx == 0) ux = 0; else { ux = sx - 1 + offX; if (ux >= uniformDim[0]) valid = false; }
        if (valid) { if (sy == 0) uy = 0; else { uy = sy - 1 + offY; if (uy >= uniformDim[1]) valid = false; } }
        if (valid) { if (sz == 0) uz = 0; else { uz = sz - 1 + offZ; if (uz >= uniformDim[2]) valid = false; } }
        if (valid) relatedIds[relatedCount++] = encodeFlatId3D(ux, uy, uz, uniformDim);
    }
}

// ---------- 链式量化 / 反量化（UNIFORM 用；未改动） ----------
template<typename T>
void chainQuantizeHilbert2(
    const std::vector<T>& data, T error_bound,
    std::vector<int>& typeVector, std::vector<T>& outLayerData, std::vector<T>& reconstructedData)
{
    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T checkRadius = (quantization_intervals - 1) * error_bound;
    const T interval = 2 * error_bound;
    const T recip_precision = 1 / error_bound;

    typeVector.clear(); outLayerData.clear();
    reconstructedData.resize(data.size()); typeVector.reserve(data.size());
    T predData = 0, reconstructed = 0;

    for (size_t i = 0; i < data.size(); i++) {
        T curData = data[i];
        if (i == 0) {
            typeVector.push_back(0); outLayerData.push_back(curData); reconstructed = curData;
        } else {
            T predAbsErr = std::abs(curData - predData);
            if (predAbsErr < checkRadius) {
                int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                if (curData >= predData) { typeVector.push_back(intvRadius + state); reconstructed = predData + state * interval; }
                else { typeVector.push_back(intvRadius - state); reconstructed = predData - state * interval; }
            } else {
                typeVector.push_back(0); outLayerData.push_back(curData); reconstructed = curData;
            }
        }
        reconstructedData[i] = reconstructed;
        predData = reconstructed;
    }
}

template<typename T>
std::vector<T> chainDequantizeHilbert2(
    const std::vector<int>& typeVector, const std::vector<T>& outLayerData, T error_bound)
{
    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T interval = 2 * error_bound;

    std::vector<T> reconstructed(typeVector.size());
    T predData = 0; size_t outlierIdx = 0;
    for (size_t i = 0; i < typeVector.size(); i++) {
        int t = typeVector[i]; T curVal;
        if (t == 0) curVal = outLayerData[outlierIdx++];
        else { int state = t - intvRadius; curVal = predData + state * interval; }
        reconstructed[i] = curVal; predData = curVal;
    }
    return reconstructed;
}

// ---------- 独立量化 / 反量化（STAGGER 用；未改动） ----------
template<typename T>
void quantizeAgainstPredicted(
    const std::vector<T>& data, const std::vector<T>& predicted, T error_bound,
    std::vector<int>& typeVector, std::vector<T>& outLayerData)
{
    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T checkRadius = (quantization_intervals - 1) * error_bound;
    const T recip_precision = 1 / error_bound;

    typeVector.clear(); outLayerData.clear(); typeVector.reserve(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        T curData = data[i], predictedVal = predicted[i];
        T predAbsErr = std::abs(curData - predictedVal);
        if (predAbsErr < checkRadius) {
            int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
            if (curData >= predictedVal) typeVector.push_back(intvRadius + state);
            else typeVector.push_back(intvRadius - state);
        } else {
            typeVector.push_back(0); outLayerData.push_back(curData);
        }
    }
}

template<typename T>
std::vector<T> dequantizeAgainstPredicted(
    const std::vector<int>& typeVector, const std::vector<T>& predicted,
    const std::vector<T>& outLayerData, T error_bound)
{
    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T interval = 2 * error_bound;

    std::vector<T> result(typeVector.size()); size_t outlierIdx = 0;
    for (size_t i = 0; i < typeVector.size(); i++) {
        int t = typeVector[i];
        if (t == 0) result[i] = outLayerData[outlierIdx++];
        else { int state = t - intvRadius; result[i] = predicted[i] + state * interval; }
    }
    return result;
}

} // anonymous namespace


// ============================================================
// UNIFORM 树压缩【已修正+优化】
// 关键修正：octree 现在内部按 Morton 序存储叶子（标准八叉树），
//   不能再用 octree[leafOffset+i] 直接按下标取"行主序第i块"，
//   必须先把 Hilbert 序第h个点的空间坐标转成 Morton 码，再去取值。
// 优化点：省掉了原来"先按行主序整体读出leafMins/leafMaxs、
//   再按Hilbert序重排"这两步，直接一步从octree按Morton码读到Hilbert序数组，
//   少一次O(N)遍历和一次数组分配。
// ============================================================
template<typename T>
void compressOctreeUniformHilbertNew(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& blockCountOnEachDim,
    std::vector<T>* outListLow,
    std::vector<T>* outListHigh)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : blockCountOnEachDim) totalLeaves *= d;
    size_t leafOffset = octree.size() - totalLeaves;

    size_t dimSize = blockCountOnEachDim[0];
    int bits = hilbert2BitsFromDimSize(dimSize);

    auto compressStart_time = std::chrono::high_resolution_clock::now();

    // ========== 直接按 Hilbert 序读取（内部经 Morton 码定位到 octree 里的真实叶子） ==========
    std::vector<T> hilbertMins(totalLeaves), hilbertMaxs(totalLeaves);
    for (uint64_t h = 0; h < totalLeaves; h++) {
        uint64_t x, y, z;
        hilbert2DToXYZ(bits, h, x, y, z);
        uint64_t m = morton3D_encode((uint32_t)x, (uint32_t)y, (uint32_t)z);
        hilbertMins[h] = octree[leafOffset + m].minVal;
        hilbertMaxs[h] = octree[leafOffset + m].maxVal;
    }

    // ========== min、max 都走链式量化（沿 Hilbert 序） ==========
    std::vector<int> typeMinVector, typeMaxVector;
    std::vector<T> outLayerMinData, outLayerMaxData;
    std::vector<T> reconstructedHilbertMins, reconstructedHilbertMaxs;

    chainQuantizeHilbert2(hilbertMins, error_bound, typeMinVector, outLayerMinData, reconstructedHilbertMins);
    chainQuantizeHilbert2(hilbertMaxs, error_bound, typeMaxVector, outLayerMaxData, reconstructedHilbertMaxs);

    auto quantiEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_quanti_time = quantiEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert2 UNIFORM] pre_quanti_time (hilbert read from morton octree + chain quantize): "
              << pre_quanti_time.count() << " seconds" << std::endl;

    // ========== 把Hilbert序重建值散射回行主序，输出 listLow/listHigh（供STAGGER用，保持行主序契约不变） ==========
    if (outListLow) {
        outListLow->assign(totalLeaves, T(0));
        for (uint64_t h = 0; h < totalLeaves; h++) {
            uint64_t x, y, z;
            hilbert2DToXYZ(bits, h, x, y, z);
            size_t flatId = encodeFlatId3D((size_t)x, (size_t)y, (size_t)z, blockCountOnEachDim);
            (*outListLow)[flatId] = reconstructedHilbertMins[h];
        }
    }
    if (outListHigh) {
        outListHigh->assign(totalLeaves, T(0));
        for (uint64_t h = 0; h < totalLeaves; h++) {
            uint64_t x, y, z;
            hilbert2DToXYZ(bits, h, x, y, z);
            size_t flatId = encodeFlatId3D((size_t)x, (size_t)y, (size_t)z, blockCountOnEachDim);
            (*outListHigh)[flatId] = reconstructedHilbertMaxs[h];
        }
    }

    // ========== Huffman + Zstd 压缩 ==========
    std::string Minfilename = treeID + "-h2LeafMin";
    std::string Maxfilename = treeID + "-h2LeafMax";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);

    auto huffmanBuildEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanBuild_time = huffmanBuildEnd_time - quantiEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert2 UNIFORM] huffmanBuild_time: " << huffmanBuild_time.count()
              << " seconds" << std::endl;

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);
    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);

    auto compressEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> encode_time = compressEnd_time - huffmanBuildEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert2 UNIFORM] encode_time: " << encode_time.count()
              << " seconds" << std::endl;

 

    // ========== 写文件 ==========
    auto writeBegin_time = std::chrono::high_resolution_clock::now();

    writeHilbert2BinaryVector(treeID + "-H2OutMin.bin", outLayerMinData);
    writeHilbert2BinaryVector(treeID + "-H2OutMax.bin", outLayerMaxData);
    {
        std::ofstream outMin(treeID + "-h2LeafMin-compressedIndexData", std::ios::binary);
        outMin.write(reinterpret_cast<const char*>(compressedMin.data()), compressedMin.size());
        outMin.close();
        std::ofstream outMax(treeID + "-h2LeafMax-compressedIndexData", std::ios::binary);
        outMax.write(reinterpret_cast<const char*>(compressedMax.data()), compressedMax.size());
        outMax.close();
    }

    auto writeEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = writeEnd_time - writeBegin_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert2 UNIFORM] write_time: " << write_time.count()
              << " seconds" << std::endl;
}


// ============================================================
// STAGGER 树压缩【已修正】
// 关键修正：不做Hilbert重排，仍按行主序i逐叶子处理，
//   但读取 octree 时必须先把 i 转成空间坐标、再转成 Morton 码，
//   不能再用 octree[leafOffset+i] 直接按下标取值。
// ============================================================
template<typename T>
void compressOctreeStaggerHilbertNew(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<T>& listLow,
    const std::vector<T>& listHigh)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : staggerBlockCountOnEachDim) totalLeaves *= d;
    size_t leafOffset = octree.size() - totalLeaves;

    auto compressStart_time = std::chrono::high_resolution_clock::now();

    // ========== 逐叶子：Morton码读取真实值 + 8邻居预测值，一次遍历合并完成 ==========
    std::vector<T> leafMins(totalLeaves), leafMaxs(totalLeaves);
    std::vector<T> predictedLow(totalLeaves), predictedHigh(totalLeaves);
    size_t relatedIds[8];
    size_t relatedCount;

    for (size_t i = 0; i < totalLeaves; i++) {
        size_t sx, sy, sz;
        decodeFlatId3D(i, staggerBlockCountOnEachDim, sx, sy, sz);

        // 关键修正：行主序i -> 空间坐标 -> Morton码，才能从Morton序存储的octree里取到正确的值
        uint64_t m = morton3D_encode((uint32_t)sx, (uint32_t)sy, (uint32_t)sz);
        leafMins[i] = octree[leafOffset + m].minVal;
        leafMaxs[i] = octree[leafOffset + m].maxVal;

        relatedUniformIds3D(sx, sy, sz, uniformBlockCountOnEachDim, relatedIds, relatedCount);
        T sumLow = 0, sumHigh = 0;
        size_t countLow = 0, countHigh = 0;
        for (size_t k = 0; k < relatedCount; k++) {
            size_t rid = relatedIds[k];
            if (rid < listLow.size() && !std::isnan(listLow[rid])) { sumLow += listLow[rid]; countLow++; }
            if (rid < listHigh.size() && !std::isnan(listHigh[rid])) { sumHigh += listHigh[rid]; countHigh++; }
        }
        predictedLow[i]  = (countLow  > 0) ? (sumLow  / static_cast<T>(countLow))  : static_cast<T>(0);
        predictedHigh[i] = (countHigh > 0) ? (sumHigh / static_cast<T>(countHigh)) : static_cast<T>(0);
    }

    // ========== min、max 都做逐元素独立量化（用邻居平均值当预测值） ==========
    std::vector<int> typeMinVector, typeMaxVector;
    std::vector<T> outLayerMinData, outLayerMaxData;

    quantizeAgainstPredicted(leafMins, predictedLow,  error_bound, typeMinVector, outLayerMinData);
    quantizeAgainstPredicted(leafMaxs, predictedHigh, error_bound, typeMaxVector, outLayerMaxData);

    auto quantiEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_quanti_time = quantiEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert2 STAGGER] pre_quanti_time (morton read + neighbor predict + quantize): "
              << pre_quanti_time.count() << " seconds" << std::endl;

    // ========== Huffman + Zstd 压缩 ==========
    std::string Minfilename = treeID + "-h2LeafMin";
    std::string Maxfilename = treeID + "-h2LeafMax";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);

    auto huffmanBuildEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanBuild_time = huffmanBuildEnd_time - quantiEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert2 STAGGER] huffmanBuild_time: " << huffmanBuild_time.count()
              << " seconds" << std::endl;

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);
    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);

    auto compressEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> encode_time = compressEnd_time - huffmanBuildEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert2 STAGGER] encode_time: " << encode_time.count()
              << " seconds" << std::endl;


    // ========== 写文件 ==========
    auto writeBegin_time = std::chrono::high_resolution_clock::now();

    writeHilbert2BinaryVector(treeID + "-H2OutMin.bin", outLayerMinData);
    writeHilbert2BinaryVector(treeID + "-H2OutMax.bin", outLayerMaxData);
    {
        std::ofstream outMin(treeID + "-h2LeafMin-compressedIndexData", std::ios::binary);
        outMin.write(reinterpret_cast<const char*>(compressedMin.data()), compressedMin.size());
        outMin.close();
        std::ofstream outMax(treeID + "-h2LeafMax-compressedIndexData", std::ios::binary);
        outMax.write(reinterpret_cast<const char*>(compressedMax.data()), compressedMax.size());
        outMax.close();
    }

    auto writeEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = writeEnd_time - writeBegin_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert2 STAGGER] write_time: " << write_time.count()
              << " seconds" << std::endl;
}


// ============================================================
// UNIFORM 树解压【未改动逻辑，只是确认无需修正】
// 说明：这里 leafMins/leafMaxs 是按行主序 flatId 组织后传给 buildOctree 的，
//   而标准版 buildOctree 内部自己会做"行主序->Morton"转换（散射进正确槽位），
//   所以这一侧不需要改任何东西，天然兼容。
// ============================================================
template<typename T>
std::vector<OctreeNode<T>> decompressOctreeUniformHilbertNew(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& blockCountOnEachDim,
    std::vector<T>* outListLow,
    std::vector<T>* outListHigh)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : blockCountOnEachDim) totalLeaves *= d;

    size_t dimSize = blockCountOnEachDim[0];
    int bits = hilbert2BitsFromDimSize(dimSize);

    auto beginRead = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> huffmanOutMin = loadHuffmanTree(treeID + "-h2LeafMin-Huffman");
    std::vector<unsigned char> huffmanOutMax = loadHuffmanTree(treeID + "-h2LeafMax-Huffman");
    std::vector<unsigned char> compressedMin = readCompressedData(treeID + "-h2LeafMin-compressedIndexData");
    std::vector<unsigned char> compressedMax = readCompressedData(treeID + "-h2LeafMax-compressedIndexData");

    std::vector<T> outLayerMinData = readHilbert2BinaryVector<T>(treeID + "-H2OutMin.bin");
    std::vector<T> outLayerMaxData = readHilbert2BinaryVector<T>(treeID + "-H2OutMax.bin");

    auto endRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readTime = endRead - beginRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time3][Hilbert2 UNIFORM] readIndexTime (" << treeID << "): "
              << readTime.count() << " seconds" << std::endl;

    auto beginHuffman = std::chrono::high_resolution_clock::now();

    std::vector<int> typeMinVector = singleHuffmanDecodeZstdAVL(compressedMin, totalLeaves, huffmanOutMin.data());
    std::vector<int> typeMaxVector = singleHuffmanDecodeZstdAVL(compressedMax, totalLeaves, huffmanOutMax.data());

    auto endHuffman = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanTime = endHuffman - beginHuffman;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4a][Hilbert2 UNIFORM] huffmanDecode (" << treeID << "): "
              << huffmanTime.count() << " seconds" << std::endl;

    auto beginDequant = std::chrono::high_resolution_clock::now();

    std::vector<T> hilbertMins = chainDequantizeHilbert2(typeMinVector, outLayerMinData, error_bound);
    std::vector<T> hilbertMaxs = chainDequantizeHilbert2(typeMaxVector, outLayerMaxData, error_bound);

    auto endDequant = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dequantTime = endDequant - beginDequant;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4b][Hilbert2 UNIFORM] chainDequantMinAndMax (" << treeID << "): "
              << dequantTime.count() << " seconds" << std::endl;

    auto beginScatter = std::chrono::high_resolution_clock::now();

    std::vector<T> leafMins(totalLeaves), leafMaxs(totalLeaves);
    for (uint64_t h = 0; h < totalLeaves; h++) {
        uint64_t x, y, z;
        hilbert2DToXYZ(bits, h, x, y, z);
        size_t flatId = encodeFlatId3D((size_t)x, (size_t)y, (size_t)z, blockCountOnEachDim);
        leafMins[flatId] = hilbertMins[h];
        leafMaxs[flatId] = hilbertMaxs[h];
    }

    if (outListLow)  *outListLow  = leafMins;
    if (outListHigh) *outListHigh = leafMaxs;

    auto endScatter = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> scatterTime = endScatter - beginScatter;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4c][Hilbert2 UNIFORM] scatterToRowMajor (" << treeID << "): "
              << scatterTime.count() << " seconds" << std::endl;

    auto beginBuild = std::chrono::high_resolution_clock::now();

    // buildOctree（标准Morton版）接受行主序输入，内部自行转换，这里不需要改
    std::vector<OctreeNode<T>> tree = buildOctree(leafMins, leafMaxs, blockCountOnEachDim);

    auto endBuild = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> buildTime = endBuild - beginBuild;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4d][Hilbert2 UNIFORM] buildOctree (" << treeID << "): "
              << buildTime.count() << " seconds" << std::endl;

    std::chrono::duration<double> totalReconstructTime = endBuild - endRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4][Hilbert2 UNIFORM] decodeAndReconstructTree total (" << treeID << "): "
              << totalReconstructTime.count() << " seconds" << std::endl;

    return tree;
}


// ============================================================
// STAGGER 树解压【未改动逻辑，只是确认无需修正，理由同UNIFORM解压】
// ============================================================
template<typename T>
std::vector<OctreeNode<T>> decompressOctreeStaggerHilbertNew(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<T>& listLow,
    const std::vector<T>& listHigh)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : staggerBlockCountOnEachDim) totalLeaves *= d;

    auto beginRead = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> huffmanOutMin = loadHuffmanTree(treeID + "-h2LeafMin-Huffman");
    std::vector<unsigned char> huffmanOutMax = loadHuffmanTree(treeID + "-h2LeafMax-Huffman");
    std::vector<unsigned char> compressedMin = readCompressedData(treeID + "-h2LeafMin-compressedIndexData");
    std::vector<unsigned char> compressedMax = readCompressedData(treeID + "-h2LeafMax-compressedIndexData");

    std::vector<T> outLayerMinData = readHilbert2BinaryVector<T>(treeID + "-H2OutMin.bin");
    std::vector<T> outLayerMaxData = readHilbert2BinaryVector<T>(treeID + "-H2OutMax.bin");

    auto endRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readTime = endRead - beginRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time3][Hilbert2 STAGGER] readIndexTime (" << treeID << "): "
              << readTime.count() << " seconds" << std::endl;

    auto beginHuffman = std::chrono::high_resolution_clock::now();

    std::vector<int> typeMinVector = singleHuffmanDecodeZstdAVL(compressedMin, totalLeaves, huffmanOutMin.data());
    std::vector<int> typeMaxVector = singleHuffmanDecodeZstdAVL(compressedMax, totalLeaves, huffmanOutMax.data());

    auto endHuffman = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanTime = endHuffman - beginHuffman;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4a][Hilbert2 STAGGER] huffmanDecode (" << treeID << "): "
              << huffmanTime.count() << " seconds" << std::endl;

    auto beginPredict = std::chrono::high_resolution_clock::now();

    std::vector<T> predictedLow(totalLeaves), predictedHigh(totalLeaves);
    size_t relatedIds[8];
    size_t relatedCount;

    for (size_t i = 0; i < totalLeaves; i++) {
        size_t sx, sy, sz;
        decodeFlatId3D(i, staggerBlockCountOnEachDim, sx, sy, sz);
        relatedUniformIds3D(sx, sy, sz, uniformBlockCountOnEachDim, relatedIds, relatedCount);

        T sumLow = 0, sumHigh = 0;
        size_t countLow = 0, countHigh = 0;
        for (size_t k = 0; k < relatedCount; k++) {
            size_t rid = relatedIds[k];
            if (rid < listLow.size() && !std::isnan(listLow[rid])) { sumLow += listLow[rid]; countLow++; }
            if (rid < listHigh.size() && !std::isnan(listHigh[rid])) { sumHigh += listHigh[rid]; countHigh++; }
        }
        predictedLow[i]  = (countLow  > 0) ? (sumLow  / static_cast<T>(countLow))  : static_cast<T>(0);
        predictedHigh[i] = (countHigh > 0) ? (sumHigh / static_cast<T>(countHigh)) : static_cast<T>(0);
    }

    auto endPredict = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> predictTime = endPredict - beginPredict;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4b][Hilbert2 STAGGER] neighborPredict (" << treeID << "): "
              << predictTime.count() << " seconds" << std::endl;

    auto beginDequant = std::chrono::high_resolution_clock::now();

    std::vector<T> leafMins = dequantizeAgainstPredicted(typeMinVector, predictedLow,  outLayerMinData, error_bound);
    std::vector<T> leafMaxs = dequantizeAgainstPredicted(typeMaxVector, predictedHigh, outLayerMaxData, error_bound);

    auto endDequant = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dequantTime = endDequant - beginDequant;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4c][Hilbert2 STAGGER] dequantMinAndMax (" << treeID << "): "
              << dequantTime.count() << " seconds" << std::endl;

    auto beginBuild = std::chrono::high_resolution_clock::now();

    std::vector<OctreeNode<T>> tree = buildOctree(leafMins, leafMaxs, staggerBlockCountOnEachDim);

    auto endBuild = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> buildTime = endBuild - beginBuild;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4d][Hilbert2 STAGGER] buildOctree (" << treeID << "): "
              << buildTime.count() << " seconds" << std::endl;

    std::chrono::duration<double> totalReconstructTime = endBuild - endRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4][Hilbert2 STAGGER] decodeAndReconstructTree total (" << treeID << "): "
              << totalReconstructTime.count() << " seconds" << std::endl;

    return tree;
}


// ============================================================
// 显式实例化
// ============================================================
template void compressOctreeUniformHilbertNew<float>(
    const std::vector<OctreeNode<float>>&, float, const std::string&,
    const std::vector<size_t>&, std::vector<float>*, std::vector<float>*);
template void compressOctreeUniformHilbertNew<double>(
    const std::vector<OctreeNode<double>>&, double, const std::string&,
    const std::vector<size_t>&, std::vector<double>*, std::vector<double>*);

template void compressOctreeStaggerHilbertNew<float>(
    const std::vector<OctreeNode<float>>&, float, const std::string&,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const std::vector<float>&, const std::vector<float>&);
template void compressOctreeStaggerHilbertNew<double>(
    const std::vector<OctreeNode<double>>&, double, const std::string&,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const std::vector<double>&, const std::vector<double>&);

template std::vector<OctreeNode<float>> decompressOctreeUniformHilbertNew<float>(
    const std::string&, float, const std::vector<size_t>&,
    std::vector<float>*, std::vector<float>*);
template std::vector<OctreeNode<double>> decompressOctreeUniformHilbertNew<double>(
    const std::string&, double, const std::vector<size_t>&,
    std::vector<double>*, std::vector<double>*);

template std::vector<OctreeNode<float>> decompressOctreeStaggerHilbertNew<float>(
    const std::string&, float, const std::vector<size_t>&, const std::vector<size_t>&,
    const std::vector<float>&, const std::vector<float>&);
template std::vector<OctreeNode<double>> decompressOctreeStaggerHilbertNew<double>(
    const std::string&, double, const std::vector<size_t>&, const std::vector<size_t>&,
    const std::vector<double>&, const std::vector<double>&);

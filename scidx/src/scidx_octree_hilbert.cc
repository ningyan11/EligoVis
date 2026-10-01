#include "scidx_octree_hilbert.h"
#include <scidx_Huffman.h>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <mpi.h>
#include <zstd.h>

using namespace scidx;

// ============================================================
// 非模板函数，来自 scidx_avl.cc，跨文件复用（跟原 scidx_octree.cc 一致）
// ============================================================
HuffmanTree* processHuffmanTreeNaiveAVL(std::vector<int> treeType, std::string& fileName);
std::vector<unsigned char> singleHuffmanEncodeZstdAVL(std::vector<int> type, HuffmanTree *huffmanTree);
std::vector<unsigned char> loadHuffmanTree(const std::string& filename);
std::vector<unsigned char> readCompressedData(const std::string& filename);
std::vector<int> singleHuffmanDecodeZstdAVL(std::vector<unsigned char> compressedByZstdData, size_t typeSize, unsigned char *s);

// ============================================================
// 工具函数：读/写 outlier 二进制数据
// ============================================================
template<typename T>
std::vector<T> readHilbertBinaryVector(const std::string& filename) {
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
void writeHilbertBinaryVector(const std::string& filename, const std::vector<T>& data) {
    std::ofstream outFile(filename, std::ios::binary);
    size_t size = data.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    outFile.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
    outFile.close();
}

// ============================================================
// 链式量化 / 反量化（min、max 通用，跟原来 min 用的算法完全一致，
// 只是现在 min 和 max 都走这一套，且作用在 Hilbert 序排好的数组上）
// ============================================================
template<typename T>
std::pair<std::vector<int>, std::vector<T>> chainQuantizeHilbert(
    const std::vector<T>& data, T error_bound)
{
    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T checkRadius = (quantization_intervals - 1) * error_bound;
    const T interval = 2 * error_bound;
    const T recip_precision = 1 / error_bound;

    std::vector<int> typeVector;
    std::vector<T> outLayerData;
    typeVector.reserve(data.size());

    T predData = 0, reconstructed = 0;

    for (size_t i = 0; i < data.size(); i++) {
        T curData = data[i];
        if (i == 0) {
            typeVector.push_back(0);
            outLayerData.push_back(curData);
            reconstructed = curData;
        } else {
            T predAbsErr = std::abs(curData - predData);
            if (predAbsErr < checkRadius) {
                int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                if (curData >= predData) {
                    typeVector.push_back(intvRadius + state);
                    reconstructed = predData + state * interval;
                } else {
                    typeVector.push_back(intvRadius - state);
                    reconstructed = predData - state * interval;
                }
            } else {
                typeVector.push_back(0);
                outLayerData.push_back(curData);
                reconstructed = curData;
            }
        }
        predData = reconstructed;
    }

    return {typeVector, outLayerData};
}

template<typename T>
std::vector<T> chainDequantizeHilbert(
    const std::vector<int>& typeVector,
    const std::vector<T>& outLayerData,
    T error_bound)
{
    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T interval = 2 * error_bound;

    std::vector<T> reconstructed(typeVector.size());
    T predData = 0;
    size_t outlierIdx = 0;

    for (size_t i = 0; i < typeVector.size(); i++) {
        int t = typeVector[i];
        T curVal;
        if (t == 0) {
            curVal = outLayerData[outlierIdx++];
        } else {
            int state = t - intvRadius;
            curVal = predData + state * interval;
        }
        reconstructed[i] = curVal;
        predData = curVal;
    }
    return reconstructed;
}

// ============================================================
// uniform Octree 压缩（Hilbert 版）
// ============================================================
template<typename T>
void compressOctreeUniformHilbert(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& blockCountOnEachDim)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : blockCountOnEachDim) totalLeaves *= d;
    size_t leafOffset = octree.size() - totalLeaves;

    size_t dimSize = blockCountOnEachDim[0];
    int bits = hilbertBitsFromDimSize(dimSize);

    // 原始 leafMins/leafMaxs，索引 i 对应行主序 flat id：
    // i = x + y*dimSize + z*dimSize*dimSize（跟 buildOctree 的叶子排列约定一致）
    std::vector<T> leafMins(totalLeaves), leafMaxs(totalLeaves);
    for (size_t i = 0; i < totalLeaves; i++) {
        leafMins[i] = octree[leafOffset + i].minVal;
        leafMaxs[i] = octree[leafOffset + i].maxVal;
    }

    auto compressStart_time = std::chrono::high_resolution_clock::now();

    // ========== 按 Hilbert 曲线顺序重排 min/max ==========
    std::vector<T> hilbertMins(totalLeaves), hilbertMaxs(totalLeaves);
    for (uint64_t h = 0; h < totalLeaves; h++) {
        uint64_t x, y, z;
        hilbertDToXYZ(bits, h, x, y, z);
        size_t flatId = x + y * dimSize + z * dimSize * dimSize;
        hilbertMins[h] = leafMins[flatId];
        hilbertMaxs[h] = leafMaxs[flatId];
    }

    // ========== min、max 都走链式量化（沿 Hilbert 序） ==========
    auto [typeMinVector, outLayerMinData] = chainQuantizeHilbert(hilbertMins, error_bound);
    auto [typeMaxVector, outLayerMaxData] = chainQuantizeHilbert(hilbertMaxs, error_bound);

    auto quantiEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_quanti_time = quantiEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert UNIFORM] pre_quanti_time (hilbert reorder + chain quantize min&max): "
              << pre_quanti_time.count() << " seconds" << std::endl;

    // ========== Huffman + Zstd 压缩 ==========
    std::string Minfilename = treeID + "-hilbertLeafMin";
    std::string Maxfilename = treeID + "-hilbertLeafMax";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);

    auto huffmanBuildEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanBuild_time = huffmanBuildEnd_time - quantiEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert UNIFORM] huffmanBuild_time: " << huffmanBuild_time.count()
              << " seconds" << std::endl;

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);
    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);

    auto compressEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> encode_time = compressEnd_time - huffmanBuildEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert UNIFORM] encode_time: " << encode_time.count()
              << " seconds" << std::endl;

    // ========== 写文件 ==========
    auto writeBegin_time = std::chrono::high_resolution_clock::now();

    writeHilbertBinaryVector(treeID + "-HilbertOutMin.bin", outLayerMinData);
    writeHilbertBinaryVector(treeID + "-HilbertOutMax.bin", outLayerMaxData);

    {
        std::ofstream outMin(treeID + "-hilbertLeafMin-compressedIndexData", std::ios::binary);
        outMin.write(reinterpret_cast<const char*>(compressedMin.data()), compressedMin.size());
        outMin.close();

        std::ofstream outMax(treeID + "-hilbertLeafMax-compressedIndexData", std::ios::binary);
        outMax.write(reinterpret_cast<const char*>(compressedMax.data()), compressedMax.size());
        outMax.close();
    }

    auto writeEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = writeEnd_time - writeBegin_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert UNIFORM] write_time: " << write_time.count()
              << " seconds" << std::endl;
}

// ============================================================
// stagger Octree 压缩（Hilbert 版）
// 注意：不再依赖 UNIFORM 树的 listHigh 做跨树预测，
// STAGGER 树自己内部沿 Hilbert 序对 min/max 各自做链式预测即可。
// ============================================================
template<typename T>
void compressOctreeStaggerHilbert(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& staggerBlockCountOnEachDim)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : staggerBlockCountOnEachDim) totalLeaves *= d;
    size_t leafOffset = octree.size() - totalLeaves;

    size_t dimSize = staggerBlockCountOnEachDim[0];
    int bits = hilbertBitsFromDimSize(dimSize);

    std::vector<T> leafMins(totalLeaves), leafMaxs(totalLeaves);
    for (size_t i = 0; i < totalLeaves; i++) {
        leafMins[i] = octree[leafOffset + i].minVal;
        leafMaxs[i] = octree[leafOffset + i].maxVal;
    }

    auto compressStart_time = std::chrono::high_resolution_clock::now();

    std::vector<T> hilbertMins(totalLeaves), hilbertMaxs(totalLeaves);
    for (uint64_t h = 0; h < totalLeaves; h++) {
        uint64_t x, y, z;
        hilbertDToXYZ(bits, h, x, y, z);
        size_t flatId = x + y * dimSize + z * dimSize * dimSize;
        hilbertMins[h] = leafMins[flatId];
        hilbertMaxs[h] = leafMaxs[flatId];
    }

    auto [typeMinVector, outLayerMinData] = chainQuantizeHilbert(hilbertMins, error_bound);
    auto [typeMaxVector, outLayerMaxData] = chainQuantizeHilbert(hilbertMaxs, error_bound);

    auto quantiEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_quanti_time = quantiEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert STAGGER] pre_quanti_time (hilbert reorder + chain quantize min&max): "
              << pre_quanti_time.count() << " seconds" << std::endl;

    std::string Minfilename = treeID + "-hilbertLeafMin";
    std::string Maxfilename = treeID + "-hilbertLeafMax";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);

    auto huffmanBuildEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanBuild_time = huffmanBuildEnd_time - quantiEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert STAGGER] huffmanBuild_time: " << huffmanBuild_time.count()
              << " seconds" << std::endl;

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);
    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);

    auto compressEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> encode_time = compressEnd_time - huffmanBuildEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert STAGGER] encode_time: " << encode_time.count()
              << " seconds" << std::endl;

    auto writeBegin_time = std::chrono::high_resolution_clock::now();

    writeHilbertBinaryVector(treeID + "-HilbertOutMin.bin", outLayerMinData);
    writeHilbertBinaryVector(treeID + "-HilbertOutMax.bin", outLayerMaxData);

    {
        std::ofstream outMin(treeID + "-hilbertLeafMin-compressedIndexData", std::ios::binary);
        outMin.write(reinterpret_cast<const char*>(compressedMin.data()), compressedMin.size());
        outMin.close();

        std::ofstream outMax(treeID + "-hilbertLeafMax-compressedIndexData", std::ios::binary);
        outMax.write(reinterpret_cast<const char*>(compressedMax.data()), compressedMax.size());
        outMax.close();
    }

    auto writeEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = writeEnd_time - writeBegin_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Hilbert STAGGER] write_time: " << write_time.count()
              << " seconds" << std::endl;
}

// ============================================================
// uniform Octree 解压（Hilbert 版），分段计时对齐之前约定的风格
//   [Time3]  读文件（I/O）
//   [Time4a] Huffman解码（CPU）
//   [Time4b] 链式反量化 min+max（沿Hilbert序，CPU）
//   [Time4c] 把Hilbert序结果散射回行主序数组（CPU）
//   [Time4d] 建树（CPU）
//   [Time4]  总和
// ============================================================
template<typename T>
std::vector<OctreeNode<T>> decompressOctreeUniformHilbert(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& blockCountOnEachDim)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : blockCountOnEachDim) totalLeaves *= d;

    size_t dimSize = blockCountOnEachDim[0];
    int bits = hilbertBitsFromDimSize(dimSize);

    // ========== [Time3] 读文件 ==========
    auto beginRead = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> huffmanOutMin = loadHuffmanTree(treeID + "-hilbertLeafMin-Huffman");
    std::vector<unsigned char> huffmanOutMax = loadHuffmanTree(treeID + "-hilbertLeafMax-Huffman");
    std::vector<unsigned char> compressedMin = readCompressedData(treeID + "-hilbertLeafMin-compressedIndexData");
    std::vector<unsigned char> compressedMax = readCompressedData(treeID + "-hilbertLeafMax-compressedIndexData");

    std::vector<T> outLayerMinData = readHilbertBinaryVector<T>(treeID + "-HilbertOutMin.bin");
    std::vector<T> outLayerMaxData = readHilbertBinaryVector<T>(treeID + "-HilbertOutMax.bin");

    auto endRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readTime = endRead - beginRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time3][Hilbert UNIFORM] readIndexTime (" << treeID << "): "
              << readTime.count() << " seconds" << std::endl;

    // ========== [Time4a] Huffman解码 ==========
    auto beginHuffman = std::chrono::high_resolution_clock::now();

    std::vector<int> typeMinVector = singleHuffmanDecodeZstdAVL(compressedMin, totalLeaves, huffmanOutMin.data());
    std::vector<int> typeMaxVector = singleHuffmanDecodeZstdAVL(compressedMax, totalLeaves, huffmanOutMax.data());

    auto endHuffman = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanTime = endHuffman - beginHuffman;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4a][Hilbert UNIFORM] huffmanDecode (" << treeID << "): "
              << huffmanTime.count() << " seconds" << std::endl;

    // ========== [Time4b] 链式反量化 min+max（沿Hilbert序） ==========
    auto beginDequant = std::chrono::high_resolution_clock::now();

    std::vector<T> hilbertMins = chainDequantizeHilbert(typeMinVector, outLayerMinData, error_bound);
    std::vector<T> hilbertMaxs = chainDequantizeHilbert(typeMaxVector, outLayerMaxData, error_bound);

    auto endDequant = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dequantTime = endDequant - beginDequant;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4b][Hilbert UNIFORM] chainDequantMinAndMax (" << treeID << "): "
              << dequantTime.count() << " seconds" << std::endl;

    // ========== [Time4c] 散射回行主序数组 ==========
    auto beginScatter = std::chrono::high_resolution_clock::now();

    std::vector<T> leafMins(totalLeaves), leafMaxs(totalLeaves);
    for (uint64_t h = 0; h < totalLeaves; h++) {
        uint64_t x, y, z;
        hilbertDToXYZ(bits, h, x, y, z);
        size_t flatId = x + y * dimSize + z * dimSize * dimSize;
        leafMins[flatId] = hilbertMins[h];
        leafMaxs[flatId] = hilbertMaxs[h];
    }

    auto endScatter = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> scatterTime = endScatter - beginScatter;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4c][Hilbert UNIFORM] scatterToRowMajor (" << treeID << "): "
              << scatterTime.count() << " seconds" << std::endl;

    // ========== [Time4d] 建树 ==========
    auto beginBuild = std::chrono::high_resolution_clock::now();

    std::vector<OctreeNode<T>> tree = buildOctree(leafMins, leafMaxs, blockCountOnEachDim);

    auto endBuild = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> buildTime = endBuild - beginBuild;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4d][Hilbert UNIFORM] buildOctree (" << treeID << "): "
              << buildTime.count() << " seconds" << std::endl;

    // ========== [Time4] 总和 ==========
    std::chrono::duration<double> totalReconstructTime = endBuild - endRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4][Hilbert UNIFORM] decodeAndReconstructTree total (" << treeID << "): "
              << totalReconstructTime.count() << " seconds" << std::endl;

    return tree;
}

// ============================================================
// stagger Octree 解压（Hilbert 版），结构与 UNIFORM 完全对齐
// ============================================================
template<typename T>
std::vector<OctreeNode<T>> decompressOctreeStaggerHilbert(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& staggerBlockCountOnEachDim)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : staggerBlockCountOnEachDim) totalLeaves *= d;

    size_t dimSize = staggerBlockCountOnEachDim[0];
    int bits = hilbertBitsFromDimSize(dimSize);

    // ========== [Time3] 读文件 ==========
    auto beginRead = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> huffmanOutMin = loadHuffmanTree(treeID + "-hilbertLeafMin-Huffman");
    std::vector<unsigned char> huffmanOutMax = loadHuffmanTree(treeID + "-hilbertLeafMax-Huffman");
    std::vector<unsigned char> compressedMin = readCompressedData(treeID + "-hilbertLeafMin-compressedIndexData");
    std::vector<unsigned char> compressedMax = readCompressedData(treeID + "-hilbertLeafMax-compressedIndexData");

    std::vector<T> outLayerMinData = readHilbertBinaryVector<T>(treeID + "-HilbertOutMin.bin");
    std::vector<T> outLayerMaxData = readHilbertBinaryVector<T>(treeID + "-HilbertOutMax.bin");

    auto endRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readTime = endRead - beginRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time3][Hilbert STAGGER] readIndexTime (" << treeID << "): "
              << readTime.count() << " seconds" << std::endl;

    // ========== [Time4a] Huffman解码 ==========
    auto beginHuffman = std::chrono::high_resolution_clock::now();

    std::vector<int> typeMinVector = singleHuffmanDecodeZstdAVL(compressedMin, totalLeaves, huffmanOutMin.data());
    std::vector<int> typeMaxVector = singleHuffmanDecodeZstdAVL(compressedMax, totalLeaves, huffmanOutMax.data());

    auto endHuffman = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanTime = endHuffman - beginHuffman;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4a][Hilbert STAGGER] huffmanDecode (" << treeID << "): "
              << huffmanTime.count() << " seconds" << std::endl;

    // ========== [Time4b] 链式反量化 min+max ==========
    auto beginDequant = std::chrono::high_resolution_clock::now();

    std::vector<T> hilbertMins = chainDequantizeHilbert(typeMinVector, outLayerMinData, error_bound);
    std::vector<T> hilbertMaxs = chainDequantizeHilbert(typeMaxVector, outLayerMaxData, error_bound);

    auto endDequant = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dequantTime = endDequant - beginDequant;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4b][Hilbert STAGGER] chainDequantMinAndMax (" << treeID << "): "
              << dequantTime.count() << " seconds" << std::endl;

    // ========== [Time4c] 散射回行主序数组 ==========
    auto beginScatter = std::chrono::high_resolution_clock::now();

    std::vector<T> leafMins(totalLeaves), leafMaxs(totalLeaves);
    for (uint64_t h = 0; h < totalLeaves; h++) {
        uint64_t x, y, z;
        hilbertDToXYZ(bits, h, x, y, z);
        size_t flatId = x + y * dimSize + z * dimSize * dimSize;
        leafMins[flatId] = hilbertMins[h];
        leafMaxs[flatId] = hilbertMaxs[h];
    }

    auto endScatter = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> scatterTime = endScatter - beginScatter;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4c][Hilbert STAGGER] scatterToRowMajor (" << treeID << "): "
              << scatterTime.count() << " seconds" << std::endl;

    // ========== [Time4d] 建树 ==========
    auto beginBuild = std::chrono::high_resolution_clock::now();

    std::vector<OctreeNode<T>> tree = buildOctree(leafMins, leafMaxs, staggerBlockCountOnEachDim);

    auto endBuild = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> buildTime = endBuild - beginBuild;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4d][Hilbert STAGGER] buildOctree (" << treeID << "): "
              << buildTime.count() << " seconds" << std::endl;

    // ========== [Time4] 总和 ==========
    std::chrono::duration<double> totalReconstructTime = endBuild - endRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4][Hilbert STAGGER] decodeAndReconstructTree total (" << treeID << "): "
              << totalReconstructTime.count() << " seconds" << std::endl;

    return tree;
}

// ============================================================
// 显式实例化
// ============================================================
template void compressOctreeUniformHilbert<float>(
    const std::vector<OctreeNode<float>>&, float, const std::string&, const std::vector<size_t>&);
template void compressOctreeUniformHilbert<double>(
    const std::vector<OctreeNode<double>>&, double, const std::string&, const std::vector<size_t>&);

template void compressOctreeStaggerHilbert<float>(
    const std::vector<OctreeNode<float>>&, float, const std::string&, const std::vector<size_t>&);
template void compressOctreeStaggerHilbert<double>(
    const std::vector<OctreeNode<double>>&, double, const std::string&, const std::vector<size_t>&);

template std::vector<OctreeNode<float>> decompressOctreeUniformHilbert<float>(
    const std::string&, float, const std::vector<size_t>&);
template std::vector<OctreeNode<double>> decompressOctreeUniformHilbert<double>(
    const std::string&, double, const std::vector<size_t>&);

template std::vector<OctreeNode<float>> decompressOctreeStaggerHilbert<float>(
    const std::string&, float, const std::vector<size_t>&);
template std::vector<OctreeNode<double>> decompressOctreeStaggerHilbert<double>(
    const std::string&, double, const std::vector<size_t>&);
#include "scidx_octree.h"
#include <scidx_octree_interval.h>
#include <scidx_Huffman.h>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <sstream>
#include <mpi.h>
#include <zstd.h>
#include <array>  



using namespace scidx;
std::string formatSize(size_t bytes);

template <typename T>
struct MinMaxPair {
    T reconstructedMin;
    T originalMax;
};



// ============================================================
// 工具函数
// ============================================================




// ============================================================
// [新增-性能优化] 3维专用、无堆分配版本的坐标转换 + 邻居查找
// 仅用于 decompressOctreeStagger 的热点循环，避免每次迭代产生
// std::vector 的堆分配/释放开销
// ============================================================
inline void positionToIndicesColMajor3D(
    size_t position,
    const std::vector<size_t>& shape,
    std::array<size_t, 3>& indices)
{
    indices[0] = position % shape[0];
    position /= shape[0];
    indices[1] = position % shape[1];
    position /= shape[1];
    indices[2] = position;
}

inline void getRelated8UniformBlockIds3D(
    const std::array<size_t, 3>& staggerCoord,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    std::array<size_t, 8>& relatedIds,
    size_t& relatedCount)
{
    relatedCount = 0;

    for (size_t mask = 0; mask < 8; mask++) {
        size_t offsetX = mask & 1;
        size_t offsetY = (mask >> 1) & 1;
        size_t offsetZ = (mask >> 2) & 1;

        size_t ux, uy, uz;
        bool valid = true;

        if (staggerCoord[0] == 0) {
            ux = 0;
        } else {
            ux = staggerCoord[0] - 1 + offsetX;
            if (ux >= uniformBlockCountOnEachDim[0]) valid = false;
        }

        if (valid) {
            if (staggerCoord[1] == 0) {
                uy = 0;
            } else {
                uy = staggerCoord[1] - 1 + offsetY;
                if (uy >= uniformBlockCountOnEachDim[1]) valid = false;
            }
        }

        if (valid) {
            if (staggerCoord[2] == 0) {
                uz = 0;
            } else {
                uz = staggerCoord[2] - 1 + offsetZ;
                if (uz >= uniformBlockCountOnEachDim[2]) valid = false;
            }
        }

        if (valid) {
            relatedIds[relatedCount++] = ux
                + uy * uniformBlockCountOnEachDim[0]
                + uz * uniformBlockCountOnEachDim[0] * uniformBlockCountOnEachDim[1];
        }
    }
}

std::vector<size_t> positionToIndicesColMajor(size_t position, const std::vector<size_t>& shape);

size_t indicesToPosition(const std::vector<size_t>& shape, const std::vector<size_t>& indices);

std::vector<size_t> getRelated8UniformBlockIds(
    const std::vector<size_t>& staggerBlockCoord,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    size_t nDim)
{
    std::vector<size_t> relatedIds;
    size_t numCombinations = 1 << nDim;

    for (size_t mask = 0; mask < numCombinations; mask++) {
        std::vector<size_t> uniformCoord(nDim);
        bool valid = true;

        for (size_t d = 0; d < nDim; d++) {
            size_t offset = (mask >> d) & 1;
            if (staggerBlockCoord[d] == 0) {
                uniformCoord[d] = 0;
            } else {
                uniformCoord[d] = staggerBlockCoord[d] - 1 + offset;
            }
            if (uniformCoord[d] >= uniformBlockCountOnEachDim[d]) {
                valid = false;
                break;
            }
        }

        if (valid) {
            size_t uid = indicesToPosition(uniformBlockCountOnEachDim, uniformCoord);
            relatedIds.push_back(uid);
        }
    }
    return relatedIds;
}

void exportCountMatrixToCSVNew(const std::vector<std::vector<int>>& countMatrix, const std::string& filename);

std::pair<int, int> getBinIndicesNew(double low, double high, double lowMin, double lowMax, double highMin, double highMax, int lowBinCount, int highBinCount);

template<typename T>
int findLowBin(T lowValue, T lowMin, T lowMax, int lowBinCount) {
    return std::floor((lowValue - lowMin) / (lowMax - lowMin) * lowBinCount);
}

template<typename T>
std::tuple<T, T, T, T, int, int, std::vector<std::pair<int, T>>>
loadBinMappingAndStats(const std::string& csvFile)
{
    T lowMin, lowMax, highMin, highMax;
    int lowBinCount, highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMap;

    std::ifstream file(csvFile);
    std::string line;

    if (file.is_open()) {
        std::getline(file, line);
        std::getline(file, line);
        std::stringstream ss(line);
        char comma;
        ss >> lowMin >> comma >> lowMax >> comma >> highMin >> comma >> highMax
           >> comma >> lowBinCount >> comma >> highBinCount;

        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream binStream(line);
            int lowBin;
            T highBin;
            binStream >> lowBin >> comma >> highBin;
            lowHighBinMap.push_back(std::make_pair(lowBin, highBin));
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << csvFile << std::endl;
    }

    return std::make_tuple(lowMin, lowMax, highMin, highMax,
                           lowBinCount, highBinCount, lowHighBinMap);
}

template<typename T>
void calculateConditionalProbability(
    const std::vector<MinMaxPair<T>>& minMaxPairs,
    int lowBinCount, int highBinCount,
    const std::string& csvFile)
{
    T lowMin = minMaxPairs[0].reconstructedMin, lowMax = minMaxPairs[0].reconstructedMin;
    T highMin = minMaxPairs[0].originalMax,     highMax = minMaxPairs[0].originalMax;

    for (const auto& pair : minMaxPairs) {
        if (pair.reconstructedMin < lowMin) lowMin = pair.reconstructedMin;
        if (pair.reconstructedMin > lowMax) lowMax = pair.reconstructedMin;
        if (pair.originalMax < highMin)     highMin = pair.originalMax;
        if (pair.originalMax > highMax)     highMax = pair.originalMax;
    }

    T epsilon = static_cast<T>(1e-6);
    lowMax  += epsilon;
    highMax += epsilon;

    std::vector<std::vector<int>> countMatrix(lowBinCount, std::vector<int>(highBinCount, 0));
    std::vector<int> lowBinTotalCount(lowBinCount, 0);

    for (const auto& pair : minMaxPairs) {
        auto [lowBin, highBin] = getBinIndicesNew(
            pair.reconstructedMin, pair.originalMax,
            lowMin, lowMax, highMin, highMax,
            lowBinCount, highBinCount);
        countMatrix[lowBin][highBin]++;
        lowBinTotalCount[lowBin]++;
    }

    exportCountMatrixToCSVNew(countMatrix, "count_matrix.csv");

    std::vector<std::pair<int, double>> nonZeroBins;
    for (int i = 0; i < lowBinCount; ++i) {
        if (lowBinTotalCount[i] == 0) continue;
        double expectedValue = 0.0;
        for (int j = 0; j < highBinCount; ++j) {
            double conditionalProb = static_cast<double>(countMatrix[i][j]) / lowBinTotalCount[i];
            expectedValue += conditionalProb * j;
        }
        if (expectedValue > 0)
            nonZeroBins.emplace_back(i, expectedValue);
    }

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto start_time = std::chrono::high_resolution_clock::now();

    std::ofstream file(csvFile);
    if (file.is_open()) {
        file << "lowMin,lowMax,highMin,highMax,lowBinCount,highBinCount\n";
        file << lowMin << "," << lowMax << "," << highMin << "," << highMax << ","
             << lowBinCount << "," << highBinCount << "\n";
        file << "Low Bin,Expected High Bin\n";
        for (const auto& bin : nonZeroBins) {
            file << bin.first << "," << bin.second << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << csvFile << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = end_time - start_time;
    std::cout << "Current MPI rank: " << rank
              << "[writing time a] write bayes: " << write_time.count() << " seconds" << std::endl;
}

template<typename T>
auto computeHighBayesOptimizedAVLFixOrderNew(
    const std::vector<MinMaxPair<T>>& minMaxPairs,
    T error_bound,
    std::string treeID)
-> std::tuple<std::vector<int>, std::vector<T>, std::vector<T>>
{
    T lowMin, lowMax, highMin, highMax;
    int lowBinCount, highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMapping;

    std::string csvFileName = treeID + "-low_high_bin_mapping.csv";
    std::tie(lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, lowHighBinMapping) =
        loadBinMappingAndStats<T>(csvFileName);

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;

    auto getHighBinMidValue = [&](T highBin) {
        T binWidth = (highMax - highMin) / highBinCount;
        return highMin + highBin * binWidth + binWidth / 2.0;
    };

    std::vector<int> type(minMaxPairs.size(), 0);
    std::vector<T> outLayerData;
    std::vector<T> predictedHighArray(minMaxPairs.size());

    for (size_t i = 0; i < minMaxPairs.size(); i++) {
        T curLow  = minMaxPairs[i].reconstructedMin;
        T curHigh = minMaxPairs[i].originalMax;

        int lowBin      = findLowBin(curLow, lowMin, lowMax, lowBinCount);
        T highBin       = lowHighBinMapping[lowBin].second;
        T predictedHigh = getHighBinMidValue(highBin);

        predictedHighArray[i] = predictedHigh;

        T predAbsErr = std::abs(predictedHigh - curHigh);
        if (predAbsErr < checkRadius) {
            int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
            if (curHigh >= predictedHigh) {
                type[i] = intvRadius + state;
            } else {
                type[i] = intvRadius - state;
            }
        } else {
            type[i] = 0;
            outLayerData.push_back(curHigh);
        }
    }

    return {type, outLayerData, predictedHighArray};
}

HuffmanTree* processHuffmanTreeNaiveAVL(std::vector<int> treeType, std::string& fileName);

std::vector<unsigned char> compressWithZstdAVL(const std::vector<unsigned char> &data, size_t dataLength);

// 对 type 进行 Huffman encode 和 zstd 压缩
std::vector<unsigned char> singleHuffmanEncodeZstdAVL(std::vector<int> type, HuffmanTree *huffmanTree);

// 非模板函数，来自 scidx_avl.cc，可以安全跨文件复用
std::vector<unsigned char> loadHuffmanTree(const std::string& filename);
std::vector<unsigned char> readCompressedData(const std::string& filename);
std::vector<int> singleHuffmanDecodeZstdAVL(std::vector<unsigned char> compressedByZstdData, size_t typeSize, unsigned char *s);

// 读回 outlier 二进制数据（对应压缩时 writeBinary 的格式）
template<typename T>
std::vector<T> readOctreeBinaryVector(const std::string& filename) {
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

// 顺序反量化：对应压缩时 leafMins 的构建逻辑，逐个复原
template<typename T>
std::vector<T> sequentialDequantize(
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

// [新增] 只负责读取 bin mapping csv 文件（纯I/O，从原 predictHighFromBinMapping 中拆分出来，
// 目的是让 decompressOctreeUniform 能把这次磁盘读取归入"读文件"阶段统一计时，
// 而不是让它混在"重建"阶段的耗时里）
template<typename T>
std::tuple<T, T, T, T, int, int, std::vector<std::pair<int, T>>>
loadBinMappingForPredict(const std::string& treeID)
{
    T lowMin, lowMax, highMin, highMax;
    int lowBinCount, highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMapping;

    std::ifstream file(treeID + "-low_high_bin_mapping.csv");
    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    {
        std::stringstream ss(line);
        char comma;
        ss >> lowMin >> comma >> lowMax >> comma >> highMin >> comma >> highMax
           >> comma >> lowBinCount >> comma >> highBinCount;
    }
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream binStream(line);
        int lowBin; T highBin; char comma;
        binStream >> lowBin >> comma >> highBin;
        lowHighBinMapping.push_back({lowBin, highBin});
    }
    file.close();

    return std::make_tuple(lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, lowHighBinMapping);
}

// Bayes 查表预测（对应 compressOctreeUniform 里的 predictedHigh 计算，
// 只依赖 leafMins，跟 curHigh 无关，所以解压时可以在不知道 leafMaxs 的情况下重算）
// 保留原函数（当前未被内部调用，避免破坏可能存在的外部引用）
template<typename T>
std::vector<T> predictHighFromBinMapping(
    const std::vector<T>& leafMins,
    const std::string& treeID)
{
    T lowMin, lowMax, highMin, highMax;
    int lowBinCount, highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMapping;

    std::ifstream file(treeID + "-low_high_bin_mapping.csv");
    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    {
        std::stringstream ss(line);
        char comma;
        ss >> lowMin >> comma >> lowMax >> comma >> highMin >> comma >> highMax
           >> comma >> lowBinCount >> comma >> highBinCount;
    }
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream binStream(line);
        int lowBin; T highBin; char comma;
        binStream >> lowBin >> comma >> highBin;
        lowHighBinMapping.push_back({lowBin, highBin});
    }
    file.close();

    T binWidth = (highMax - highMin) / highBinCount;

    std::vector<T> predictedHighArray(leafMins.size());
    for (size_t i = 0; i < leafMins.size(); i++) {
        int lowBin = std::floor((leafMins[i] - lowMin) / (lowMax - lowMin) * lowBinCount);
        if (lowBin >= lowBinCount) lowBin = lowBinCount - 1;
        if (lowBin < 0) lowBin = 0;
        T highBin = lowHighBinMapping[lowBin].second;
        predictedHighArray[i] = highMin + highBin * binWidth + binWidth / 2.0;
    }
    return predictedHighArray;
}

// [新增] predictHighFromBinMapping 的纯计算版本：不再读文件，
// 接收已经读好的映射表数据作为参数，只做 bin 查找计算
template<typename T>
std::vector<T> predictHighFromBinMappingCompute(
    const std::vector<T>& leafMins,
    T lowMin, T lowMax, T highMin, T highMax,
    int lowBinCount, int highBinCount,
    const std::vector<std::pair<int, T>>& lowHighBinMapping)
{
    T binWidth = (highMax - highMin) / highBinCount;

    std::vector<T> predictedHighArray(leafMins.size());
    for (size_t i = 0; i < leafMins.size(); i++) {
        int lowBin = std::floor((leafMins[i] - lowMin) / (lowMax - lowMin) * lowBinCount);
        if (lowBin >= lowBinCount) lowBin = lowBinCount - 1;
        if (lowBin < 0) lowBin = 0;
        T highBin = lowHighBinMapping[lowBin].second;
        predictedHighArray[i] = highMin + highBin * binWidth + binWidth / 2.0;
    }
    return predictedHighArray;
}

// 用预测值数组 + type + outlier 重建 leafMaxs（每个元素独立预测，非链式）
template<typename T>
std::vector<T> reconstructFromPredicted(
    const std::vector<int>& typeVector,
    const std::vector<T>& predictedArray,
    const std::vector<T>& outLayerData,
    T error_bound)
{
    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T interval = 2 * error_bound;

    std::vector<T> result(typeVector.size());
    size_t outlierIdx = 0;
    for (size_t i = 0; i < typeVector.size(); i++) {
        int t = typeVector[i];
        if (t == 0) {
            result[i] = outLayerData[outlierIdx++];
        } else {
            int state = t - intvRadius;
            result[i] = predictedArray[i] + state * interval;
        }
    }
    return result;
}




// ============================================================
// uniform Octree 压缩
// ============================================================
template<typename T>
void compressOctreeUniform(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& blockCountOnEachDim,
    std::vector<T>* outListHigh)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : blockCountOnEachDim) totalLeaves *= d;
    size_t leafOffset = octree.size() - totalLeaves;

    std::vector<T> leafMins(totalLeaves), leafMaxs(totalLeaves);
    for (size_t i = 0; i < totalLeaves; i++) {
        leafMins[i] = octree[leafOffset + i].minVal;
        leafMaxs[i] = octree[leafOffset + i].maxVal;
    }

    auto compressStart_time = std::chrono::high_resolution_clock::now();

    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T checkRadius = (quantization_intervals - 1) * error_bound;
    const T interval = 2 * error_bound;
    const T recip_precision = 1 / error_bound;

    // 压缩 min
    std::vector<int> typeMinVector;
    std::vector<T> outLayerMinData;
    std::vector<T> reconstructedMins(totalLeaves);
    T predData = 0, reconstructedMin = 0;

    for (size_t i = 0; i < totalLeaves; i++) {
        T curData = leafMins[i];
        if (i == 0) {
            typeMinVector.push_back(0);
            outLayerMinData.push_back(curData);
            reconstructedMin = curData;
            predData = curData;
        } else {
            T predAbsErr = std::abs(curData - predData);
            if (predAbsErr < checkRadius) {
                int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                if (curData >= predData) {
                    typeMinVector.push_back(intvRadius + state);
                    reconstructedMin = predData + state * interval;
                } else {
                    typeMinVector.push_back(intvRadius - state);
                    reconstructedMin = predData - state * interval;
                }
            } else {
                typeMinVector.push_back(0);
                outLayerMinData.push_back(curData);
                reconstructedMin = curData;
            }
            predData = reconstructedMin;
        }
        reconstructedMins[i] = reconstructedMin;
    }

    // 构建 MinMaxPair
    std::vector<MinMaxPair<T>> minMaxPairs(totalLeaves);
    for (size_t i = 0; i < totalLeaves; i++) {
        minMaxPairs[i].reconstructedMin = reconstructedMins[i];
        minMaxPairs[i].originalMax = leafMaxs[i];
    }

    // 压缩 max：Bayes 预测
    int lowBinCount = 1000, highBinCount = 1000;
    std::string csvFileName = treeID + "-low_high_bin_mapping.csv";
    calculateConditionalProbability(minMaxPairs, lowBinCount, highBinCount, csvFileName);

    auto [typeMaxVector, outLayerMaxData, predictedHighArray] =
        computeHighBayesOptimizedAVLFixOrderNew(minMaxPairs, error_bound, treeID);

    auto quantiEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_quanti_time = quantiEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Octree UNIFORM] pre_quanti_time: " << pre_quanti_time.count()
              << " seconds" << std::endl;

    // 填充 outListHigh
    if (outListHigh) {
        outListHigh->resize(totalLeaves);
        size_t outlierIdx = 0;
        for (size_t i = 0; i < totalLeaves; i++) {
            int t = typeMaxVector[i];
            T reconstructedHigh;
            if (t == 0) {
                reconstructedHigh = outLayerMaxData[outlierIdx++];
            } else {
                T predictedHigh = predictedHighArray[i];
                int state = t - intvRadius;
                reconstructedHigh = predictedHigh + state * interval;
            }
            (*outListHigh)[i] = reconstructedHigh;
        }
    }

    // Huffman + Zstd 压缩
    std::string Minfilename = treeID + "-leafMin";
    std::string Maxfilename = treeID + "-leafMax";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);

    auto huffmanBuildEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanBuild_time = huffmanBuildEnd_time - quantiEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Octree UNIFORM] huffmanBuild_time: " << huffmanBuild_time.count()
              << " seconds" << std::endl;

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);
    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);

    auto compressEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> encode_time = compressEnd_time - huffmanBuildEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Octree UNIFORM] encode_time: " << encode_time.count()
              << " seconds" << std::endl;

    // 写文件
    auto writeBegin_time = std::chrono::high_resolution_clock::now();

    {
        std::string filename = treeID + "-OutMin.bin";
        std::ofstream outFile(filename, std::ios::binary);
        size_t size = outLayerMinData.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        outFile.write(reinterpret_cast<const char*>(outLayerMinData.data()), size * sizeof(T));
        outFile.close();
    }
    {
        std::string filename = treeID + "-OutMax.bin";
        std::ofstream outFile(filename, std::ios::binary);
        size_t size = outLayerMaxData.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        outFile.write(reinterpret_cast<const char*>(outLayerMaxData.data()), size * sizeof(T));
        outFile.close();
    }
    {
        std::string filenameMin = treeID + "-leafMin-compressedIndexData";
        std::ofstream outMin(filenameMin, std::ios::binary);
        outMin.write(reinterpret_cast<const char*>(compressedMin.data()), compressedMin.size());
        outMin.close();

        std::string filenameMax = treeID + "-leafMax-compressedIndexData";
        std::ofstream outMax(filenameMax, std::ios::binary);
        outMax.write(reinterpret_cast<const char*>(compressedMax.data()), compressedMax.size());
        outMax.close();
    }

    auto writeEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = writeEnd_time - writeBegin_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Octree UNIFORM] write_time: " << write_time.count()
              << " seconds" << std::endl;
}

// ============================================================
// stagger Octree 压缩
// ============================================================
template<typename T>
void compressOctreeStagger(
    const std::vector<OctreeNode<T>>& octree,
    T error_bound,
    const std::string& treeID,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<T>& listHigh)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : staggerBlockCountOnEachDim) totalLeaves *= d;
    size_t leafOffset = octree.size() - totalLeaves;

    std::vector<T> leafMins(totalLeaves), leafMaxs(totalLeaves);
    for (size_t i = 0; i < totalLeaves; i++) {
        leafMins[i] = octree[leafOffset + i].minVal;
        leafMaxs[i] = octree[leafOffset + i].maxVal;
    }

    auto compressStart_time = std::chrono::high_resolution_clock::now();

    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T checkRadius = (quantization_intervals - 1) * error_bound;
    const T interval = 2 * error_bound;
    const T recip_precision = 1 / error_bound;

    // 压缩 min
    std::vector<int> typeMinVector;
    std::vector<T> outLayerMinData;
    std::vector<T> reconstructedMins(totalLeaves);
    T predData = 0, reconstructedMin = 0;

    for (size_t i = 0; i < totalLeaves; i++) {
        T curData = leafMins[i];
        if (i == 0) {
            typeMinVector.push_back(0);
            outLayerMinData.push_back(curData);
            reconstructedMin = curData;
            predData = curData;
        } else {
            T predAbsErr = std::abs(curData - predData);
            if (predAbsErr < checkRadius) {
                int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                if (curData >= predData) {
                    typeMinVector.push_back(intvRadius + state);
                    reconstructedMin = predData + state * interval;
                } else {
                    typeMinVector.push_back(intvRadius - state);
                    reconstructedMin = predData - state * interval;
                }
            } else {
                typeMinVector.push_back(0);
                outLayerMinData.push_back(curData);
                reconstructedMin = curData;
            }
            predData = reconstructedMin;
        }
        reconstructedMins[i] = reconstructedMin;
    }

    // 压缩 max：用 uniform listHigh 的8个关联邻居平均值预测
    size_t nDim = staggerBlockCountOnEachDim.size();
    std::vector<int> typeMaxVector;
    std::vector<T> outLayerMaxData;

    for (size_t i = 0; i < totalLeaves; i++) {
        T curHigh = leafMaxs[i];

        std::vector<size_t> staggerCoord = positionToIndicesColMajor(i, staggerBlockCountOnEachDim);
        std::vector<size_t> relatedIds = getRelated8UniformBlockIds(
            staggerCoord, uniformBlockCountOnEachDim, nDim);

        T sum = 0;
        size_t count = 0;
        for (auto rid : relatedIds) {
            if (rid < listHigh.size() && !std::isnan(listHigh[rid])) {
                sum += listHigh[rid];
                count++;
            }
        }

        T predictedHigh = (count > 0) ? (sum / static_cast<T>(count)) : static_cast<T>(0);

        T predAbsErr = std::abs(curHigh - predictedHigh);
        if (predAbsErr < checkRadius) {
            int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
            if (curHigh >= predictedHigh) {
                typeMaxVector.push_back(intvRadius + state);
            } else {
                typeMaxVector.push_back(intvRadius - state);
            }
        } else {
            typeMaxVector.push_back(0);
            outLayerMaxData.push_back(curHigh);
        }
    }

    auto quantiEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_quanti_time = quantiEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Octree STAGGER] pre_quanti_time: " << pre_quanti_time.count()
              << " seconds" << std::endl;

    // Huffman + Zstd 压缩
    std::string Minfilename = treeID + "-leafMin";
    std::string Maxfilename = treeID + "-leafMax";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);

    auto huffmanBuildEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanBuild_time = huffmanBuildEnd_time - quantiEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Octree STAGGER] huffmanBuild_time: " << huffmanBuild_time.count()
              << " seconds" << std::endl;

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);
    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);

    auto compressEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> encode_time = compressEnd_time - huffmanBuildEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Octree STAGGER] encode_time: " << encode_time.count()
              << " seconds" << std::endl;

    // 写文件
    auto writeBegin_time = std::chrono::high_resolution_clock::now();

    {
        std::string filename = treeID + "-OutMin.bin";
        std::ofstream outFile(filename, std::ios::binary);
        size_t size = outLayerMinData.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        outFile.write(reinterpret_cast<const char*>(outLayerMinData.data()), size * sizeof(T));
        outFile.close();
    }
    {
        std::string filename = treeID + "-OutMax.bin";
        std::ofstream outFile(filename, std::ios::binary);
        size_t size = outLayerMaxData.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        outFile.write(reinterpret_cast<const char*>(outLayerMaxData.data()), size * sizeof(T));
        outFile.close();
    }
    {
        std::string filenameMin = treeID + "-leafMin-compressedIndexData";
        std::ofstream outMin(filenameMin, std::ios::binary);
        outMin.write(reinterpret_cast<const char*>(compressedMin.data()), compressedMin.size());
        outMin.close();

        std::string filenameMax = treeID + "-leafMax-compressedIndexData";
        std::ofstream outMax(filenameMax, std::ios::binary);
        outMax.write(reinterpret_cast<const char*>(compressedMax.data()), compressedMax.size());
        outMax.close();
    }

    auto writeEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = writeEnd_time - writeBegin_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time][Octree STAGGER] write_time: " << write_time.count()
              << " seconds" << std::endl;
}

// ============================================================
// [修改] uniform Octree 解压 —— 加入分段计时
//   [Time3]     读文件（I/O）
//   [Time4abc]  Huffman解码 + 反量化leafMins + Bayes预测（CPU）
//   [Time4d]    反量化leafMaxs（CPU）
//   [Time4e]    建树（CPU）
//   [Time4]     Time4abc + Time4d + Time4e 的总和
// ============================================================
template<typename T>
std::vector<OctreeNode<T>> decompressOctreeUniform(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& blockCountOnEachDim)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : blockCountOnEachDim) totalLeaves *= d;

    // ========== [Time3] 读文件（I/O） ==========
    auto beginRead = std::chrono::high_resolution_clock::now();

    // 反 Huffman + 反 zstd 所需的原始文件
    std::vector<unsigned char> huffmanOutMin = loadHuffmanTree(treeID + "-leafMin-Huffman");
    std::vector<unsigned char> huffmanOutMax = loadHuffmanTree(treeID + "-leafMax-Huffman");
    std::vector<unsigned char> compressedMin = readCompressedData(treeID + "-leafMin-compressedIndexData");
    std::vector<unsigned char> compressedMax = readCompressedData(treeID + "-leafMax-compressedIndexData");

    // 读 outlier
    std::vector<T> outLayerMinData = readOctreeBinaryVector<T>(treeID + "-OutMin.bin");
    std::vector<T> outLayerMaxData = readOctreeBinaryVector<T>(treeID + "-OutMax.bin");

    // 原本藏在 predictHighFromBinMapping 内部的 csv 读取，挪到这里统一算作"读文件"
    T lowMin, lowMax, highMin, highMax;
    int lowBinCount, highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMapping;
    std::tie(lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, lowHighBinMapping) =
        loadBinMappingForPredict<T>(treeID);

    auto endRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readTime = endRead - beginRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time3][Octree UNIFORM] readIndexTime (" << treeID << "): "
              << readTime.count() << " seconds" << std::endl;

    // ========== [Time4abc] Huffman解码 + 反量化leafMins + Bayes预测 ==========
    auto beginDecode = std::chrono::high_resolution_clock::now();

    std::vector<int> typeMinVector = singleHuffmanDecodeZstdAVL(compressedMin, totalLeaves, huffmanOutMin.data());
    std::vector<int> typeMaxVector = singleHuffmanDecodeZstdAVL(compressedMax, totalLeaves, huffmanOutMax.data());

    // 重建 leafMins（链式反量化）
    std::vector<T> leafMins = sequentialDequantize(typeMinVector, outLayerMinData, error_bound);

    // Bayes 预测（纯计算，用已读好的映射表）
    std::vector<T> predictedHighArray = predictHighFromBinMappingCompute<T>(
        leafMins, lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, lowHighBinMapping);

    auto endDecode = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decodeTime = endDecode - beginDecode;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4abc][Octree UNIFORM] huffmanDecode+dequantMin+bayesPredict (" << treeID << "): "
              << decodeTime.count() << " seconds" << std::endl;

    // ========== [Time4d] 反量化leafMaxs ==========
    auto beginMax = std::chrono::high_resolution_clock::now();

    std::vector<T> leafMaxs = reconstructFromPredicted(typeMaxVector, predictedHighArray, outLayerMaxData, error_bound);

    auto endMax = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> maxTime = endMax - beginMax;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4d][Octree UNIFORM] dequantLeafMaxs (" << treeID << "): "
              << maxTime.count() << " seconds" << std::endl;

    // ========== [Time4e] 建树 ==========
    auto beginBuild = std::chrono::high_resolution_clock::now();

    // 复用现成的 buildOctree，自动重建所有内部节点的 min/max
    std::vector<OctreeNode<T>> tree = buildOctree(leafMins, leafMaxs, blockCountOnEachDim);

    auto endBuild = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> buildTime = endBuild - beginBuild;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4e][Octree UNIFORM] buildOctree (" << treeID << "): "
              << buildTime.count() << " seconds" << std::endl;

    // ========== [Time4] 总和 ==========
    std::chrono::duration<double> totalReconstructTime = endBuild - endRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4][Octree UNIFORM] decodeAndReconstructTree total (" << treeID << "): "
              << totalReconstructTime.count() << " seconds" << std::endl;

    return tree;
}


// ============================================================
// [修改] stagger Octree 解压 —— 分段计时 + [Time4c] 邻居预测优化
//   [Time3]  读文件（I/O）
//   [Time4a] Huffman解码（CPU）
//   [Time4b] 反量化leafMins（CPU）
//   [Time4c] 邻居预测（CPU，已优化：去掉热点循环内的堆分配）
//   [Time4d] 反量化leafMaxs（CPU）
//   [Time4e] 建树（CPU）
//   [Time4]  Time4a + Time4b + Time4c + Time4d + Time4e 的总和
// ============================================================
template<typename T>
std::vector<OctreeNode<T>> decompressOctreeStagger(
    const std::string& treeID,
    T error_bound,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<T>& listHigh)
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t totalLeaves = 1;
    for (size_t d : staggerBlockCountOnEachDim) totalLeaves *= d;
    size_t nDim = staggerBlockCountOnEachDim.size();

    // ========== [Time3] 读文件（I/O） ==========
    auto beginRead = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> huffmanOutMin = loadHuffmanTree(treeID + "-leafMin-Huffman");
    std::vector<unsigned char> huffmanOutMax = loadHuffmanTree(treeID + "-leafMax-Huffman");
    std::vector<unsigned char> compressedMin = readCompressedData(treeID + "-leafMin-compressedIndexData");
    std::vector<unsigned char> compressedMax = readCompressedData(treeID + "-leafMax-compressedIndexData");

    std::vector<T> outLayerMinData = readOctreeBinaryVector<T>(treeID + "-OutMin.bin");
    std::vector<T> outLayerMaxData = readOctreeBinaryVector<T>(treeID + "-OutMax.bin");

    auto endRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readTime = endRead - beginRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time3][Octree STAGGER] readIndexTime (" << treeID << "): "
              << readTime.count() << " seconds" << std::endl;

    // ========== [Time4a] Huffman解码 ==========
    auto beginHuffman = std::chrono::high_resolution_clock::now();

    std::vector<int> typeMinVector = singleHuffmanDecodeZstdAVL(compressedMin, totalLeaves, huffmanOutMin.data());
    std::vector<int> typeMaxVector = singleHuffmanDecodeZstdAVL(compressedMax, totalLeaves, huffmanOutMax.data());

    auto endHuffman = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanTime = endHuffman - beginHuffman;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4a][Octree STAGGER] huffmanDecode (" << treeID << "): "
              << huffmanTime.count() << " seconds" << std::endl;

    // ========== [Time4b] 反量化leafMins ==========
    auto beginDequantMin = std::chrono::high_resolution_clock::now();

    std::vector<T> leafMins = sequentialDequantize(typeMinVector, outLayerMinData, error_bound);

    auto endDequantMin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dequantMinTime = endDequantMin - beginDequantMin;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4b][Octree STAGGER] dequantMin (" << treeID << "): "
              << dequantMinTime.count() << " seconds" << std::endl;

    // ========== [Time4c] 邻居预测（已优化：无堆分配版本） ==========
    auto beginNeighborPredict = std::chrono::high_resolution_clock::now();

    std::vector<T> predictedHighArray(totalLeaves);

    if (nDim == 3) {
        // 快速路径：栈上固定大小数组，循环体内零堆分配
        std::array<size_t, 3> staggerCoord;
        std::array<size_t, 8> relatedIdsBuf;
        size_t relatedCount;

        for (size_t i = 0; i < totalLeaves; i++) {
            positionToIndicesColMajor3D(i, staggerBlockCountOnEachDim, staggerCoord);
            getRelated8UniformBlockIds3D(staggerCoord, uniformBlockCountOnEachDim, relatedIdsBuf, relatedCount);

            T sum = 0;
            size_t count = 0;
            for (size_t k = 0; k < relatedCount; k++) {
                size_t rid = relatedIdsBuf[k];
                if (rid < listHigh.size() && !std::isnan(listHigh[rid])) {
                    sum += listHigh[rid];
                    count++;
                }
            }
            predictedHighArray[i] = (count > 0) ? (sum / static_cast<T>(count)) : static_cast<T>(0);
        }
    } else {
        // 兜底路径：非3维情况，保留原有通用实现，避免破坏兼容性
        for (size_t i = 0; i < totalLeaves; i++) {
            std::vector<size_t> staggerCoord = positionToIndicesColMajor(i, staggerBlockCountOnEachDim);
            std::vector<size_t> relatedIds = getRelated8UniformBlockIds(staggerCoord, uniformBlockCountOnEachDim, nDim);

            T sum = 0;
            size_t count = 0;
            for (auto rid : relatedIds) {
                if (rid < listHigh.size() && !std::isnan(listHigh[rid])) {
                    sum += listHigh[rid];
                    count++;
                }
            }
            predictedHighArray[i] = (count > 0) ? (sum / static_cast<T>(count)) : static_cast<T>(0);
        }
    }

    auto endNeighborPredict = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> neighborPredictTime = endNeighborPredict - beginNeighborPredict;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4c][Octree STAGGER] neighborPredict (" << treeID << "): "
              << neighborPredictTime.count() << " seconds" << std::endl;

    // ========== [Time4d] 反量化leafMaxs ==========
    auto beginMax = std::chrono::high_resolution_clock::now();

    std::vector<T> leafMaxs = reconstructFromPredicted(typeMaxVector, predictedHighArray, outLayerMaxData, error_bound);

    auto endMax = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> maxTime = endMax - beginMax;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4d][Octree STAGGER] dequantLeafMaxs (" << treeID << "): "
              << maxTime.count() << " seconds" << std::endl;

    // ========== [Time4e] 建树 ==========
    auto beginBuild = std::chrono::high_resolution_clock::now();

    std::vector<OctreeNode<T>> tree = buildOctree(leafMins, leafMaxs, staggerBlockCountOnEachDim);

    auto endBuild = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> buildTime = endBuild - beginBuild;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4e][Octree STAGGER] buildOctree (" << treeID << "): "
              << buildTime.count() << " seconds" << std::endl;

    // ========== [Time4] 总和 ==========
    std::chrono::duration<double> totalReconstructTime = endBuild - endRead;
    std::cerr << "Current MPI rank: " << rank
              << "[Time4][Octree STAGGER] decodeAndReconstructTree total (" << treeID << "): "
              << totalReconstructTime.count() << " seconds" << std::endl;

    return tree;
}








// ============================================================
// 显式实例化
// ============================================================
template void compressOctreeUniform<float>(
    const std::vector<OctreeNode<float>>&, float, const std::string&,
    const std::vector<size_t>&, std::vector<float>*);

template void compressOctreeUniform<double>(
    const std::vector<OctreeNode<double>>&, double, const std::string&,
    const std::vector<size_t>&, std::vector<double>*);

template void compressOctreeStagger<float>(
    const std::vector<OctreeNode<float>>&, float, const std::string&,
    const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<float>&);

template void compressOctreeStagger<double>(
    const std::vector<OctreeNode<double>>&, double, const std::string&,
    const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<double>&);


// ============================================================
// 显式实例化：解压函数
// ============================================================
template std::vector<OctreeNode<float>> decompressOctreeUniform<float>(
    const std::string&, float, const std::vector<size_t>&);

template std::vector<OctreeNode<double>> decompressOctreeUniform<double>(
    const std::string&, double, const std::vector<size_t>&);

template std::vector<OctreeNode<float>> decompressOctreeStagger<float>(
    const std::string&, float, const std::vector<size_t>&,
    const std::vector<size_t>&, const std::vector<float>&);

template std::vector<OctreeNode<double>> decompressOctreeStagger<double>(
    const std::string&, double, const std::vector<size_t>&,
    const std::vector<size_t>&, const std::vector<double>&);
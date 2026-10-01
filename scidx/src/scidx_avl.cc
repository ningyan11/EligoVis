#include "scidx_avl.h"
#include <adios2.h>
#include <cmath>
#include <cstdlib>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <scidx_defines.h>
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <zstd.h>
#include <scidx_BytesToolkit.h>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <mpi.h> 
#include <SZ3/api/sz.hpp> 
#include <zfp.h>



using namespace scidx;
std::string formatSize(size_t bytes);

template <typename T>
struct MinMaxPair {
    T reconstructedMin;  // 可解压 min
    T originalMax;       // 原始 max
};


void saveCompressedDataToFilesTest(const std::string& treeID,
    const std::vector<unsigned char>& compressedMin,
    const std::vector<unsigned char>& compressedMax, const std::vector<unsigned char>& compressedSkippedMin, const std::vector<unsigned char>& compressedSkippedMax );

    template <typename T>
    void writeOutLayerDataToFileOptimizedTest(const std::string& treeID, 
                                       const std::vector<T>& outLayerMinData, 
                                       const std::vector<T>& outLayerMaxData, const std::vector<T>& outLayerSkippedMinData, const std::vector<T>& outLayerSkipedMaxData );

template<typename T>
T getHighBinMidValue(T highBin, T highMin, T highMax, int highBinCount);

template<typename T>
int findLowBin(T lowValue, T lowMin, T lowMax, int lowBinCount);


template<typename T>
std::tuple<T, T, T, T, int, int, std::vector<std::pair<int, T>>> 
loadBinMappingAndStats(const std::string& csvFile);

template<typename T>
std::tuple<ScidxAVLNode<T>*, std::vector<T>, std::vector<T>, std::vector<size_t>>  decompressOptimizedAVL(const std::string& treeID ,  T error_bound);

template<typename T>
ScidxAVLNode<T>* decompressOptimizedTestAVL(const std::string& treeID ,  T error_bound);


template<typename T>
void compressNaiveAVL(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
    T error_bound, 
    std::string treeID
);

template<typename T>
void computeOptimizedAVL(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
    T error_bound, 
    std::string treeID, std::vector<SkippedNode<T>> skippedAllIntervals
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
void computeOptimizedSZ3(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
    T error_bound, 
    std::string treeID, std::vector<SkippedNode<T>> skippedAllIntervals
);


template<typename T>
void computeOptimizedTestAVL(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
    T error_bound, 
    std::string treeID
);

template <typename T>
void writeOutLayerDataToFileOptimized(const std::string& treeID, 
                                   const std::vector<T>& outLayerMinData, 
                                   const std::vector<T>& outLayerMaxData, 
                                   const std::vector<T>& outLayerMaxHighData) ;

template<typename T>
auto computeHighBayesOptimizedAVL(const std::vector<std::pair<T, T>>& flattenedNodeLowHigh, T error_bound, std::string treeID ) -> std::pair<std::vector<int>, std::vector<T>> ;


template<typename T>
std::vector<int> computeArrayTypeBayesAVL(size_t k, const std::vector<std::pair<T, T>>& flattenedNodeLowHigh, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName, const std::string& csvFile);

template<typename T>
std::pair<std::vector<int>, std::vector<T>> computeTypeForFlattenedArray(
    const std::vector<T>& data, T error_bound);



template<typename T>
std::vector<std::pair<T, T>> flattenedTreeNodeLowHigh(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree);


template<typename T>
std::vector<int> computeArrayTypeBayesAVL(size_t k, const std::vector<std::pair<T, T>>& flattenedNodeLowHigh, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName, const std::string& csvFile);



std::pair<std::vector<uint8_t>, unsigned char> idConvertToBytes(const std::vector<size_t>& currentIds);

template<typename T>
void compressTree(size_t k, std::vector<int> currentTypesLow, std::vector<int> currentTypesHigh, std::vector<size_t> currentIds, std::vector<int> currentTypesMaxHigh, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t step, scidx::HuffmanTree* fullHuffmanTreeLow, scidx::HuffmanTree* fullHuffmanTreeHigh, scidx::HuffmanTree* fullHuffmanTreeMaxHigh);

template<typename T>
ScidxAVLNode<T>* decompressTreeAVL(size_t k, adios2::Engine& engine, adios2::IO& io, size_t step, std::vector<int>& firstVector, T error_bound, std::vector<unsigned char> huffmanOut);

template<typename T>
std::vector<T> flattenedTreeNodeMax(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree);

template<typename T>
std::vector<T> flattenedTreeNodeMaxHigh(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree);


template<typename T>
std::vector<T> flattenedTreeNodeMin(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree);

template<typename T>
std::vector<T> getLeafNodeMaxHigh(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree);

template<typename T>
std::vector<int> preQuantiSingleTreeMin(size_t k, std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName);

template<typename T>
std::vector<int> computeArrayTypeAVL(size_t k, const std::vector<T> &flattenedTreeNodeValue, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName);

std::vector<unsigned char> singleHuffmanEncodeZstdAVL(std::vector<int> type, HuffmanTree *huffmanTree);

std::vector<int> singleHuffmanDecodeZstdAVL(std::vector<unsigned char> compressedByZstdData, size_t typeSize, unsigned char *s);

std::vector<unsigned char> compressWithZstdAVL(const std::vector<unsigned char> &data, size_t dataLength);

std::vector<unsigned char> decompressWithZstdAVL(const std::vector<unsigned char> &compressedData, uint64_t targetOriSize);

std::vector<int> compress_data(float *oriData, size_t dataLength, float error_bound);

void decode_withSubTreeAVL(unsigned char *s, unsigned char *encode, size_t targetLength, int *out);

template<typename T>
std::vector<T> flattenTreeMaxHigh(const std::vector<std::vector<ScidxAVLNode<T>*>> &tree);

template<typename T>
std::vector<T> decodeArrayType(const std::vector<int>& type, T error_bound, std::vector<T> outlayerData);

template<typename T>
T dequantizeValue(int quantizedValue, T error_bound, int intvRadius, T interval, T& baseValue);

template<typename T>
ScidxAVLNode<T>* reconstructAndDequantizeTree(const std::vector<int>& type, const std::vector<int>& structureVec, T error_bound, std::vector<T> outlayerLow);

//template<typename T>
//void reconstructMax(ScidxAVLNode<T>* root, const std::vector<T>& decodeFlattenedMax, std::vector<unsigned int> idOut);


template <typename T>
void reconstructMax(ScidxAVLNode<T>* root, const std::vector<T>& decodeFlattenedMax, const std::vector<size_t>& idOut);



template<typename T>
void reconstractLeafMaxHigh(ScidxAVLNode<T>* root, const std::vector<T>& decodeFlattenedLeafMaxHigh);

template<typename T>
void updateMaxHighOfTree(ScidxAVLNode<T>* node);

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

void writeNodeBitsIdToFile(const std::string& filename, const std::vector<size_t>& nodeId);
void saveCompressedDataToFiles(const std::string& treeID,
                               const std::vector<uint8_t>& compressedMin,
                               const std::vector<uint8_t>& compressedMax,
                               const std::vector<uint8_t>& compressedMaxHigh);





std::vector<scidx::HuffmanTree*> fullHuffmanAVL(size_t k, const std::vector<std::vector<int>>& allTreeTypesLow,
                                          const std::vector<std::vector<int>>& allTreeTypesHigh,
                                          const std::vector<std::vector<int>>& allTreeTypesMaxHigh,
                                          adios2::Engine& engine, adios2::IO& io);

HuffmanTree* processHuffmanTreeAVL(size_t k, const std::vector<std::vector<int>>& allTreeTypes, adios2::Engine& engine, adios2::IO& io, const std::string& variableName);

std::vector<scidx::HuffmanTree*> fullHuffmanAVL(size_t k, const std::vector<std::vector<int>>& allTreeTypesLow,
                                          const std::vector<std::vector<int>>& allTreeTypesHigh,
                                          const std::vector<std::vector<int>>& allTreeTypesMaxHigh,
                                          adios2::Engine& engine, adios2::IO& io) {

    std::vector<HuffmanTree*> huffmanTrees;

    // 处理并写入不同的 Huffman 树
    HuffmanTree* huffmanTreeLow = processHuffmanTreeAVL(k, allTreeTypesLow, engine, io, "Hu_l");
    huffmanTrees.push_back(huffmanTreeLow);

    HuffmanTree* huffmanTreeHigh = processHuffmanTreeAVL(k, allTreeTypesHigh, engine, io, "Hu_h");
    huffmanTrees.push_back(huffmanTreeHigh);

    HuffmanTree* huffmanTreeMaxHigh = processHuffmanTreeAVL(k, allTreeTypesMaxHigh, engine, io, "Hu_mh");
    huffmanTrees.push_back(huffmanTreeMaxHigh);

    return huffmanTrees;
}





HuffmanTree* processHuffmanTreeAVL(size_t k, const std::vector<std::vector<int>>& allTreeTypes, adios2::Engine& engine, adios2::IO& io, const std::string& variableName) {
    std::string newVariableName = std::to_string(k) +"_" + variableName;
    std::vector<int> allTypes;
    for (const auto& type : allTreeTypes) {
        allTypes.insert(allTypes.end(), type.begin(), type.end());
    }

    int stateNum = 2 * 16384;
    HuffmanTree *huffmanTreeFull = createHuffmanTree(stateNum);

    unsigned char *huffmanOut = nullptr;
    size_t huffmanOutSize = 0;
    init_and_serialize_Huffmantree(huffmanTreeFull, allTypes.data(), allTypes.size(), &huffmanOut, &huffmanOutSize);

    // Ensure huffmanOut and huffmanOutSize are valid
    if (huffmanOut == nullptr || huffmanOutSize == 0) {
        std::cerr << "Error: huffmanOut is null or huffmanOutSize is zero for " << variableName << std::endl;
        free(huffmanOut);
        return nullptr;
    }

    // Define the variable for Huffman tree output
    adios2::Variable<unsigned char> fullHuffmanTreeVar = io.DefineVariable<unsigned char>(
        newVariableName, {huffmanOutSize}, {0}, {huffmanOutSize}, adios2::ConstantDims);

    // Write the Huffman tree output to the file
    engine.Put(fullHuffmanTreeVar, huffmanOut, adios2::Mode::Sync);

    // Clean up the serialized output buffer
    free(huffmanOut);

    return huffmanTreeFull;
}

HuffmanTree* processHuffmanTreeNaiveAVL(std::vector<int> treeType, std::string& fileName) {

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int stateNum = 2 * 16384;
    HuffmanTree *huffmanTreeFull = createHuffmanTree(stateNum);

    unsigned char *huffmanOut = nullptr;
    size_t huffmanOutSize = 0;
    init_and_serialize_Huffmantree(huffmanTreeFull, treeType.data(), treeType.size(), &huffmanOut, &huffmanOutSize);

    // Ensure huffmanOut and huffmanOutSize are valid
    if (huffmanOut == nullptr || huffmanOutSize == 0) {
        std::cerr << "Error: huffmanOut is null or huffmanOutSize is zero for "<< std::endl;
        free(huffmanOut);
        return nullptr;
    }

    std::string HuffmanfileName = fileName + "-Huffman";

    auto start_time = std::chrono::high_resolution_clock::now();

    // 将 Huffman 写入文件
    std::ofstream outFile(HuffmanfileName, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file " << HuffmanfileName << " for writing." << std::endl;
        free(huffmanOut);
        return nullptr;
    }
    outFile.write(reinterpret_cast<char*>(huffmanOut), huffmanOutSize);
    outFile.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = end_time - start_time;


    std::cout << "Current MPI rank: " << rank << "[writing time b] write huffman tree: " << write_time.count() << " seconds" << std::endl;


    free(huffmanOut);

    return huffmanTreeFull;
}

void writeTypeSizesToFile(const std::string& treeID, const std::vector<size_t>& typeSizes) {
    std::string filename = treeID + "-typeSize";
    std::ofstream outFile(filename, std::ios::binary);

    if (!outFile) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // 先写入数据大小（方便解压时检查）
    size_t size = typeSizes.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

    // 写入 `typeSizes` 数据
    outFile.write(reinterpret_cast<const char*>(typeSizes.data()), size * sizeof(size_t));

    outFile.close();
    //std::cout << "Type sizes saved to " << filename << std::endl;
}

// 计算每个 low bin 和 high bin 的索引，返回 std::pair
std::pair<int, int> getBinIndicesNew(double low, double high, double lowMin, double lowMax, double highMin, double highMax, int lowBinCount, int highBinCount) {
    int lowBin = std::floor((low - lowMin) / (lowMax - lowMin) * lowBinCount);
    int highBin = std::floor((high - highMin) / (highMax - highMin) * highBinCount);

    // 防止索引越界
    if (lowBin >= lowBinCount) lowBin = lowBinCount - 1;
    if (lowBin < 0) lowBin = 0;

    if (highBin >= highBinCount) highBin = highBinCount - 1;
    if (highBin < 0) highBin = 0;

    return std::make_pair(lowBin, highBin);  // 返回 std::pair
}

// 导出 countMatrix 到 CSV 文件
void exportCountMatrixToCSVNew(const std::vector<std::vector<int>>& countMatrix, const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        // 遍历 countMatrix，并将其写入 CSV 文件
        for (const auto& row : countMatrix) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) {
                    file << ",";  // 添加逗号分隔
                }
            }
            file << "\n";  // 每一行之后换行
        }
        file.close();
        //std::cout << "Count matrix exported to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}


template <typename T>
void calculateConditionalProbability(
    const std::vector<MinMaxPair<T>>& minMaxPairs,
    int lowBinCount,
    int highBinCount,
    const std::string& csvFile)
{
    // 1. 计算 min 和 max 的最小/最大值
    T lowMin = minMaxPairs[0].reconstructedMin, lowMax = minMaxPairs[0].reconstructedMin;
    T highMin = minMaxPairs[0].originalMax, highMax = minMaxPairs[0].originalMax;

    for (const auto& pair : minMaxPairs) {
        if (pair.reconstructedMin < lowMin) lowMin = pair.reconstructedMin;
        if (pair.reconstructedMin > lowMax) lowMax = pair.reconstructedMin;
        if (pair.originalMax < highMin) highMin = pair.originalMax;
        if (pair.originalMax > highMax) highMax = pair.originalMax;
    }

    // 增加微小 epsilon 防止边界值越界
    T epsilon = static_cast<T>(1e-6);
    lowMax += epsilon;
    highMax += epsilon;

    // 2. 初始化计数矩阵
    std::vector<std::vector<int>> countMatrix(lowBinCount, std::vector<int>(highBinCount, 0));
    std::vector<int> lowBinTotalCount(lowBinCount, 0);

    // 3. 遍历每个 pair，计算其 bin
    for (const auto& pair : minMaxPairs) {
        std::pair<int, int> binIndices = getBinIndicesNew(
            pair.reconstructedMin, pair.originalMax,
            lowMin, lowMax, highMin, highMax,
            lowBinCount, highBinCount);

        int lowBin = binIndices.first;
        int highBin = binIndices.second;

        countMatrix[lowBin][highBin]++;
        lowBinTotalCount[lowBin]++;
    }

    exportCountMatrixToCSVNew(countMatrix, "count_matrix.csv");

    // 4. 计算条件概率和期望值
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

    auto start_time = std::chrono::high_resolution_clock::now();

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // 5. 写入 CSV
    std::ofstream file(csvFile);
    if (file.is_open()) {
        file << "lowMin,lowMax,highMin,highMax,lowBinCount,highBinCount\n";
        file << lowMin << "," << lowMax << "," << highMin << "," << highMax << "," << lowBinCount << "," << highBinCount << "\n";

        file << "Low Bin,Expected High Bin\n";
        for (const auto& bin : nonZeroBins) {
            file << bin.first << "," << bin.second << "\n";
        }

        file.close();
        //std::cout << "CSV file saved as " << csvFile << std::endl;
    } else {
        std::cerr << "Unable to open file: " << csvFile << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = end_time - start_time;
    std::cout  << "Current MPI rank: " << rank<< "[writing time a] write bayes: " << write_time.count() << " seconds" << std::endl;

}





/*template<class T>
void SZ3_LORENZO_compress_1D(
    double *input, size_t numElements, double absErrorBound, const std::string& filename1, const std::string& filename2,const std::string& filename3
) {
   
    // 配置器
    SZ3::Config conf(numElements);
    conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;  // 使用Lorenzo算法
    conf.lorenzo = true;                   // 启用Lorenzo预测器
    conf.regression = false;               // 不启用回归
    conf.errorBoundMode = SZ3::EB_ABS;     // 使用绝对误差模式
    conf.absErrorBound = absErrorBound;
    conf.quantbinCnt = 1024;

    // 创建预测器、量化器和分解器
    auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
    auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
    auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);

    // 压缩数据
    std::vector<int> quant_inds = decompose.compress(conf, input);

    // 保存元数据
    size_t bufferSize = 10 * sizeof(double) * numElements;  // 缓冲区大小足够容纳2倍原始数据
    auto *buffer = new SZ3::uchar[bufferSize];
    auto *buffer_zstd = new SZ3::uchar[bufferSize];
    SZ3::uchar *buffer_pos = buffer;

    // 保存分解器元数据
    decompose.save(buffer_pos);
    size_t metadataSize = buffer_pos - buffer;
    SZ3::writefile(filename1.c_str(), buffer, metadataSize);  // 保存元数据到文件

    // 构建 Huffman 树
    SZ3::HuffmanEncoder<int> huffman;
    huffman.preprocess_encode(quant_inds, 0);

    // 保存 Huffman 树元数据
    buffer_pos = buffer;
    huffman.save(buffer_pos);
    SZ3::writefile(filename2.c_str(), buffer, buffer_pos - buffer);

    // 使用 Huffman 编码
    buffer_pos = buffer;
    huffman.encode(quant_inds.data(), numElements, buffer_pos);

    // 使用 Zstd 压缩编码后的数据
    SZ3::Lossless_zstd zstd;
    size_t csize = zstd.compress(buffer, buffer_pos - buffer, buffer_zstd, bufferSize);

    // 保存最终压缩结果到文件
    SZ3::writefile(filename3.c_str(), buffer_zstd, csize);

    delete[] buffer;
    delete[] buffer_zstd;

}*/

template<class T>
void SZ3_LORENZO_compress_1D(
    T *input, size_t numElements, double absErrorBound, 
    const std::string& filename1, const std::string& filename2, const std::string& filename3
) {
    SZ3::Config conf(numElements);
    conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
    conf.lorenzo = true;
    conf.regression = false;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = absErrorBound;
    conf.quantbinCnt = 1024;

    auto predictor = SZ3::LorenzoPredictor<T, 1, 1>(conf.absErrorBound);
    auto quantizer = SZ3::LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2);
    auto decompose = SZ3::make_decomposition_lorenzo_regression<T, 1>(conf, quantizer);

    std::vector<int> quant_inds = decompose.compress(conf, input);

    size_t bufferSize = 10 * sizeof(T) * numElements;
    auto *buffer = new SZ3::uchar[bufferSize];
    auto *buffer_zstd = new SZ3::uchar[bufferSize];
    SZ3::uchar *buffer_pos = buffer;

    decompose.save(buffer_pos);
    size_t metadataSize = buffer_pos - buffer;
    SZ3::writefile(filename1.c_str(), buffer, metadataSize);

    SZ3::HuffmanEncoder<int> huffman;
    huffman.preprocess_encode(quant_inds, 0);

    buffer_pos = buffer;
    huffman.save(buffer_pos);
    SZ3::writefile(filename2.c_str(), buffer, buffer_pos - buffer);

    buffer_pos = buffer;
    huffman.encode(quant_inds.data(), numElements, buffer_pos);

    SZ3::Lossless_zstd zstd;
    size_t csize = zstd.compress(buffer, buffer_pos - buffer, buffer_zstd, bufferSize);
    SZ3::writefile(filename3.c_str(), buffer_zstd, csize);

    delete[] buffer;
    delete[] buffer_zstd;
}






// 以二进制方式写入 nodeId 到文件
void writeNodeIdToFile(const std::string& filename, const std::vector<size_t>& nodeId) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // 写入数据
    outFile.write(reinterpret_cast<const char*>(nodeId.data()), nodeId.size() * sizeof(size_t));

    outFile.close();
}



template<typename T>
void computeOptimizedSZ3(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
    T error_bound, 
    std::string treeID, std::vector<SkippedNode<T>> skippedAllIntervals
) {

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<size_t> skippedIds;
    std::vector<T> skippedLows;
    std::vector<T> skippedHighs;

    //提取skipped数据
    for (const auto& node : skippedAllIntervals) {
        skippedIds.push_back(node.id);           // 提取 id
        skippedLows.push_back(node.interval.low); // 提取 low 值
        skippedHighs.push_back(node.interval.high); // 提取 high 值
    }

     
    std::string skippedIdfilename = treeID + "-SkippedNodeID";
    std::string Idfilename = treeID + "-NodeID";  

    std::string Minfilename = treeID + "-Min";
    std::string MinMetafilename = treeID + "-Min-meta";
    std::string MinHuffmanfilename = treeID + "-Min-huffman";

    std::string Maxfilename = treeID + "-Max";
    std::string MaxMetafilename = treeID + "-Max-meta";
    std::string MaxHuffmanfilename = treeID + "-Max-huffman";


    std::string MaxHighfilename = treeID + "-MaxHigh";
    std::string MaxHighMetafilename = treeID + "-MaxHigh-meta";
    std::string MaxHighHuffmanfilename = treeID + "-MaxHigh-huffman";


    std::string SkippedMinfilename = treeID + "-SkippedMin";
    std::string skipMinMetafilename = treeID + "-SkippedMin-meta";
    std::string skipMinHuffmanfilename = treeID + "-SkippedMin-huffman";


    std::string SkippedMaxfilename = treeID + "-SkippedMax";
    std::string skipMaxMetafilename = treeID + "-SkippedMax-meta";
    std::string skipMaxHuffmanfilename = treeID + "-SkippedMax-huffman";
    


    auto compressStart_time = std::chrono::high_resolution_clock::now();


    std::vector<size_t> treeNodeIdArray = flattenedTreeNodeId(singleTree);
    std::vector<T> treeNodeMinArray = flattenedTreeNodeMin(singleTree);
    std::vector<T> treeNodeMaxArray = flattenedTreeNodeMax(singleTree);
    std::vector<T> treeNodeMaxHighArray = flattenedTreeNodeMaxHigh(singleTree);


    SZ3_LORENZO_compress_1D<T>(skippedLows.data(),skippedLows.size(),error_bound,skipMinMetafilename, skipMinHuffmanfilename,  SkippedMinfilename );
    SZ3_LORENZO_compress_1D<T>(skippedHighs.data(),skippedHighs.size(),error_bound,skipMaxMetafilename, skipMaxHuffmanfilename,  SkippedMaxfilename );

    SZ3_LORENZO_compress_1D<T>(treeNodeMinArray.data(),treeNodeMinArray.size(),error_bound,MinMetafilename, MinHuffmanfilename,  Minfilename );
    SZ3_LORENZO_compress_1D<T>(treeNodeMaxArray.data(),treeNodeMaxArray.size(),error_bound,MaxMetafilename, MaxHuffmanfilename,  Maxfilename );
    SZ3_LORENZO_compress_1D<T>(treeNodeMaxHighArray.data(),treeNodeMaxHighArray.size(),error_bound,MaxHighMetafilename, MaxHighHuffmanfilename,  MaxHighfilename );


    writeNodeIdToFile(skippedIdfilename, skippedIds);
    writeNodeIdToFile(Idfilename, treeNodeIdArray);



}

template<typename T>
std::vector<size_t> flattenedTreeNodeId(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree) {
    std::vector<size_t> flattenedNodeId;

    for (const auto &row : singleSubTree) {
        for (const auto &node : row) {
            flattenedNodeId.push_back(node->id);
        }
    }

    return flattenedNodeId;
}


/*void ZFP_compress_1D(double* data, size_t numElements, double tolerance, const std::string& outputFile) {
    // 设置维度信息
    zfp_type type = zfp_type_double;
    zfp_field* field = zfp_field_1d(data, type, numElements);

    // 创建 zfp 流并设置压缩容忍度
    zfp_stream* zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, tolerance);

    // 分配临时压缩缓冲区
    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    void* buffer = malloc(bufsize);
    if (!buffer) {
        std::cerr << "Failed to allocate compression buffer" << std::endl;
        return;
    }

    // 设置 bitstream 输出
    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    // 执行压缩
    size_t compressedSize = zfp_compress(zfp, field);
    if (compressedSize == 0) {
        std::cerr << "ZFP compression failed" << std::endl;
        free(buffer);
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);
        return;
    }

    // 写出压缩结果
    std::ofstream out(outputFile, std::ios::binary);
    out.write(reinterpret_cast<const char*>(buffer), compressedSize);
    out.close();

    // 清理资源
    free(buffer);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
}*/

template <typename T>
void ZFP_compress_1D(T* data, size_t numElements, T tolerance, const std::string& outputFile) {
    // 设置维度信息
    zfp_type type;
    if constexpr (std::is_same<T, float>::value) {
        type = zfp_type_float;
    } else if constexpr (std::is_same<T, double>::value) {
        type = zfp_type_double;
    } else {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                      "ZFP_compress_1D only supports float and double.");
    }

    zfp_field* field = zfp_field_1d(data, type, numElements);

    zfp_stream* zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, static_cast<double>(tolerance));

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    void* buffer = malloc(bufsize);
    if (!buffer) {
        std::cerr << "Failed to allocate compression buffer" << std::endl;
        return;
    }

    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    size_t compressedSize = zfp_compress(zfp, field);
    if (compressedSize == 0) {
        std::cerr << "ZFP compression failed" << std::endl;
        free(buffer);
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);
        return;
    }

    std::ofstream out(outputFile, std::ios::binary);
    out.write(reinterpret_cast<const char*>(buffer), compressedSize);
    out.close();

    free(buffer);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
}



/*void computeOptimizedZFP(
    std::vector<std::vector<ScidxAVLNode<double>*>>& singleTree,
    double error_bound,
    const std::string& treeID,
    std::vector<SkippedNode<double>> skippedAllIntervals
) {
    std::vector<size_t> skippedIds;
    std::vector<double> skippedLows;
    std::vector<double> skippedHighs;

    for (const auto& node : skippedAllIntervals) {
        skippedIds.push_back(node.id);
        skippedLows.push_back(node.interval.low);
        skippedHighs.push_back(node.interval.high);
    }

    std::string skippedIdfilename = treeID + "-SkippedNodeID";
    std::string Idfilename = treeID + "-NodeID";

    std::string Minfilename = treeID + "-Min.zfp";
    std::string Maxfilename = treeID + "-Max.zfp";
    std::string MaxHighfilename = treeID + "-MaxHigh.zfp";
    std::string SkippedMinfilename = treeID + "-SkippedMin.zfp";
    std::string SkippedMaxfilename = treeID + "-SkippedMax.zfp";

    std::vector<size_t> treeNodeIdArray = flattenedTreeNodeId(singleTree);
    std::vector<double> treeNodeMinArray = flattenedTreeNodeMin(singleTree);
    std::vector<double> treeNodeMaxArray = flattenedTreeNodeMax(singleTree);
    std::vector<double> treeNodeMaxHighArray = flattenedTreeNodeMaxHigh(singleTree);

    ZFP_compress_1D(skippedLows.data(), skippedLows.size(), error_bound, SkippedMinfilename);
    ZFP_compress_1D(skippedHighs.data(), skippedHighs.size(), error_bound, SkippedMaxfilename);

    ZFP_compress_1D(treeNodeMinArray.data(), treeNodeMinArray.size(), error_bound, Minfilename);
    ZFP_compress_1D(treeNodeMaxArray.data(), treeNodeMaxArray.size(), error_bound, Maxfilename);
    ZFP_compress_1D(treeNodeMaxHighArray.data(), treeNodeMaxHighArray.size(), error_bound, MaxHighfilename);

    writeNodeIdToFile(skippedIdfilename, skippedIds);
    writeNodeIdToFile(Idfilename, treeNodeIdArray);
}*/


template <typename T>
void computeOptimizedZFP(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree,
    T error_bound,
    const std::string& treeID,
    std::vector<SkippedNode<T>> skippedAllIntervals)
{
    std::vector<size_t> skippedIds;
    std::vector<T> skippedLows;
    std::vector<T> skippedHighs;

    for (const auto& node : skippedAllIntervals) {
        skippedIds.push_back(node.id);
        skippedLows.push_back(node.interval.low);
        skippedHighs.push_back(node.interval.high);
    }

    std::string skippedIdfilename = treeID + "-SkippedNodeID";
    std::string Idfilename = treeID + "-NodeID";
    std::string Minfilename = treeID + "-Min.zfp";
    std::string Maxfilename = treeID + "-Max.zfp";
    std::string MaxHighfilename = treeID + "-MaxHigh.zfp";
    std::string SkippedMinfilename = treeID + "-SkippedMin.zfp";
    std::string SkippedMaxfilename = treeID + "-SkippedMax.zfp";

    std::vector<size_t> treeNodeIdArray = flattenedTreeNodeId(singleTree);
    std::vector<T> treeNodeMinArray = flattenedTreeNodeMin(singleTree);
    std::vector<T> treeNodeMaxArray = flattenedTreeNodeMax(singleTree);
    std::vector<T> treeNodeMaxHighArray = flattenedTreeNodeMaxHigh(singleTree);

    ZFP_compress_1D<T>(skippedLows.data(), skippedLows.size(), error_bound, SkippedMinfilename);
    ZFP_compress_1D<T>(skippedHighs.data(), skippedHighs.size(), error_bound, SkippedMaxfilename);
    ZFP_compress_1D<T>(treeNodeMinArray.data(), treeNodeMinArray.size(), error_bound, Minfilename);
    ZFP_compress_1D<T>(treeNodeMaxArray.data(), treeNodeMaxArray.size(), error_bound, Maxfilename);
    ZFP_compress_1D<T>(treeNodeMaxHighArray.data(), treeNodeMaxHighArray.size(), error_bound, MaxHighfilename);

    writeNodeIdToFile(skippedIdfilename, skippedIds);
    writeNodeIdToFile(Idfilename, treeNodeIdArray);
}




// ----------------------------------------------------------------
// 1. computeHighBayesOptimizedAVLFixOrder
//    改动：返回值新增 predictedHighArray(在压缩的时候返回解压的high用于第二个index的预测)
// ----------------------------------------------------------------
template<typename T>
auto computeHighBayesOptimizedAVLFixOrderNew(
    const std::vector<MinMaxPair<T>>& minMaxPairs,
    T error_bound,
    std::string treeID
) -> std::tuple<std::vector<int>, std::vector<T>, std::vector<T>>
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
        T curLow = minMaxPairs[i].reconstructedMin;
        T curHigh = minMaxPairs[i].originalMax;

        int lowBin = findLowBin(curLow, lowMin, lowMax, lowBinCount);
        T highBin = lowHighBinMapping[lowBin].second;
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

// ----------------------------------------------------------------
// 2. 反量化函数: 从 type 还原 high (用于Bayes预测产生的type, 非链式)
//    state=0 对应偏移量为0（核对自computeTypeForFlattenedArray的行为）
// ----------------------------------------------------------------
template<typename T>
T reconstructHighFromType(
    int type_i,
    T predictedHigh,
    T error_bound,
    int intvRadius,
    T interval,                      // = 2*error_bound
    const std::vector<T>& outLayerData,
    size_t& outlierIdx
) {
    if (type_i == 0) {
        T val = outLayerData[outlierIdx];
        outlierIdx++;
        return val;
    }

    int state;
    bool isPositive;
    if (type_i >= intvRadius) {
        state = type_i - intvRadius;
        isPositive = true;
    } else {
        state = intvRadius - type_i;
        isPositive = false;
    }

    T offset = static_cast<T>(state) * interval;  // state * 2 * error_bound

    return isPositive ? (predictedHigh + offset) : (predictedHigh - offset);
}

// ----------------------------------------------------------------
// 3. 反量化函数：用于 computeTypeForFlattenedArray 产生的链式 type
//    （skipped节点的High值走的是这条链式逻辑）
//    必须按顺序、从头模拟整条链，不能随机访问单个元素
// ----------------------------------------------------------------
template<typename T>
std::vector<T> reconstructChainFromType(
    const std::vector<int>& type,
    T error_bound,
    int intvRadius,
    T interval,
    const std::vector<T>& outLayerData
) {
    std::vector<T> reconstructed(type.size());
    size_t outlierIdx = 0;
    T predData = 0;

    for (size_t i = 0; i < type.size(); i++) {
        if (i == 0) {
            // 第一个数据无损存储，直接是outLayerData[0]
            T val = outLayerData[outlierIdx];
            outlierIdx++;
            predData = val;
            reconstructed[i] = val;
        } else {
            if (type[i] == 0) {
                // outlier
                T val = outLayerData[outlierIdx];
                outlierIdx++;
                predData = val;
                reconstructed[i] = val;
            } else {
                int state;
                bool isPositive;
                if (type[i] >= intvRadius) {
                    state = type[i] - intvRadius;
                    isPositive = true;
                } else {
                    state = intvRadius - type[i];
                    isPositive = false;
                }
                T offset = static_cast<T>(state) * interval;
                predData = isPositive ? (predData + offset) : (predData - offset);
                reconstructed[i] = predData;
            }
        }
    }

    return reconstructed;
}

// ----------------------------------------------------------------
// 4. computeOptimizedAVL（用于I1/uniform）
//    改动：新增可选输出参数 outListHigh
// ----------------------------------------------------------------
template<typename T>
void computeOptimizedAVLNew(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree,
    T error_bound,
    std::string treeID,
    std::vector<SkippedNode<T>> skippedAllIntervals,
    std::vector<T>* outListHigh 
) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<size_t> skippedIds;
    std::vector<T> skippedLows;
    std::vector<T> skippedHighs;

    for (const auto& node : skippedAllIntervals) {
        skippedIds.push_back(node.id);
        skippedLows.push_back(node.interval.low);
        skippedHighs.push_back(node.interval.high);
    }

    std::string skippedIdfilename = treeID + "-SkippedNodeID";
    std::vector<size_t> treeNodeIdArray = flattenedTreeNodeId(singleTree);
    std::string Idfilename = treeID + "-NodeID";

    auto compressStart_time = std::chrono::high_resolution_clock::now();

    std::pair<std::vector<int>, std::vector<T>> skippedTypeMin =
        computeTypeForFlattenedArray(skippedLows, error_bound);
    std::pair<std::vector<int>, std::vector<T>> skippedTypeMax =
        computeTypeForFlattenedArray(skippedHighs, error_bound);

    std::vector<int> skippedTypeMinVector = skippedTypeMin.first;
    std::vector<T> skippedOutLayerMinData = skippedTypeMin.second;
    std::vector<int> skippedTypeMaxVector = skippedTypeMax.first;
    std::vector<T> skippedOutLayerMaxData = skippedTypeMax.second;

    auto [typeMinVector, outLayerMinData, minMaxPairs] =
        preQuantiOptimizedMinBayes(singleTree, error_bound);

    std::string csvFileName = treeID + "-low_high_bin_mapping.csv";
    calculateConditionalProbability(minMaxPairs, 100, 100, csvFileName);

    auto [typeMaxVector, outLayerMaxData, predictedHighArray] =
        computeHighBayesOptimizedAVLFixOrderNew(minMaxPairs, error_bound, treeID);

    auto quantiEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_quanti_time = quantiEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time new] pre_quanti_time: " << pre_quanti_time.count()
              << " seconds" << std::endl;

    // ============================================================
    // 新增：构建 list_high
    // ============================================================
    if (outListHigh != nullptr) {
        const int quantization_intervals = 16384;
        const int intvRadius = quantization_intervals / 2;
        const T interval = 2 * error_bound;

        size_t maxBlockId = 0;
        bool hasValidId = false;
        for (auto id : treeNodeIdArray) {
            if (id != static_cast<size_t>(-1)) {
                maxBlockId = std::max(maxBlockId, id);
                hasValidId = true;
            }
        }
        for (auto id : skippedIds) {
            if (id != static_cast<size_t>(-1)) {
                maxBlockId = std::max(maxBlockId, id);
                hasValidId = true;
            }
        }

        outListHigh->assign(hasValidId ? (maxBlockId + 1) : 0,
                             std::numeric_limits<T>::quiet_NaN());

        // 正常(非skipped)节点：用Bayes预测的反量化
        size_t outlierIdx = 0;
        for (size_t i = 0; i < typeMaxVector.size(); i++) {
            size_t block_id = treeNodeIdArray[i];
            if (block_id == static_cast<size_t>(-1)) {
                continue;
            }

            T reconstructed_high = reconstructHighFromType<T>(
                typeMaxVector[i], predictedHighArray[i], error_bound,
                intvRadius, interval, outLayerMaxData, outlierIdx
            );

            (*outListHigh)[block_id] = reconstructed_high;
        }

        // skipped节点：用链式反量化（必须重建整条链，顺序对应skippedIds的顺序）
        std::vector<T> reconstructedSkippedHigh = reconstructChainFromType<T>(
            skippedTypeMaxVector, error_bound, intvRadius, interval, skippedOutLayerMaxData
        );

        for (size_t i = 0; i < skippedIds.size(); i++) {
            size_t block_id = skippedIds[i];
            if (block_id != static_cast<size_t>(-1) && block_id < outListHigh->size()) {
                (*outListHigh)[block_id] = reconstructedSkippedHigh[i];
            }
        }
    }

    // ============================================================
    // 后续流程：完全不变
    // ============================================================
    std::vector<size_t> typeSizes;
    typeSizes.push_back(typeMinVector.size());
    typeSizes.push_back(typeMaxVector.size());
    typeSizes.push_back(skippedTypeMinVector.size());
    typeSizes.push_back(skippedTypeMaxVector.size());

    std::string Minfilename = treeID + "-Min";
    std::string Maxfilename = treeID + "-Max";
    std::string SkippedMinfilename = treeID + "-SkippedMin";
    std::string SkippedMaxfilename = treeID + "-SkippedMax";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);
    HuffmanTree* huffmanTreeSkippedMin = processHuffmanTreeNaiveAVL(skippedTypeMinVector, SkippedMinfilename);
    HuffmanTree* huffmanTreeSkippedMax = processHuffmanTreeNaiveAVL(skippedTypeMaxVector, SkippedMaxfilename);

    auto huffmanBuildEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanBuild_time = huffmanBuildEnd_time - quantiEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time new] huffmanBuild_time_withIo: " << huffmanBuild_time.count()
              << " seconds" << std::endl;

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);
    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);
    std::vector<unsigned char> compressedSkippedMin = singleHuffmanEncodeZstdAVL(skippedTypeMinVector, huffmanTreeSkippedMin);
    std::vector<unsigned char> compressedSkippedMax = singleHuffmanEncodeZstdAVL(skippedTypeMaxVector, huffmanTreeSkippedMax);

    auto compressEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> encode_sztd_time = compressEnd_time - huffmanBuildEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time new] encode_sztd_time: " << encode_sztd_time.count()
              << " seconds" << std::endl;

    std::chrono::duration<double> compress_time = compressEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time 2] compress time with time a ,b: " << compress_time.count()
              << " seconds" << std::endl;

    auto writeBegin_time = std::chrono::high_resolution_clock::now();

    writeNodeBitsIdToFile(skippedIdfilename, skippedIds);
    writeNodeBitsIdToFile(Idfilename, treeNodeIdArray);

    


    writeOutLayerDataToFileOptimizedTest(treeID, outLayerMinData, outLayerMaxData,
                                          skippedOutLayerMinData, skippedOutLayerMaxData);
    writeTypeSizesToFile(treeID, typeSizes);
    saveCompressedDataToFilesTest(treeID, compressedMin, compressedMax,
                                   compressedSkippedMin, compressedSkippedMax);

    auto writeEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = writeEnd_time - writeBegin_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time 3] write time without time a ,b: " << write_time.count()
              << " seconds" << std::endl;
}

// ----------------------------------------------------------------
// 5. 几何映射：stagger坐标 -> 8个uniform邻居id
//    按你的分块对应关系：
//    if (global_id < halfBlockShape) stagger_id = 0
//    else stagger_id = 1 + (global_id - halfBlockShape)/smallBlockShape
//
//    反推：stagger坐标c(c≥1)覆盖的体素范围与uniform坐标(c-1)和(c)重叠
//          stagger坐标0覆盖的体素范围只与uniform坐标0重叠（不完整边界块）
// ----------------------------------------------------------------

std::vector<size_t> positionToIndicesColMajor(size_t position, const std::vector<size_t>& shape) {
    std::vector<size_t> indices(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        indices[i] = position % shape[i];
        position /= shape[i];
    }
    return indices;
}

inline std::vector<size_t> getRelated8UniformBlockIds(
    const std::vector<size_t>& staggerBlockCoord,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    size_t nDim
) {
    std::vector<size_t> relatedIds;
    size_t numNeighbors = static_cast<size_t>(1) << nDim;

    for (size_t mask = 0; mask < numNeighbors; mask++) {
        std::vector<size_t> uniformCoord(nDim);
        bool valid = true;

        for (size_t d = 0; d < nDim; d++) {
            size_t offset = (mask >> d) & 1;

            if (staggerBlockCoord[d] == 0) {
                // 边界：只对应uniform坐标0
                if (offset == 1) {
                    valid = false;
                    break;
                }
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
            relatedIds.push_back(indicesToPosition(uniformBlockCountOnEachDim, uniformCoord));
        }
    }

    return relatedIds;
}



// ----------------------------------------------------------------
// 6. I2(stagger)专用压缩函数
// ----------------------------------------------------------------
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
) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<size_t> skippedIds;
    std::vector<T> skippedLows;
    std::vector<T> skippedHighs;
    for (const auto& node : skippedAllIntervals) {
        skippedIds.push_back(node.id);
        skippedLows.push_back(node.interval.low);
        skippedHighs.push_back(node.interval.high);
    }

    std::string skippedIdfilename = treeID + "-SkippedNodeID";
    std::vector<size_t> treeNodeIdArray = flattenedTreeNodeId(staggerTree);
    std::string Idfilename = treeID + "-NodeID";

    auto compressStart_time = std::chrono::high_resolution_clock::now();

    std::pair<std::vector<int>, std::vector<T>> skippedTypeMin =
        computeTypeForFlattenedArray(skippedLows, error_bound);
    std::pair<std::vector<int>, std::vector<T>> skippedTypeMax =
        computeTypeForFlattenedArray(skippedHighs, error_bound);

    std::vector<int> skippedTypeMinVector = skippedTypeMin.first;
    std::vector<T> skippedOutLayerMinData = skippedTypeMin.second;
    std::vector<int> skippedTypeMaxVector = skippedTypeMax.first;
    std::vector<T> skippedOutLayerMaxData = skippedTypeMax.second;

    auto [typeMinVector, outLayerMinData, minMaxPairs] =
        preQuantiOptimizedMinBayes(staggerTree, error_bound);

    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T checkRadius = (quantization_intervals - 1) * error_bound;
    const T interval = 2 * error_bound;
    const T recip_precision = 1 / error_bound;

    std::vector<int> typeMaxVector(minMaxPairs.size(), 0);
    std::vector<T> outLayerMaxData;

    for (size_t i = 0; i < minMaxPairs.size(); i++) {
        T curHigh = minMaxPairs[i].originalMax;
        size_t block_id = treeNodeIdArray[i];

        if (block_id == static_cast<size_t>(-1)) {
            typeMaxVector[i] = 0;
            outLayerMaxData.push_back(curHigh);
            continue;
        }

        std::vector<size_t> staggerCoord = positionToIndicesColMajor(block_id, staggerBlockCountOnEachDim);
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

        if (count == 0) {
            typeMaxVector[i] = 0;
            outLayerMaxData.push_back(curHigh);
        } else {
            T predictedHigh = sum / static_cast<T>(count);
            T predAbsErr = std::abs(predictedHigh - curHigh);

            if (predAbsErr < checkRadius) {
                int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                if (curHigh >= predictedHigh) {
                    typeMaxVector[i] = intvRadius + state;
                } else {
                    typeMaxVector[i] = intvRadius - state;
                }
            } else {
                typeMaxVector[i] = 0;
                outLayerMaxData.push_back(curHigh);
            }
        }
    }

    auto quantiEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_quanti_time = quantiEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time new][Stagger CrossPred] pre_quanti_time: " << pre_quanti_time.count()
              << " seconds" << std::endl;

    // ============================================================
    // 输出 predictedHigh vs curHigh 差值的 csv 文件
    // ============================================================
    {
        std::cout << "DEBUG stagger shape: ["
                  << staggerBlockCountOnEachDim[0] << ","
                  << staggerBlockCountOnEachDim[1] << ","
                  << staggerBlockCountOnEachDim[2] << "]" << std::endl;
        std::cout << "DEBUG uniform shape: ["
                  << uniformBlockCountOnEachDim[0] << ","
                  << uniformBlockCountOnEachDim[1] << ","
                  << uniformBlockCountOnEachDim[2] << "]" << std::endl;

        size_t dim0 = staggerBlockCountOnEachDim[0];
        size_t dim1 = staggerBlockCountOnEachDim[1];
        size_t dim2 = staggerBlockCountOnEachDim[2];
        size_t sliceDim1 = dim1 / 2;
        size_t gridDim0 = dim0 - 1;
        size_t gridDim2 = dim2 - 1;

        std::vector<float> diffGrid(gridDim0 * gridDim2, 0.0f);
        std::vector<float> predGrid(gridDim0 * gridDim2, 0.0f);
        std::vector<float> curGrid(gridDim0 * gridDim2, 0.0f);

        bool printed = false;
        for (size_t i = 0; i < minMaxPairs.size(); i++) {
            size_t block_id = treeNodeIdArray[i];
            if (block_id == static_cast<size_t>(-1)) continue;

            std::vector<size_t> coord = positionToIndicesColMajor(block_id, staggerBlockCountOnEachDim);

            if (!printed) {
                std::cout << "DEBUG first block_id=" << block_id
                          << " coord=[" << coord[0] << "," << coord[1] << "," << coord[2] << "]" << std::endl;
                printed = true;
            }

            if (coord[1] != sliceDim1) continue;
            if (coord[0] >= gridDim0 || coord[2] >= gridDim2) continue;

            T curHigh = minMaxPairs[i].originalMax;

            std::vector<size_t> relatedIds = getRelated8UniformBlockIds(
                coord, uniformBlockCountOnEachDim, nDim);

            T sum = 0;
            size_t count = 0;
            for (auto rid : relatedIds) {
                if (rid < listHigh.size() && !std::isnan(listHigh[rid])) {
                    sum += listHigh[rid];
                    count++;
                }
            }
            if (count == 0) continue;
            T predictedHigh = sum / static_cast<T>(count);

            // dim0对应行，dim2对应列
            
            
            size_t idx = coord[2] + coord[0] * gridDim2;
            diffGrid[idx] = static_cast<float>(predictedHigh - curHigh);
            predGrid[idx] = static_cast<float>(predictedHigh);
            curGrid[idx]  = static_cast<float>(curHigh);
        }

        std::string csvName = treeID + "-highPredDiff-sliceY.csv";
        std::ofstream csv(csvName);
        csv << "dim2,dim0,predDiff,predictedHigh,curHigh\n";
        for (size_t d0 = 0; d0 < gridDim0; d0++) {
            for (size_t d2 = 0; d2 < gridDim2; d2++) {
                size_t idx = d2 + d0 * gridDim2;
                csv << d0 << "," << d2 << "," << diffGrid[idx] << "," << predGrid[idx] << "," << curGrid[idx] << "\n";
            }
        }
        csv.close();
        std::cout << "Written: " << csvName << " (sliceDim1=" << sliceDim1 << ")" << std::endl;
    }

    std::vector<size_t> typeSizes;
    typeSizes.push_back(typeMinVector.size());
    typeSizes.push_back(typeMaxVector.size());
    typeSizes.push_back(skippedTypeMinVector.size());
    typeSizes.push_back(skippedTypeMaxVector.size());

    std::string Minfilename = treeID + "-Min";
    std::string Maxfilename = treeID + "-Max";
    std::string SkippedMinfilename = treeID + "-SkippedMin";
    std::string SkippedMaxfilename = treeID + "-SkippedMax";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);
    HuffmanTree* huffmanTreeSkippedMin = processHuffmanTreeNaiveAVL(skippedTypeMinVector, SkippedMinfilename);
    HuffmanTree* huffmanTreeSkippedMax = processHuffmanTreeNaiveAVL(skippedTypeMaxVector, SkippedMaxfilename);

    auto huffmanBuildEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanBuild_time = huffmanBuildEnd_time - quantiEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time new][Stagger CrossPred] huffmanBuild_time_withIo: " << huffmanBuild_time.count()
              << " seconds" << std::endl;

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);
    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);
    std::vector<unsigned char> compressedSkippedMin = singleHuffmanEncodeZstdAVL(skippedTypeMinVector, huffmanTreeSkippedMin);
    std::vector<unsigned char> compressedSkippedMax = singleHuffmanEncodeZstdAVL(skippedTypeMaxVector, huffmanTreeSkippedMax);

    auto compressEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> encode_sztd_time = compressEnd_time - huffmanBuildEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time new][Stagger CrossPred] encode_sztd_time: " << encode_sztd_time.count()
              << " seconds" << std::endl;

    std::chrono::duration<double> compress_time = compressEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time 2][Stagger CrossPred] compress time: " << compress_time.count()
              << " seconds" << std::endl;

    auto writeBegin_time = std::chrono::high_resolution_clock::now();

    writeNodeBitsIdToFile(skippedIdfilename, skippedIds);
    writeNodeBitsIdToFile(Idfilename, treeNodeIdArray);
    writeOutLayerDataToFileOptimizedTest(treeID, outLayerMinData, outLayerMaxData,
                                          skippedOutLayerMinData, skippedOutLayerMaxData);
    writeTypeSizesToFile(treeID, typeSizes);
    saveCompressedDataToFilesTest(treeID, compressedMin, compressedMax,
                                   compressedSkippedMin, compressedSkippedMax);

    auto writeEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = writeEnd_time - writeBegin_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time 3][Stagger CrossPred] write time: " << write_time.count()
              << " seconds" << std::endl;
}






/*template<typename T>
void computeOptimizedAVLForStaggerWithCrossPrediction(
    std::vector<std::vector<ScidxAVLNode<T>*>>& staggerTree,
    T error_bound,
    std::string treeID,
    std::vector<SkippedNode<T>> skippedAllIntervals,
    const std::vector<T>& listHigh,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    size_t nDim
) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<size_t> skippedIds;
    std::vector<T> skippedLows;
    std::vector<T> skippedHighs;
    for (const auto& node : skippedAllIntervals) {
        skippedIds.push_back(node.id);
        skippedLows.push_back(node.interval.low);
        skippedHighs.push_back(node.interval.high);
    }

    std::string skippedIdfilename = treeID + "-SkippedNodeID";
    std::vector<size_t> treeNodeIdArray = flattenedTreeNodeId(staggerTree);
    std::string Idfilename = treeID + "-NodeID";

    auto compressStart_time = std::chrono::high_resolution_clock::now();

    std::pair<std::vector<int>, std::vector<T>> skippedTypeMin =
        computeTypeForFlattenedArray(skippedLows, error_bound);
    std::pair<std::vector<int>, std::vector<T>> skippedTypeMax =
        computeTypeForFlattenedArray(skippedHighs, error_bound);

    std::vector<int> skippedTypeMinVector = skippedTypeMin.first;
    std::vector<T> skippedOutLayerMinData = skippedTypeMin.second;
    std::vector<int> skippedTypeMaxVector = skippedTypeMax.first;
    std::vector<T> skippedOutLayerMaxData = skippedTypeMax.second;

    auto [typeMinVector, outLayerMinData, minMaxPairs] =
        preQuantiOptimizedMinBayes(staggerTree, error_bound);

    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T checkRadius = (quantization_intervals - 1) * error_bound;
    const T interval = 2 * error_bound;
    const T recip_precision = 1 / error_bound;

    std::vector<int> typeMaxVector(minMaxPairs.size(), 0);
    std::vector<T> outLayerMaxData;

    for (size_t i = 0; i < minMaxPairs.size(); i++) {
        T curHigh = minMaxPairs[i].originalMax;
        size_t block_id = treeNodeIdArray[i];

        if (block_id == static_cast<size_t>(-1)) {
            typeMaxVector[i] = 0;
            outLayerMaxData.push_back(curHigh);
            continue;
        }

        std::vector<size_t> staggerCoord = positionToIndices(block_id, staggerBlockCountOnEachDim);
        std::vector<size_t> relatedIds = getRelated8UniformBlockIds(
            staggerCoord, uniformBlockCountOnEachDim, nDim);

        // ============================================================
        // 改动点：只取固定方向的单一邻居（最后一个，对应mask最大、
        // 即所有维度offset=1的"右上后"方向邻居），不再做任何统计聚合
        // ============================================================
        bool hasValidNeighbor = false;
        T predictedHigh = 0;

        if (!relatedIds.empty()) {
            size_t fixedRid = relatedIds.back();
            if (fixedRid < listHigh.size() && !std::isnan(listHigh[fixedRid])) {
                predictedHigh = listHigh[fixedRid];
                hasValidNeighbor = true;
            }
        }
        // ============================================================

        if (!hasValidNeighbor) {
            typeMaxVector[i] = 0;
            outLayerMaxData.push_back(curHigh);
        } else {
            T predAbsErr = std::abs(predictedHigh - curHigh);

            if (predAbsErr < checkRadius) {
                int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                if (curHigh >= predictedHigh) {
                    typeMaxVector[i] = intvRadius + state;
                } else {
                    typeMaxVector[i] = intvRadius - state;
                }
            } else {
                typeMaxVector[i] = 0;
                outLayerMaxData.push_back(curHigh);
            }
        }
    }

    auto quantiEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pre_quanti_time = quantiEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time new][Stagger CrossPred] pre_quanti_time: " << pre_quanti_time.count()
              << " seconds" << std::endl;

    std::vector<size_t> typeSizes;
    typeSizes.push_back(typeMinVector.size());
    typeSizes.push_back(typeMaxVector.size());
    typeSizes.push_back(skippedTypeMinVector.size());
    typeSizes.push_back(skippedTypeMaxVector.size());

    std::string Minfilename = treeID + "-Min";
    std::string Maxfilename = treeID + "-Max";
    std::string SkippedMinfilename = treeID + "-SkippedMin";
    std::string SkippedMaxfilename = treeID + "-SkippedMax";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);
    HuffmanTree* huffmanTreeSkippedMin = processHuffmanTreeNaiveAVL(skippedTypeMinVector, SkippedMinfilename);
    HuffmanTree* huffmanTreeSkippedMax = processHuffmanTreeNaiveAVL(skippedTypeMaxVector, SkippedMaxfilename);

    auto huffmanBuildEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> huffmanBuild_time = huffmanBuildEnd_time - quantiEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time new][Stagger CrossPred] huffmanBuild_time_withIo: " << huffmanBuild_time.count()
              << " seconds" << std::endl;

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);
    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);
    std::vector<unsigned char> compressedSkippedMin = singleHuffmanEncodeZstdAVL(skippedTypeMinVector, huffmanTreeSkippedMin);
    std::vector<unsigned char> compressedSkippedMax = singleHuffmanEncodeZstdAVL(skippedTypeMaxVector, huffmanTreeSkippedMax);

    auto compressEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> encode_sztd_time = compressEnd_time - huffmanBuildEnd_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time new][Stagger CrossPred] encode_sztd_time: " << encode_sztd_time.count()
              << " seconds" << std::endl;

    std::chrono::duration<double> compress_time = compressEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time 2][Stagger CrossPred] compress time: " << compress_time.count()
              << " seconds" << std::endl;

    auto writeBegin_time = std::chrono::high_resolution_clock::now();

    writeNodeBitsIdToFile(skippedIdfilename, skippedIds);
    writeNodeBitsIdToFile(Idfilename, treeNodeIdArray);
    writeOutLayerDataToFileOptimizedTest(treeID, outLayerMinData, outLayerMaxData,
                                          skippedOutLayerMinData, skippedOutLayerMaxData);
    writeTypeSizesToFile(treeID, typeSizes);
    saveCompressedDataToFilesTest(treeID, compressedMin, compressedMax,
                                   compressedSkippedMin, compressedSkippedMax);

    auto writeEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> write_time = writeEnd_time - writeBegin_time;
    std::cout << "Current MPI rank: " << rank
              << "[Time 3][Stagger CrossPred] write time: " << write_time.count()
              << " seconds" << std::endl;
}*/





template<typename T>
void computeOptimizedAVL(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
    T error_bound, 
    std::string treeID, std::vector<SkippedNode<T>> skippedAllIntervals
) {

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);



    std::vector<size_t> skippedIds;
    std::vector<T> skippedLows;
    std::vector<T> skippedHighs;

    //提取skipped数据
    for (const auto& node : skippedAllIntervals) {
        skippedIds.push_back(node.id);           // 提取 id
        skippedLows.push_back(node.interval.low); // 提取 low 值
        skippedHighs.push_back(node.interval.high); // 提取 high 值
    }

     
    std::string skippedIdfilename = treeID + "-SkippedNodeID";

    //tree展平id
    std::vector<size_t> treeNodeIdArray = flattenedTreeNodeId(singleTree);

    //id不压缩直接写文件
    std::string Idfilename = treeID + "-NodeID";

  

    auto compressStart_time = std::chrono::high_resolution_clock::now();



    //展平skipped数据
    std::pair<std::vector<int>, std::vector<T>> skippedTypeMin = computeTypeForFlattenedArray(skippedLows, error_bound);
    std::pair<std::vector<int>, std::vector<T>> skippedTypeMax = computeTypeForFlattenedArray(skippedHighs, error_bound);

    std::vector<int> skippedTypeMinVector = skippedTypeMin.first;
    std::vector<T> skippedOutLayerMinData = skippedTypeMin.second;

    std::vector<int> skippedTypeMaxVector = skippedTypeMax.first;
    std::vector<T> skippedOutLayerMaxData = skippedTypeMax.second;



    //只获取叶子节点的max_high
    //std::vector<T> leafNodeMaxHighArray = getLeafNodeMaxHigh(singleTree);
    //std::pair<std::vector<int>, std::vector<T>> typeMaxHigh = computeTypeForFlattenedArray(leafNodeMaxHighArray, error_bound);


    

    //minMaxPairs是“解压min”和“max”的pair,并是min压缩相同的顺序
    auto [typeMinVector, outLayerMinData, minMaxPairs] = preQuantiOptimizedMinBayes(singleTree, error_bound);

    std::string csvFileName = treeID + "-low_high_bin_mapping.csv";

    calculateConditionalProbability(minMaxPairs, 100, 100, csvFileName);

    //用压缩前的节点high，对应的节点low去算bayes得到的作为high的预测值
    std::pair<std::vector<int>, std::vector<T>> typeMax = computeHighBayesOptimizedAVLFixOrder(minMaxPairs, error_bound, treeID);

    auto quantiEnd_time = std::chrono::high_resolution_clock::now();

    //pre+quanti时间
    std::chrono::duration<double> pre_quanti_time = quantiEnd_time - compressStart_time;
    std::cout << "Current MPI rank: " << rank << "[Time new] pre_quanti_time: " << pre_quanti_time.count() << " seconds" << std::endl;


    std::vector<int> typeMaxVector = typeMax.first;
    std::vector<T> outLayerMaxData = typeMax.second;
    
    std::string Minfilename = treeID + "-Min";
    std::string Maxfilename = treeID + "-Max";
    std::string SkippedMinfilename = treeID + "-SkippedMin";
    std::string SkippedMaxfilename = treeID + "-SkippedMax";


    //std::string MaxHighfilename = treeID + "-MaxHigh";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);
    //HuffmanTree* huffmanTreeMaxHigh = processHuffmanTreeNaiveAVL(typeMaxHighVector, MaxHighfilename);

    HuffmanTree* huffmanTreeSkippedMin = processHuffmanTreeNaiveAVL(skippedTypeMinVector, SkippedMinfilename);
    HuffmanTree* huffmanTreeSkippedMax = processHuffmanTreeNaiveAVL(skippedTypeMaxVector, SkippedMaxfilename);

    auto huffmanBuildEnd_time = std::chrono::high_resolution_clock::now();

    //Huffman构建时间
    std::chrono::duration<double> huffmanBuild_time = huffmanBuildEnd_time - quantiEnd_time;
    std::cout << "Current MPI rank: " << rank << "[Time new] huffmanBuild_time_withIo: " << huffmanBuild_time.count() << " seconds" << std::endl;



    //压缩前type的大小，解压后还原大小
    std::vector<size_t> typeSizes;
    typeSizes.push_back(typeMinVector.size());
    typeSizes.push_back(typeMaxVector.size());
    typeSizes.push_back(skippedTypeMinVector.size());
    typeSizes.push_back(skippedTypeMaxVector.size());

    //typeSizes.push_back(typeMaxHighVector.size());

    //<< "[debug] typeMin.size(): " << typeMinVector.size() << std::endl;
    //std::cout << "[debug] typeMax.size(): " << typeMaxVector.size() << std::endl;

   

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);

    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);

    std::vector<unsigned char> compressedSkippedMin = singleHuffmanEncodeZstdAVL(skippedTypeMinVector, huffmanTreeSkippedMin);

    std::vector<unsigned char> compressedSkippedMax = singleHuffmanEncodeZstdAVL(skippedTypeMaxVector, huffmanTreeSkippedMax);

    auto compressEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> encode_sztd_time = compressEnd_time - huffmanBuildEnd_time;
    std::cout << "Current MPI rank: " << rank << "[Time new] encode_sztd_time: " << encode_sztd_time.count() << " seconds" << std::endl;


    std::chrono::duration<double> compress_time = compressEnd_time - compressStart_time;

    
    std::cout << "Current MPI rank: " << rank << "[Time 2] compress time with time a ,b: " << compress_time.count() << " seconds" << std::endl;



    auto writeBegin_time = std::chrono::high_resolution_clock::now();

    //std::vector<unsigned char> compressedMaxHigh = singleHuffmanEncodeZstdAVL(typeMaxHighVector, huffmanTreeMaxHigh);

    //写文件
    writeNodeBitsIdToFile(skippedIdfilename, skippedIds);

    //写，采用小端bit_packing存储id
    writeNodeBitsIdToFile(Idfilename, treeNodeIdArray);

    //写min,max的outlayer数据(写成binary)
    writeOutLayerDataToFileOptimizedTest(treeID, outLayerMinData, outLayerMaxData, skippedOutLayerMinData, skippedOutLayerMaxData);

    writeTypeSizesToFile(treeID, typeSizes);

    saveCompressedDataToFilesTest(treeID, compressedMin, compressedMax, compressedSkippedMin, compressedSkippedMax);  
    
    auto writeEnd_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> write_time = writeEnd_time - writeBegin_time;

  
    std::cout << "Current MPI rank: " << rank << "[Time 3] write time without time a ,b: " << write_time.count() << " seconds" << std::endl;


}


template<typename T>
void computeOptimizedTestAVL(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
    T error_bound, 
    std::string treeID
) {
    //展平id
    std::vector<size_t> treeNodeIdArray = flattenedTreeNodeId(singleTree);

    //id不压缩直接写文件
    std::string Idfilename = treeID + "-NodeID";
    //采用小端bit_packing存储id
    writeNodeBitsIdToFile(Idfilename, treeNodeIdArray);

    //只获取叶子节点的max_high
    //std::vector<T> leafNodeMaxHighArray = getLeafNodeMaxHigh(singleTree);


    //获取low bin -> high bin 的映射，用于后续通过bayes预测high


    //按照行序得到压缩前low,high的pair
    //std::vector<std::pair<T, T>> flattenedNodeLowHigh = flattenedTreeNodeLowHigh(singleTree);

    //用压缩前的节点high，对应的节点low去算bayes得到的作为high的预测值
    //std::pair<std::vector<int>, std::vector<T>> typeMax = computeHighBayesOptimizedAVL(flattenedNodeLowHigh, error_bound, treeID);

    //std::pair<std::vector<int>, std::vector<T>> typeMaxHigh = computeTypeForFlattenedArray(leafNodeMaxHighArray, error_bound);


    //
    std::vector<T> treeNodeMaxArray = flattenedTreeNodeMax(singleTree);

    std::pair<std::vector<int>, std::vector<T>> typeMax = computeTypeForFlattenedArray(treeNodeMaxArray, error_bound);

 
    std::pair<std::vector<int>, std::vector<T>> typeMin = preQuantiOptimizedMin(singleTree, error_bound);

    std::vector<int> typeMinVector = typeMin.first;
    std::vector<T> outLayerMinData = typeMin.second;

    std::vector<int> typeMaxVector = typeMax.first;
    std::vector<T> outLayerMaxData = typeMax.second;
    std::vector<T> skippedOutLayerMinData; 
    std::vector<T> skippedOutLayerMaxData; 

    //std::vector<int> typeMaxHighVector = typeMaxHigh.first;
    //std::vector<T> outLayerMaxHighData = typeMaxHigh.second;

    //写min,max的outlayer数据(写成binary)
    writeOutLayerDataToFileOptimizedTest(treeID, outLayerMinData, outLayerMaxData, skippedOutLayerMinData, skippedOutLayerMaxData);

    std::string Minfilename = treeID + "-Min";
    std::string Maxfilename = treeID + "-Max";
    //std::string MaxHighfilename = treeID + "-MaxHigh";

    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);
    //HuffmanTree* huffmanTreeMaxHigh = processHuffmanTreeNaiveAVL(typeMaxHighVector, MaxHighfilename);


    //压缩前type的大小，解压后还原大小
    std::vector<size_t> typeSizes;
    typeSizes.push_back(typeMinVector.size());
    typeSizes.push_back(typeMaxVector.size());
    //typeSizes.push_back(typeMaxHighVector.size());


    writeTypeSizesToFile(treeID, typeSizes);

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);

    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);

    //std::vector<unsigned char> compressedMaxHigh = singleHuffmanEncodeZstdAVL(typeMaxHighVector, huffmanTreeMaxHigh);
    std::vector<unsigned char> compressedSkippedMin;
    std::vector<unsigned char> compressedSkippedMax;
    
    saveCompressedDataToFilesTest(treeID, compressedMin, compressedMax, compressedSkippedMin, compressedSkippedMax);   

} 




std::vector<unsigned char> loadHuffmanTree(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open " << filename << std::endl;
        return {};
    }
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::vector<unsigned char> readCompressedData(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open " << filename << std::endl;
        return {};
    }
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::vector<size_t> readTypeSizes(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open " << filename << std::endl;
        return {};
    }

    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));

    std::vector<size_t> typeSizes(size);
    file.read(reinterpret_cast<char*>(typeSizes.data()), size * sizeof(size_t));

    return typeSizes;
}

/*std::vector<int> singleHuffmanDecodeZstdAVL(std::vector<unsigned char> compressedByZstdData, size_t typeSize, unsigned char *s) {
    std::vector<unsigned char> decompressedData = decompressWithZstdAVL(compressedByZstdData, typeSize * 4);
    std::vector<int> decodedData(typeSize);

    decode_withSubTreeAVL(s, decompressedData.data(), decodedData.size(), decodedData.data());
    return decodedData;
}*/

template<typename T>
std::vector<T> readOutLayerData(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open " << filename << std::endl;
        return {};
    }

    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));

    std::vector<T> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));

    // for test
    /*std::cout << "Data: ";
    for (size_t i = 0; i < std::min(size_t(10), data.size()); ++i) {
        std::cout << data[i] << " ";
    }
    if (data.size() > 10) std::cout << "...";
    std::cout << std::endl;*/

    return data;
}

std::vector<size_t> readNodeBitsIdFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open " << filename << std::endl;
        return {};
    }

    unsigned char bitsPerId;
    file.read(reinterpret_cast<char*>(&bitsPerId), sizeof(bitsPerId));
    //std::cout << "bitsPerId = " << static_cast<int>(bitsPerId) << std::endl;

    if (bitsPerId == 0) {
        return { static_cast<size_t>(-1) };
    }

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t totalBits = bytes.size() * 8;
    size_t totalIds = totalBits / bitsPerId;

    std::vector<size_t> ids(totalIds, 0);
    size_t bitPos = 0;
    for (size_t& id : ids) {
        for (size_t i = 0; i < bitsPerId; ++i) {
            size_t byteIndex = bitPos / 8;
            size_t bitIndex = bitPos % 8;
            id |= ((bytes[byteIndex] >> bitIndex) & 1) << i;
            ++bitPos;
        }
    }
    return ids;
}

//读取树结构，对应写入的saveTreeStructureToByte方法
std::vector<int> loadTreeStructureFromByte(const std::string& treeID) {
    std::string filename = treeID + "-treeStructure";
    std::ifstream inFile(filename, std::ios::binary);

    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return {};
    }

    // **读取 structureSize**
    size_t structureSize = 0;
    inFile.read(reinterpret_cast<char*>(&structureSize), sizeof(size_t));

    // **解析 bit 存储**
    std::vector<int> structureVec;
    structureVec.reserve(structureSize);

    char byte;
    while (inFile.get(byte)) {
        for (int bitIndex = 0; bitIndex < 8; bitIndex++) {
            if (structureVec.size() < structureSize) {
                structureVec.push_back((byte >> bitIndex) & 1);
            } else {
                break; // **确保不会读取超过 structureSize**
            }
        }
    }

    inFile.close();
    return structureVec;
}

template<typename T>
ScidxAVLNode<T>* reconstructAndDequantizeOptimizedTree(
    const std::vector<int>& type,
    const std::vector<int>& structureVec,
    T error_bound,
    const std::vector<T>& outlayerLow,
    const std::vector<int>& typeMax,              
    const std::vector<T>& outlayerMax,
    const std::string& treeID   
)
{
    if (type.empty() || structureVec.empty()) return nullptr;


    T lowMin, lowMax, highMin, highMax;
    int lowBinCount, highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMapping;

    std::string csvFileName = treeID + "-low_high_bin_mapping.csv";
    std::tie(lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, lowHighBinMapping) =
        loadBinMappingAndStats<T>(csvFileName);

        /*// for test
        std::cout << "lowMin = " << lowMin
        << ", lowMax = " << lowMax
        << ", highMin = " << highMin
        << ", highMax = " << highMax
        << ", lowBinCount = " << lowBinCount
        << ", highBinCount = " << highBinCount
        << std::endl;

    std::cout << "First 10 entries of lowHighBinMapping:" << std::endl;
    for (int i = 0; i <  (int)lowHighBinMapping.size(); ++i) {
    std::cout << "  lowBin: " << lowHighBinMapping[i].first
                << ", highBin: " << lowHighBinMapping[i].second << std::endl;
    }*/


    auto getHighBinMidValue = [&](T highBin) {
        T binWidth = (highMax - highMin) / highBinCount;
        return highMin + highBin * binWidth + binWidth / 2.0;
    };

    const int quantization_intervals = 16384;
    const int intvRadius = quantization_intervals / 2;
    const T interval = 2 * error_bound;

    size_t outlayerIndex = 0;
    size_t typeIndex = 0;
    size_t structIndex = 0;
    size_t outlayerMaxIndex = 0;  



    std::queue<ScidxAVLNode<T>**> nodeQueue;
    ScidxAVLNode<T>* root = nullptr;
    nodeQueue.push(&root);

    T prevNodeVal = 0;                     // 当前层中前一个有效节点的值（structureVec==1）
    T prevLevelFirstValidVal = 0;          // 上一层中第一个有效节点的值
    bool foundFirstValidInThisLevel = false; // 本层是否找到第一个有效节点

    size_t nodesInCurrentLevel = 1;
    size_t processedInCurrentLevel = 0;
    size_t level = 0;

    T firstValidValThisLevel = 0; // 当前层第一个有效节点的值（记录给下一层预测用）

    while (!nodeQueue.empty() && structIndex < structureVec.size()) {
        ScidxAVLNode<T>** currentPtr = nodeQueue.front();
        nodeQueue.pop();

        if (structureVec[structIndex++] == 1) {
            T curVal;
            T curValMaxTest;
          

            if (typeIndex >= type.size()) {
                std::cerr << "Error: type index out of range\n";
                return nullptr;
            }

            int curType = type[typeIndex];

            if (!foundFirstValidInThisLevel) {
                // 当前层的第一个有效节点
                if (level == 0) {
                    // root 节点一定是原始值
                    curVal = outlayerLow[outlayerIndex++];
                } else {
                    // 用上一层的第一个有效节点预测
                    if (curType == 0) {
                        curVal = outlayerLow[outlayerIndex++];
                    } else {
                        curVal = dequantizeValue(curType, error_bound, intvRadius, interval, prevLevelFirstValidVal);
                    }
                }

                firstValidValThisLevel = curVal;  // 本层的第一个有效节点值，留给下一层预测用
                foundFirstValidInThisLevel = true;
            } else {
                // 当前层非首个有效节点，用前一个有效节点预测
                if (curType == 0) {
                    curVal = outlayerLow[outlayerIndex++];
                } else {
                    curVal = dequantizeValue(curType, error_bound, intvRadius, interval, prevNodeVal);
                }
            }

            prevNodeVal = curVal;
            typeIndex++;

            int lowBin = findLowBin(curVal, lowMin, lowMax, lowBinCount);
            T expectedHighBin = lowHighBinMapping[lowBin].second;
            T predictedMax = getHighBinMidValue(expectedHighBin);

            int curMaxType = typeMax[typeIndex - 1];  // 注意：同步索引
            T curMax;
            if (curMaxType == 0) {
                curMax = outlayerMax[outlayerMaxIndex++];  // 同步索引
            } else {

                curValMaxTest = dequantizeValue(curMaxType, error_bound, intvRadius, interval, predictedMax);
                


                if (curMaxType >= intvRadius)
                    curMax = predictedMax + (curMaxType - intvRadius) * interval;
                else
                    curMax = predictedMax - (intvRadius - curMaxType) * interval;

        
            }

            /*if (typeIndex <= 10) {
                std::cout << "[DEBUG] index: " << typeIndex - 1 << "\n";
                std::cout << "  decompressed min: " << curVal << "\n";
                std::cout << "  curMaxType: " << curMaxType << "\n";
                std::cout << "  predictedMax: " << predictedMax << "\n";
                std::cout << "  computed curMax: " << curMax << "\n";
            }*/
            

        

            *currentPtr = new ScidxAVLNode<T>({ScidxInterval<T>{curVal, curMax}, 0});
            nodeQueue.push(&((*currentPtr)->left));
            nodeQueue.push(&((*currentPtr)->right));
        }

        processedInCurrentLevel++;
        if (processedInCurrentLevel == nodesInCurrentLevel) {
            // 当前层遍历结束，准备进入下一层
            nodesInCurrentLevel = nodeQueue.size();
            processedInCurrentLevel = 0;
            level++;
            foundFirstValidInThisLevel = false;
            prevLevelFirstValidVal = firstValidValThisLevel;
        }
    }

    return root;
}










/*template<typename T>
ScidxAVLNode<T>* reconstructAndDequantizeOptimizedTree(
    const std::vector<int>& type, 
    const std::vector<int>& structureVec, 
    T error_bound, 
    const std::vector<T>& outlayerLow) 
{
    if (type.empty() || structureVec.empty() || outlayerLow.empty()) return nullptr;

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T interval = 2 * error_bound;

    size_t outlayerIndex = 0;
    size_t typeIndex = 0;
    size_t structIndex = 0;
    size_t outlayerMaxIndex = 0;



    // Get root value
    T rootVal = outlayerLow[outlayerIndex++];
    ScidxAVLNode<T>* root = new ScidxAVLNode<T>({ScidxInterval<T>{rootVal, rootVal}, 0});

  
    std::queue<ScidxAVLNode<T>**> nodesQueue;
    nodesQueue.push(&root);

    std::queue<T> baseValuesForNextLevel;
    baseValuesForNextLevel.push(rootVal);

    int currentLevelNodeCount = 1;
    int processedNodeCount = 0;
    bool isLevelFirstNode = true;

    while (!nodesQueue.empty() && structIndex < structureVec.size()) {
        ScidxAVLNode<T>** currentNodePtr = nodesQueue.front();
        nodesQueue.pop();
        
        if (isLevelFirstNode && !baseValuesForNextLevel.empty()) {
            firstNodePredData = baseValuesForNextLevel.front();
            baseValuesForNextLevel.pop();
        }

        T dequantizedValue;
        if (structureVec[structIndex] == 1) {
            if (typeIndex == 0 || type[typeIndex] == 0) {
                // **每层的第一个 min 值，直接取 outlayerLow**
                if (outlayerIndex >= outlayerLow.size()) {
                    std::cerr << "Error: outlayerLow index out of range\n";
                    return nullptr;
                }
                dequantizedValue = outlayerLow[outlayerIndex++];
            } else {
                // **后续的 min 值，用前一个 min 进行解码**
                dequantizedValue = dequantizeValue(type[typeIndex], error_bound, intvRadius, interval, firstNodePredData);
            }
            typeIndex++;

            *currentNodePtr = new ScidxAVLNode<T>({ScidxInterval<T>{dequantizedValue, dequantizedValue}, 0});

            if (isLevelFirstNode) {
                baseValuesForNextLevel.push(dequantizedValue);
                isLevelFirstNode = false;
            }
            firstNodePredData = dequantizedValue;

            nodesQueue.push(&((*currentNodePtr)->left));
            nodesQueue.push(&((*currentNodePtr)->right));
        }

        if (++processedNodeCount == currentLevelNodeCount && !nodesQueue.empty()) {
            currentLevelNodeCount = nodesQueue.size();
            processedNodeCount = 0;
            isLevelFirstNode = true;
        }
        structIndex++;
    }

    return root;
}*/


//bayes方式解压high值
template<typename T>
std::vector<T> decodeArrayMaxBayesOptimizedType(const std::vector<int>& typeMax, 
    const std::vector<T>& decompressedLow,
                                       T error_bound, 
                                       std::vector<T> outLayerData, 
                                       const std::string& treeID) {
    // **加载 low -> high bin 映射**
    T lowMin, lowMax, highMin, highMax;
    int lowBinCount, highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMapping;
    std::tie(lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, lowHighBinMapping) = loadBinMappingAndStats<T>(treeID + "-low_high_bin_mapping.csv");

    T binWidth = (highMax - highMin) / highBinCount;
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T interval = 2 * error_bound;

    std::vector<T> decodedValues;
    size_t outlayerIndex = 0;

  
    for (size_t i = 0; i < typeMax.size(); ++i) {
        T predictedHigh;
        T currentLow = decompressedLow[i];
        int lowBin = findLowBin(currentLow, lowMin, lowMax, lowBinCount);
        T highBin = lowHighBinMapping[lowBin].second;

        if (highBin != -1) {
    
            predictedHigh = highMin + highBin * binWidth;
            
        }

        if (typeMax[i] == 0) {
            if (outlayerIndex >= outLayerData.size()) {
                throw std::runtime_error("outLayerData index out of range");
            }
            decodedValues.push_back(outLayerData[outlayerIndex]);
            outlayerIndex++;
        } else {
            int state = std::abs(typeMax[i] - intvRadius);
            T adjustment = state * interval;
            T curData = (typeMax[i] >= intvRadius) ? predictedHigh + adjustment : predictedHigh - adjustment;
            decodedValues.push_back(curData);
        }
    }

    return decodedValues;
}


template<typename T>
std::vector<T> getLevelOrderLowValues(ScidxAVLNode<T>* root) {
    std::vector<T> result;
    if (!root) return result;

    std::queue<ScidxAVLNode<T>*> q;
    q.push(root);

    while (!q.empty()) {
        ScidxAVLNode<T>* node = q.front();
        q.pop();

        if (node) {
            result.push_back(node->interval.low); // 取 low/min 值
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }

    return result;
}


template <typename T>
void reconstructIdFix(ScidxAVLNode<T>* root,  const std::vector<size_t>& idOut){
    
    if (!root) return;

    std::queue<ScidxAVLNode<T>*> queue;
    queue.push(root);

    size_t index = 0;

    while (!queue.empty()  && index < idOut.size()) {
        ScidxAVLNode<T>* current = queue.front();
        queue.pop();

        // 先填入 high 值和 id，然后递增索引
  
        current->id = idOut[index];

        // 打印当前节点的 id
        //std::cout << "id: " << current->id << std::endl;

        index++; // 在使用后递增索引

        if (current->left != nullptr) {
            queue.push(current->left);
        }

        if (current->right != nullptr) {
            queue.push(current->right);
        }
    }
}




template<typename T>
std::tuple<ScidxAVLNode<T>*, std::vector<T>, std::vector<T>, std::vector<size_t>>  decompressOptimizedAVL(const std::string& treeID ,  T error_bound) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    auto beginRead = std::chrono::high_resolution_clock::now(); 
    // **1. 加载 Huffman 树**
    std::vector<unsigned char> huffmanOutLow = loadHuffmanTree(treeID + "-Min-Huffman");
    std::vector<unsigned char> huffmanOutHigh = loadHuffmanTree(treeID + "-Max-Huffman");

    std::vector<unsigned char> huffmanOutSkippedLow = loadHuffmanTree(treeID + "-SkippedMin-Huffman");
    std::vector<unsigned char> huffmanOutSkippedHigh = loadHuffmanTree(treeID + "-SkippedMax-Huffman");


    //std::vector<unsigned char> huffmanOutMaxHigh = loadHuffmanTree(treeID + "-MaxHigh-Huffman");

    // **2. 读取 Huffman 压缩的索引数据**
    std::vector<unsigned char> compressedMin = readCompressedData(treeID + "-Min-compressedIndexData");
    std::vector<unsigned char> compressedMax = readCompressedData(treeID + "-Max-compressedIndexData");
    //std::vector<unsigned char> compressedMaxHigh = readCompressedData(treeID + "-MaxHigh-compressedIndexData");

    std::vector<unsigned char> compressedSkippedMin = readCompressedData(treeID + "-SkippedMin-compressedIndexData");
    std::vector<unsigned char> compressedSkippedMax = readCompressedData(treeID + "-SkippedMax-compressedIndexData");

    // **3. 读取 Type Size**
    std::vector<size_t> typeSizes = readTypeSizes(treeID + "-typeSize");

    

    // **5. 读取 OutLayer 数据**
    std::vector<T> outLayerMin = readOutLayerData<T>(treeID + "-OutMin.bin");
    std::vector<T> outLayerMax = readOutLayerData<T>(treeID + "-OutMax.bin");
    //std::vector<T> outLayerMaxHigh = readOutLayerData<T>(treeID + "-OutMaxHigh.bin");

    std::vector<T> outLayerSkippedMin = readOutLayerData<T>(treeID + "-OutSkippedMin.bin");
    std::vector<T> outLayerSkippedMax = readOutLayerData<T>(treeID + "-OutSkippedMax.bin");


    // **6. 读取 bit-packed ID**
    std::vector<size_t> idOut = readNodeBitsIdFromFile(treeID + "-NodeID");
    
    std::vector<size_t> skippedIdOut = readNodeBitsIdFromFile(treeID + "-SkippedNodeID");


    //读取树结构
    std::vector<int> structureVec = loadTreeStructureFromByte(treeID);
   

    auto endRead = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> readIndexTiime = endRead - beginRead;
    std::cerr << "[Rank " << rank << "[Time3]: readIndexTiime: " << readIndexTiime.count() << " seconds" << std::endl;




    // **4. Huffman 解码**
    std::vector<int> decompressedTypeMin = singleHuffmanDecodeZstdAVL(compressedMin, typeSizes[0], huffmanOutLow.data());
    std::vector<int> decompressedTypeMax = singleHuffmanDecodeZstdAVL(compressedMax, typeSizes[1], huffmanOutHigh.data());
    //std::vector<int> decompressedTypeMaxHigh = singleHuffmanDecodeZstdAVL(compressedMaxHigh, typeSizes[2], huffmanOutMaxHigh.data());

    std::vector<int> decompressedTypeSkippedMin = singleHuffmanDecodeZstdAVL(compressedSkippedMin, typeSizes[2], huffmanOutSkippedLow.data());
    std::vector<int> decompressedTypeSkippedMax = singleHuffmanDecodeZstdAVL(compressedSkippedMax, typeSizes[3], huffmanOutSkippedHigh.data());

    


      //解压leaf node的max_high
    //std::vector<T> decodedMaxHigh = decodeArrayType(decompressedTypeMaxHigh, error_bound, outLayerMaxHigh);


    std::vector<T> decodedSkippedMin = decodeArrayType(decompressedTypeSkippedMin, error_bound, outLayerSkippedMin);
    
    std::vector<T> decodedSkippedMax = decodeArrayType(decompressedTypeSkippedMax, error_bound, outLayerSkippedMax);
    
    

    // **7. 重建 AVL 树**
    ScidxAVLNode<T>* reconstructMin = reconstructAndDequantizeOptimizedTree(decompressedTypeMin, structureVec, error_bound, outLayerMin, decompressedTypeMax, outLayerMax, treeID );


    //根据bayes解压high(已经填充了outlayer)
    //std::vector<double> decodeFlattenedMax = decodeArrayMaxBayesOptimizedType(decompressedTypeMax, levelOrderDecompressedMin, error_bound, outLayerMax, treeID);


    // **8. 填充 high 值和id
    //reconstructMax(reconstructMin, decodeFlattenedMax, idOut);


    reconstructIdFix(reconstructMin, idOut);

    //填充 max_high 叶子值
    //reconstractLeafMaxHigh(reconstructMin, decodedMaxHigh);
    
    // **10. 更新整棵树的 max_high**
    //updateMaxHighOfTree(reconstructMin);

    //std::cout << "Decompression Completed: " << treeID << std::endl;

    auto endDecompress = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> decompressAndReconstractTree = endDecompress - endRead;
    std::cerr << "[Rank " << rank << "[Time4]: decompressAndReconstractTree: " << decompressAndReconstractTree.count() << " seconds" << std::endl;

    return {reconstructMin, decodedSkippedMin, decodedSkippedMax, skippedIdOut};
}






template<typename T>
ScidxAVLNode<T>* decompressOptimizedTestAVL(const std::string& treeID ,  T error_bound) {
    //std::cout << "Decompressing AVL Tree: " << treeID << std::endl;

    auto beginRead = std::chrono::high_resolution_clock::now(); 


    // **1. 加载 Huffman 树**
    std::vector<unsigned char> huffmanOutLow = loadHuffmanTree(treeID + "-Min-Huffman");
    std::vector<unsigned char> huffmanOutHigh = loadHuffmanTree(treeID + "-Max-Huffman");
    //std::vector<unsigned char> huffmanOutMaxHigh = loadHuffmanTree(treeID + "-MaxHigh-Huffman");

    // **2. 读取 Huffman 压缩的索引数据**
    std::vector<unsigned char> compressedMin = readCompressedData(treeID + "-Min-compressedIndexData");
    std::vector<unsigned char> compressedMax = readCompressedData(treeID + "-Max-compressedIndexData");
    //std::vector<unsigned char> compressedMaxHigh = readCompressedData(treeID + "-MaxHigh-compressedIndexData");

    // **3. 读取 Type Size**
    std::vector<size_t> typeSizes = readTypeSizes(treeID + "-typeSize");

    // **4. Huffman 解码**
    std::vector<int> decompressedTypeMin = singleHuffmanDecodeZstdAVL(compressedMin, typeSizes[0], huffmanOutLow.data());
    std::vector<int> decompressedTypeMax = singleHuffmanDecodeZstdAVL(compressedMax, typeSizes[1], huffmanOutHigh.data());
    //std::vector<int> decompressedTypeMaxHigh = singleHuffmanDecodeZstdAVL(compressedMaxHigh, typeSizes[2], huffmanOutMaxHigh.data());


    /*std::cout << "decompressedTypeMin first 10 value: ";
    for (size_t i = 0; i < std::min(decompressedTypeMin.size(), size_t(10)); ++i) {
        std::cout << decompressedTypeMin[i] << " ";
    }
    std::cout << std::endl;*/

    // **5. 读取 OutLayer 数据**
    std::vector<T> outLayerMin = readOutLayerData<T>(treeID + "-OutMin.bin");
    std::vector<T> outLayerMax = readOutLayerData<T>(treeID + "-OutMax.bin");
    //std::vector<T> outLayerMaxHigh = readOutLayerData<T>(treeID + "-OutMaxHigh.bin");

    
    /*std::cout << "outLayerMin first 10: ";
    for (size_t i = 0; i < std::min(outLayerMin.size(), size_t(10)); ++i) {
        std::cout << outLayerMin[i] << " ";
    }
    std::cout << std::endl;*/


    // **6. 读取 bit-packed ID**
    std::vector<size_t> idOut = readNodeBitsIdFromFile(treeID + "-NodeID");


    //读取树结构
    std::vector<int> structureVec = loadTreeStructureFromByte(treeID);



    


      //解压leaf node的max_high
    //std::vector<T> decodedMaxHigh = decodeArrayType(decompressedTypeMaxHigh, error_bound, outLayerMaxHigh);

      //解压max
    std::vector<T> decodedMax = decodeArrayType(decompressedTypeMax, error_bound, outLayerMax);
    
    

    // **7. 重建 AVL 树**
    ScidxAVLNode<T>* reconstructMin = reconstructAndDequantizeOptimizedTree(decompressedTypeMin, structureVec, error_bound, outLayerMin,decompressedTypeMax,outLayerMax, treeID  );


    //std::vector<T> levelOrderDecompressedMin = getLevelOrderLowValues(reconstructMin);

    

    //根据bayes解压high(已经填充了outlayer)
    //std::vector<double> decodeFlattenedMax = decodeArrayMaxBayesOptimizedType(decompressedTypeMax, levelOrderDecompressedMin, error_bound, outLayerMax, treeID);


    // **8. 填充 high 值和id
    reconstructMax(reconstructMin, decodedMax, idOut);

    //填充 max_high 叶子值
    //reconstractLeafMaxHigh(reconstructMin, decodedMaxHigh);
    
    // **10. 更新整棵树的 max_high**
    //updateMaxHighOfTree(reconstructMin);

    //std::cout << "Decompression Completed: " << treeID << std::endl;

    
    return reconstructMin;
}






template<typename T>
void computeTypeBufferAVL(
    size_t k,
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleSubTree, 
    T error_bound, size_t i, adios2::Engine& engine, adios2::IO& io,
    std::vector<std::vector<int>>& allTreeTypesLow, 
    std::vector<std::vector<int>>& allTreeTypesHigh, 
    std::vector<std::vector<size_t>>& allTreeId,
    std::vector<std::vector<int>>& allTreeTypesMaxHigh
) {
    //std::vector<T> treeNodeMaxArray = flattenedTreeNodeMax(singleSubTree);
    std::vector<size_t> treeNodeIdArray = flattenedTreeNodeId(singleSubTree);

    std::vector<T> leafNodeMaxHighArray = getLeafNodeMaxHigh(singleSubTree);


    //获取low bin -> high bin 的映射
    std::vector<std::pair<T, T>> flattenedNodeLowHigh = flattenedTreeNodeLowHigh(singleSubTree);

    std::vector<int> bayesMaxType = computeArrayTypeBayesAVL(k, flattenedNodeLowHigh, error_bound, engine, io, i, "_h", "low_high_bin_mapping.csv");


    std::vector<int> leafNodeMaxHighType = computeArrayTypeAVL(k, leafNodeMaxHighArray, error_bound, engine, io, i, "_mh");
    std::vector<int> treeNodeMinType = preQuantiSingleTreeMin(k, singleSubTree, error_bound, engine, io, i, "_l");

    allTreeTypesHigh.push_back(bayesMaxType);
    allTreeId.push_back(treeNodeIdArray);
    allTreeTypesMaxHigh.push_back(leafNodeMaxHighType);
    allTreeTypesLow.push_back(treeNodeMinType);
}

//将outlayer数据写文件
template <typename T>
void writeOutLayerDataToFile(const std::string& treeID, 
                             const std::vector<T>& outLayerMinData, 
                             const std::vector<T>& outLayerMaxData, 
                             const std::vector<T>& outLayerMaxHighData) 
{
    // 构造文件名
    std::string minFileName = treeID + "-min.txt";
    std::string maxFileName = treeID + "-max.txt";
    std::string maxHighFileName = treeID + "-maxHigh.txt";

    // 写入 min 数据
    std::ofstream minFile(minFileName);
    if (minFile.is_open()) {
        for (const auto& value : outLayerMinData) {
            minFile << value << "\n";
        }
        minFile.close();
    } else {
        std::cerr << "Error opening file: " << minFileName << std::endl;
    }

    // 写入 max 数据
    std::ofstream maxFile(maxFileName);
    if (maxFile.is_open()) {
        for (const auto& value : outLayerMaxData) {
            maxFile << value << "\n";
        }
        maxFile.close();
    } else {
        std::cerr << "Error opening file: " << maxFileName << std::endl;
    }

    // 写入 maxHigh 数据
    std::ofstream maxHighFile(maxHighFileName);
    if (maxHighFile.is_open()) {
        for (const auto& value : outLayerMaxHighData) {
            maxHighFile << value << "\n";
        }
        maxHighFile.close();
    } else {
        std::cerr << "Error opening file: " << maxHighFileName << std::endl;
    }

    /*std::cout << "Files written successfully: " << minFileName << ", " 
              << maxFileName << ", " << maxHighFileName << std::endl;*/
}


template <typename T>
void writeOutLayerDataToFileOptimized(const std::string& treeID, 
                                   const std::vector<T>& outLayerMinData, 
                                   const std::vector<T>& outLayerMaxData, 
                                   const std::vector<T>& outLayerMaxHighData) 
{
    // for test
    //std::cout << "treeID: " << treeID << std::endl;
    //std::cout << "outLayerMinData size: " << outLayerMinData.size() << std::endl;

    auto writeBinary = [](const std::string& filename, const std::vector<T>& data) {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
            return;
        }

        // 先写入数据大小，方便后续读取
        size_t size = data.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

        // 写入数据本体
        outFile.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
        outFile.close();
    };

    writeBinary(treeID + "-OutMin.bin", outLayerMinData);
    writeBinary(treeID + "-OutMax.bin", outLayerMaxData);
    writeBinary(treeID + "-OutMaxHigh.bin", outLayerMaxHighData);

    //std::cout << "Binary outlayer data saved! " << std::endl;
}

template <typename T>
void writeOutLayerDataToFileOptimizedTest(const std::string& treeID, 
                                   const std::vector<T>& outLayerMinData, 
                                   const std::vector<T>& outLayerMaxData, const std::vector<T>& outLayerSkippedMinData, const std::vector<T>& outLayerSkippedMaxData) 
{
    // for test
    //std::cout << "treeID: " << treeID << std::endl;
    //std::cout << "outLayerMinData size: " << outLayerMinData.size() << std::endl;

    auto writeBinary = [](const std::string& filename, const std::vector<T>& data) {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
            return;
        }

        // 先写入数据大小，方便后续读取
        size_t size = data.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

        // 写入数据本体
        outFile.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
        outFile.close();
    };

    writeBinary(treeID + "-OutMin.bin", outLayerMinData);
    writeBinary(treeID + "-OutMax.bin", outLayerMaxData);

    writeBinary(treeID + "-OutSkippedMin.bin", outLayerSkippedMinData);
    writeBinary(treeID + "-OutSkippedMax.bin", outLayerSkippedMaxData);


    //writeBinary(treeID + "-OutMaxHigh.bin", outLayerMaxHighData);

    //std::cout << "Binary outlayer data saved! " << std::endl;
}





// 以二进制方式写入数据到文件
void writeCompressedDataToFile(const std::string& filename, const std::vector<unsigned char>& data) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char*>(data.data()), data.size());
    outFile.close();
}

void saveCompressedDataToFiles(const std::string& treeID,
                               const std::vector<unsigned char>& compressedMin,
                               const std::vector<unsigned char>& compressedMax,
                               const std::vector<unsigned char>& compressedMaxHigh) {
    std::string Minfilename = treeID + "-Min-compressedIndexData";
    std::string Maxfilename = treeID + "-Max-compressedIndexData";
    std::string MaxHighfilename = treeID + "-MaxHigh-compressedIndexData";

    writeCompressedDataToFile(Minfilename, compressedMin);
    writeCompressedDataToFile(Maxfilename, compressedMax);
    writeCompressedDataToFile(MaxHighfilename, compressedMaxHigh);

    /*std::cout << "Compressed data saved to files:\n"
              << Minfilename << "\n"
              << Maxfilename << "\n"
              << MaxHighfilename << std::endl;*/
}

void saveCompressedDataToFilesTest(const std::string& treeID,
    const std::vector<unsigned char>& compressedMin,
    const std::vector<unsigned char>& compressedMax, const std::vector<unsigned char>& compressedSkippedMin,const std::vector<unsigned char>& compressedSkippedMax ) {
std::string Minfilename = treeID + "-Min-compressedIndexData";
std::string Maxfilename = treeID + "-Max-compressedIndexData";
std::string skippedMinfilename = treeID + "-SkippedMin-compressedIndexData";
std::string skippedMaxfilename = treeID + "-SkippedMax-compressedIndexData";


writeCompressedDataToFile(Minfilename, compressedMin);
writeCompressedDataToFile(Maxfilename, compressedMax);

writeCompressedDataToFile(skippedMinfilename, compressedSkippedMin);
writeCompressedDataToFile(skippedMaxfilename, compressedSkippedMax);


/*std::cout << "Compressed data saved to files:\n"
<< Minfilename << "\n"
<< Maxfilename << std::endl;*/
}





// **Bit Packing（小端序压缩id）
std::pair<std::vector<uint8_t>, unsigned char> idConvertToBytesSmallEdian(const std::vector<size_t>& currentIds) {
    size_t maxId = *std::max_element(currentIds.begin(), currentIds.end());
    unsigned char bitsPerId = static_cast<unsigned char>(std::ceil(std::log2(maxId + 1)));

    size_t totalBits = bitsPerId * currentIds.size();
    size_t totalBytes = (totalBits + 7) / 8;
    std::vector<uint8_t> bytes(totalBytes, 0);

    size_t bitPos = 0;
    for (size_t id : currentIds) {
        for (size_t i = 0; i < bitsPerId; ++i) {
            size_t byteIndex = bitPos / 8;
            size_t bitIndex = bitPos % 8;
            bytes[byteIndex] |= ((id >> i) & 1) << bitIndex;
            ++bitPos;
        }
    }

    return {bytes, bitsPerId};
}



// 将 bit-packed ID 存入文件
void writeNodeBitsIdToFile(const std::string& filename, const std::vector<size_t>& nodeId) {
    auto [compressedBytes, bitsPerId] = idConvertToBytesSmallEdian(nodeId);

    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // 先写入 bitsPerId，确保解压时知道每个 ID 需要多少 bit
    outFile.write(reinterpret_cast<const char*>(&bitsPerId), sizeof(bitsPerId));

    // 写入 bit-packed ID 数据
    outFile.write(reinterpret_cast<const char*>(compressedBytes.data()), compressedBytes.size());

    outFile.close();
    //std::cout << "Node ID saved to " << filename << " (Compressed Size: " << compressedBytes.size() << " bytes)\n";
}

template<typename T>
void writeNodeBitsIdToFileByLevel(
    const std::string& filename,
    const std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree)
{
    size_t dimSize = 256;

    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    size_t numLevels = singleTree.size();
    outFile.write(reinterpret_cast<const char*>(&numLevels), sizeof(size_t));

    size_t total_bits_before = 0;
    size_t total_bits_after  = 0;

    for (size_t level = 0; level < singleTree.size(); ++level) {
        size_t levelSize = singleTree[level].size();
        outFile.write(reinterpret_cast<const char*>(&levelSize), sizeof(size_t));

        if (levelSize == 0) continue;

        // 收集当前层所有节点的3D坐标
        std::vector<uint8_t> xs, ys, zs;
        for (auto node : singleTree[level]) {
            size_t id = node->id;
            if (id == static_cast<size_t>(-1)) {
                xs.push_back(0); ys.push_back(0); zs.push_back(0);
                continue;
            }
            xs.push_back(id % dimSize);
            ys.push_back((id / dimSize) % dimSize);
            zs.push_back(id / (dimSize * dimSize));
        }

        // 对x, y, z分别做差值编码
        auto encodeDim = [&](const std::vector<uint8_t>& coords) {
            std::vector<uint8_t> zigzagDeltas;
            zigzagDeltas.push_back(coords[0]);
            for (size_t i = 1; i < coords.size(); i++) {
                int delta = (int)coords[i] - (int)coords[i-1];
                uint8_t zigzag = (uint8_t)((delta << 1) ^ (delta >> 7));
                zigzagDeltas.push_back(zigzag);
            }

            uint8_t maxVal = *std::max_element(zigzagDeltas.begin(), zigzagDeltas.end());
            unsigned char bitsPerVal = static_cast<unsigned char>(
                std::max(1, (int)std::ceil(std::log2(maxVal + 2))));

            size_t totalBits = bitsPerVal * zigzagDeltas.size();
            size_t totalBytes = (totalBits + 7) / 8;
            std::vector<uint8_t> bytes(totalBytes, 0);

            size_t bitPos = 0;
            for (uint8_t val : zigzagDeltas) {
                for (size_t i = 0; i < bitsPerVal; i++) {
                    size_t byteIndex = bitPos / 8;
                    size_t bitIndex  = bitPos % 8;
                    bytes[byteIndex] |= ((val >> i) & 1) << bitIndex;
                    ++bitPos;
                }
            }
            return std::make_pair(bytes, bitsPerVal);
        };

        auto [xBytes, xBits] = encodeDim(xs);
        auto [yBytes, yBits] = encodeDim(ys);
        auto [zBytes, zBits] = encodeDim(zs);

        total_bits_before += levelSize * 24;
        total_bits_after  += levelSize * (xBits + yBits + zBits);

        std::cout << "[Level " << level << "] "
                  << "xBits=" << (int)xBits
                  << " yBits=" << (int)yBits
                  << " zBits=" << (int)zBits
                  << " total=" << (int)(xBits + yBits + zBits)
                  << " (before=24)" << std::endl;

        // 写x, y, z各自的bitsPerVal和数据
        outFile.write(reinterpret_cast<const char*>(&xBits), sizeof(unsigned char));
        outFile.write(reinterpret_cast<const char*>(xBytes.data()), xBytes.size());
        outFile.write(reinterpret_cast<const char*>(&yBits), sizeof(unsigned char));
        outFile.write(reinterpret_cast<const char*>(yBytes.data()), yBytes.size());
        outFile.write(reinterpret_cast<const char*>(&zBits), sizeof(unsigned char));
        outFile.write(reinterpret_cast<const char*>(zBytes.data()), zBytes.size());
    }

    outFile.close();

    std::cout << "[ID 3D] total bits before=" << total_bits_before
              << " after=" << total_bits_after
              << " ratio=" << (double)total_bits_before / total_bits_after
              << std::endl;
}






template<typename T>
void compressNaiveAVL(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleTree, 
    T error_bound, 
    std::string treeID
) {

    //最naive的压缩方式，将id，min,max，max_high按层级结构转化成一维进行压缩
    auto [nodeMin, nodeMax, nodeMaxHigh, nodeId] = flattenedTreeAttributes(singleTree);

    //id不压缩直接写文件
    std::string Idfilename = treeID + "-NodeID";
    writeNodeIdToFile(Idfilename, nodeId);


    std::pair<std::vector<int>, std::vector<T>> typeMin = computeTypeForFlattenedArray(nodeMin, error_bound);
    std::pair<std::vector<int>, std::vector<T>> typeMax = computeTypeForFlattenedArray(nodeMax, error_bound);
    std::pair<std::vector<int>, std::vector<T>> typeMaxHigh = computeTypeForFlattenedArray(nodeMaxHigh, error_bound);


    std::vector<int> typeMinVector = typeMin.first;
    std::vector<T> outLayerMinData = typeMin.second;

    std::vector<int> typeMaxVector = typeMax.first;
    std::vector<T> outLayerMaxData = typeMax.second;

    std::vector<int> typeMaxHighVector = typeMaxHigh.first;
    std::vector<T> outLayerMaxHighData = typeMaxHigh.second;

    //写min,max,max_high的outlayer数据
    writeOutLayerDataToFile(treeID, outLayerMinData, outLayerMaxData, outLayerMaxHighData);

    std::string Minfilename = treeID + "-Min";
    std::string Maxfilename = treeID + "-Max";
    std::string MaxHighfilename = treeID + "-MaxHigh";


    HuffmanTree* huffmanTreeMin = processHuffmanTreeNaiveAVL(typeMinVector, Minfilename);
    HuffmanTree* huffmanTreeMax = processHuffmanTreeNaiveAVL(typeMaxVector, Maxfilename);
    HuffmanTree* huffmanTreeMaxHigh = processHuffmanTreeNaiveAVL(typeMaxHighVector, MaxHighfilename);

    //压缩前type的大小，解压后还原大小
    std::vector<size_t> typeSizes;
    typeSizes.push_back(typeMinVector.size());
    typeSizes.push_back(typeMaxVector.size());
    typeSizes.push_back(typeMaxHighVector.size());

    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(typeMinVector, huffmanTreeMin);

    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(typeMaxVector, huffmanTreeMax);

    std::vector<unsigned char> compressedMaxHigh = singleHuffmanEncodeZstdAVL(typeMaxHighVector, huffmanTreeMaxHigh);

    saveCompressedDataToFiles(treeID, compressedMin, compressedMax, compressedMaxHigh);   

}





//原始id的数据类型是size_t
template<typename T>
void compressTree(size_t k, std::vector<int> currentTypesLow, std::vector<int> currentTypesHigh, std::vector<size_t> currentIds, std::vector<int> currentTypesMaxHigh, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t step, scidx::HuffmanTree* fullHuffmanTreeLow, scidx::HuffmanTree* fullHuffmanTreeHigh, scidx::HuffmanTree* fullHuffmanTreeMaxHigh) {
    //压缩前type的大小，解压后还原大小
    std::vector<size_t> typeSizes;
    typeSizes.push_back(currentTypesHigh.size());
    typeSizes.push_back(currentTypesMaxHigh.size());
    typeSizes.push_back(currentTypesLow.size());

    std::vector<unsigned int> typeSizesUnsigned;
    for (size_t size : typeSizes) {
        typeSizesUnsigned.push_back(static_cast<unsigned int>(size));
    }

    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(currentTypesHigh, fullHuffmanTreeHigh);
    //std::cout << "before encode Max high "<< std::endl;
    std::vector<unsigned char> compressedLeafMaxHigh = singleHuffmanEncodeZstdAVL(currentTypesMaxHigh, fullHuffmanTreeMaxHigh);
    //std::cout << "after encode Max high " << std::endl;
    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(currentTypesLow, fullHuffmanTreeLow);

    std::string varNamePrefix = std::to_string(k) + "_" + std::to_string(step);

    auto [bytes, bitsPerId] = idConvertToBytes(currentIds);


    adios2::Variable<uint8_t> bpIdBytes = io.DefineVariable<uint8_t>(varNamePrefix + "_i", {bytes.size()}, {0}, {bytes.size()}, adios2::ConstantDims);
    adios2::Variable<unsigned char> bpBitsPerId = io.DefineVariable<unsigned char>(varNamePrefix + "_Bit_Id");

    adios2::Variable<unsigned char> bpCompressedMax = io.DefineVariable<unsigned char>(varNamePrefix + "_h", {compressedMax.size()}, {0}, {compressedMax.size()}, adios2::ConstantDims);
    adios2::Variable<unsigned char> bpCompressedLeafMaxHigh = io.DefineVariable<unsigned char>(varNamePrefix + "_mh", {compressedLeafMaxHigh.size()}, {0}, {compressedLeafMaxHigh.size()}, adios2::ConstantDims);
    adios2::Variable<unsigned char> bpCompressedMin = io.DefineVariable<unsigned char>(varNamePrefix + "_l", {compressedMin.size()}, {0}, {compressedMin.size()}, adios2::ConstantDims);
    adios2::Variable<unsigned int> bpTypeSizes = io.DefineVariable<unsigned int>(varNamePrefix + "_t", {typeSizesUnsigned.size()}, {0}, {typeSizesUnsigned.size()}, adios2::ConstantDims);
        
    /*std::cout << "Size of compressed Max: " << formatSize(compressedMax.size()) << std::endl;
    std::cout << "Size of compressed Leaf Max High: " << formatSize(compressedLeafMaxHigh.size()) << std::endl;
    std::cout << "Size of compressed Min: " << formatSize(compressedMin.size()) << std::endl;*/

    
    engine.Put(bpIdBytes, bytes.data(), adios2::Mode::Sync);
    engine.Put(bpBitsPerId, &bitsPerId, adios2::Mode::Sync);
    
    engine.Put(bpCompressedMax, compressedMax.data(), adios2::Mode::Sync);
    engine.Put(bpCompressedLeafMaxHigh, compressedLeafMaxHigh.data(), adios2::Mode::Sync);
    engine.Put(bpCompressedMin, compressedMin.data(), adios2::Mode::Sync);
    engine.Put(bpTypeSizes, typeSizesUnsigned.data(), adios2::Mode::Sync);
}



std::string formatSize(size_t bytes) {
    const size_t KB = 1024;
    const size_t MB = KB * 1024;
    const size_t GB = MB * 1024;

    if (bytes >= GB) {
        return std::to_string(static_cast<double>(bytes) / GB) + " GB";
    } else if (bytes >= MB) {
        return std::to_string(static_cast<double>(bytes) / MB) + " MB";
    } else if (bytes >= KB) {
        return std::to_string(static_cast<double>(bytes) / KB) + " KB";
    } else {
        return std::to_string(bytes) + " bytes";
    }
}

//将id根据范围分配规定的bits,然后依次8bits紧凑处理成一个byte
std::pair<std::vector<uint8_t>, unsigned char> idConvertToBytes(const std::vector<size_t>& currentIds) {
    size_t maxId = *max_element(currentIds.begin(), currentIds.end());
    unsigned char bitsPerId = static_cast<unsigned char>(std::ceil(std::log2(maxId + 1)));
     
    // 计算所需的总位数和字节数
    size_t totalBits = bitsPerId * currentIds.size();
    size_t totalBytes = (totalBits + 7) / 8; // 向上取整到最近的字节
    std::vector<uint8_t> bytes(totalBytes, 0);

    // 将 ID 的位打包到字节数组中
    size_t bitPos = 0;
    for (size_t id : currentIds) {
        for (size_t i = 0; i < bitsPerId; ++i) {
            size_t byteIndex = bitPos / 8;
            size_t bitIndex = bitPos % 8;

            // 插入当前位到相应的字节和位置
            bytes[byteIndex] |= ((id >> i) & 1) << bitIndex;

            // 如果当前位跨越了字节边界，需要处理下一字节的位
            if (bitIndex == 7 && i < bitsPerId - 1) {
                ++byteIndex;
                bytes[byteIndex] |= (id >> (i + 1)) & 1;
            }

            ++bitPos;
        }
    }

    return std::make_pair(bytes, bitsPerId);
}

template<typename T>
ScidxAVLNode<T>* decompressTreeAVL(size_t k, adios2::Engine& engine, adios2::IO& io, size_t step, std::vector<int>& firstVector, T error_bound, std::vector<unsigned char> huffmanOutLow, std::vector<unsigned char> huffmanOutHigh, std::vector<unsigned char> huffmanOutMaxHigh) {
    std::string varNamePrefix = std::to_string(k)+ "_" + std::to_string(step);

    //std::cout << "-shhah----" << std::endl;
    adios2::Variable<unsigned char> bpCompressedMax = io.InquireVariable<unsigned char>(varNamePrefix + "_h");
    adios2::Variable<unsigned char> bpCompressedLeafMaxHigh = io.InquireVariable<unsigned char>(varNamePrefix + "_mh");
    adios2::Variable<unsigned char> bpCompressedMin = io.InquireVariable<unsigned char>(varNamePrefix + "_l");
    adios2::Variable<unsigned int> bpTypeSizes = io.InquireVariable<unsigned int>(varNamePrefix + "_t");
    adios2::Variable<unsigned int> bpIds = io.InquireVariable<unsigned int>(varNamePrefix + "_i");

    std::vector<unsigned char> compressedMax(bpCompressedMax.Shape()[0]);
    std::vector<unsigned char> compressedLeafMaxHigh(bpCompressedLeafMaxHigh.Shape()[0]);
    std::vector<unsigned char> compressedMin(bpCompressedMin.Shape()[0]);
    std::vector<unsigned int> typeSizes(bpTypeSizes.Shape()[0]);
    std::vector<unsigned int> idOut(bpIds.Shape()[0]);
    
    engine.Get(bpCompressedMax, compressedMax);
    engine.Get(bpCompressedLeafMaxHigh, compressedLeafMaxHigh);
    engine.Get(bpCompressedMin, compressedMin);
    engine.Get(bpTypeSizes, typeSizes);
    engine.Get(bpIds, idOut);
   
    std::string varNameOutlayer = std::to_string(k)+ "_U_" + std::to_string(step);

    adios2::Variable<T> bpOutlayerHigh = io.InquireVariable<T>(varNameOutlayer + "_h");
    adios2::Variable<T> bpOutlayerMaxhigh = io.InquireVariable<T>(varNameOutlayer + "_mh");
    adios2::Variable<T> bpOutlayerLow = io.InquireVariable<T>(varNameOutlayer + "_l");

    size_t varSizeHigh = bpOutlayerHigh.Shape()[0];
    size_t varSizeMaxHigh = bpOutlayerMaxhigh.Shape()[0];
    size_t varSizeLow = bpOutlayerLow.Shape()[0];
    
    std::vector<T> outlayerHigh(varSizeHigh);
    std::vector<T> outlayerMaxHigh(varSizeMaxHigh);
    std::vector<T> outlayerLow(varSizeLow);

    engine.Get(bpOutlayerHigh, outlayerHigh.data(), adios2::Mode::Sync);
    engine.Get(bpOutlayerMaxhigh, outlayerMaxHigh.data(), adios2::Mode::Sync);
    engine.Get(bpOutlayerLow, outlayerLow.data(), adios2::Mode::Sync);


    std::vector<int> decompressedTypeMax = singleHuffmanDecodeZstdAVL(compressedMax, typeSizes[0], huffmanOutHigh.data());
    std::vector<int> decompressedTypeLeafMaxHigh = singleHuffmanDecodeZstdAVL(compressedLeafMaxHigh, typeSizes[1], huffmanOutMaxHigh.data());
    std::vector<int> decompressedTypeMin = singleHuffmanDecodeZstdAVL(compressedMin, typeSizes[2], huffmanOutLow.data());

    ScidxAVLNode<T>* reconstructMin = reconstructAndDequantizeTree(decompressedTypeMin, firstVector, error_bound, outlayerLow);



    //ScidxAVLNode<T>* reconstructMin =reconstructTreeBayesMax(reconstructMin, decompressedTypeMax, error_bound, outlayerHigh));
    
    
    std::vector<T> decodeFlattenedMax = decodeArrayType(decompressedTypeMax, error_bound, outlayerHigh);
    std::vector<T> decodeFlattenedLeafMaxHigh = decodeArrayType(decompressedTypeLeafMaxHigh, error_bound, outlayerMaxHigh);
    
    std::vector<size_t> idOutSizeT(idOut.begin(), idOut.end());
    reconstructMax(reconstructMin, decodeFlattenedMax, idOutSizeT);

    reconstractLeafMaxHigh(reconstructMin, decodeFlattenedLeafMaxHigh);

    updateMaxHighOfTree(reconstructMin);

    return reconstructMin;
}

template<typename T>
std::vector<T> flattenedTreeNodeMax(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree) {
    std::vector<T> flattenedNodeMax;

    for (const auto &row : singleSubTree) {
        for (const auto &node : row) {
            flattenedNodeMax.push_back(node->interval.high);
        }
    }

    return flattenedNodeMax;
}


template<typename T>
std::vector<std::pair<T, T>> flattenedTreeNodeLowHigh(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree) {
    std::vector<std::pair<T, T>> flattenedNodeLowHigh;

    for (const auto &row : singleSubTree) {
        for (const auto &node : row) {
            // 提取 low 和 high，存入 pair 中
            flattenedNodeLowHigh.push_back(std::make_pair(node->interval.low, node->interval.high));
        }
    }

    return flattenedNodeLowHigh;
}

// 读取 lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount 以及 low bin -> high bin 的映射
template<typename T>
std::tuple<T, T, T, T, int, int, std::vector<std::pair<int, T>>> loadBinMappingAndStats(const std::string& csvFile) {
    T lowMin, lowMax, highMin, highMax;
    int lowBinCount;
    int highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMap;

    std::ifstream file(csvFile);
    std::string line;

    // 读取统计信息：lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount
    if (file.is_open()) {
        // 读取表头
        std::getline(file, line);
        // 读取统计信息
        std::getline(file, line);
        std::stringstream ss(line);
        char comma;
        ss >> lowMin >> comma >> lowMax >> comma >> highMin >> comma >> highMax >> comma >> lowBinCount >> comma >> highBinCount;

        // 跳过映射表头
        std::getline(file, line);

        // 读取 low bin -> high bin 的映射
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

    return std::make_tuple(lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, lowHighBinMap);
}

// 找到 low 值所处的 bin
template<typename T>
int findLowBin(T lowValue, T lowMin, T lowMax, int lowBinCount) {
    return std::floor((lowValue - lowMin) / (lowMax - lowMin) * lowBinCount);
}

// 找到 high bin 对应的中值
template<typename T>
T getHighBinMidValue(T highBin, T highMin, T highMax, int highBinCount) {
    T binWidth = (highMax - highMin) / highBinCount;
    T highValue = highMin + highBin * binWidth + binWidth / 2.0;
    return highValue;
}

template<typename T>
auto computeHighBayesOptimizedAVLFixOrder(
    const std::vector<MinMaxPair<T>>& minMaxPairs,
    T error_bound,
    std::string treeID
) -> std::pair<std::vector<int>, std::vector<T>>
{
    // 从文件中读取 lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, 和 lowHighBinMapping
    T lowMin, lowMax, highMin, highMax;
    int lowBinCount, highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMapping;

    std::string csvFileName = treeID + "-low_high_bin_mapping.csv";
    std::tie(lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, lowHighBinMapping) = loadBinMappingAndStats<T>(csvFileName);

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

    for (size_t i = 0; i < minMaxPairs.size(); i++) {
        T curLow = minMaxPairs[i].reconstructedMin;
        T curHigh = minMaxPairs[i].originalMax;

        // 通过 low 值找到对应的 low bin
        int lowBin = findLowBin(curLow, lowMin, lowMax, lowBinCount);

        // 查找 low bin 对应的 high bin
        T highBin = lowHighBinMapping[lowBin].second;

        /*T binWidth = (highMax - highMin) / highBinCount;
        T predictedHigh = highMin + highBin * binWidth;*/

        T predictedHigh = getHighBinMidValue(highBin);

        // 计算预测 high 的值
        //T predictedHigh = getHighBinMidValue(highBin, highMin, highMax, highBinCount);

        // 计算预测误差
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

        /*if (i < 10) {
            std::cout << "[DEBUG] i = " << i << "\n";
            std::cout << "  reconstructedMin (decompressed Low): " << curLow << "\n";
            std::cout << "  type: " << type[i] << "\n";
            std::cout << "  predictedHigh: " << predictedHigh << "\n";
            std::cout << "  curHigh (original max): " << curHigh << "\n";
            std::cout << "  predAbsErr: " << predAbsErr << "\n";
            std::cout << "  checkRadius: " << checkRadius << "\n";
            std::cout << "  accepted? " << (predAbsErr < checkRadius ? "YES" : "NO") << "\n";
        }*/
        



    }

    return {type, outLayerData};
}





//optimized bayes计算（不分子树）
template<typename T>
auto computeHighBayesOptimizedAVL(const std::vector<std::pair<T, T>>& flattenedNodeLowHigh, T error_bound, std::string treeID ) -> std::pair<std::vector<int>, std::vector<T>> {
    // 从文件中读取 lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, 和 lowHighBinMapping
    T lowMin, lowMax, highMin, highMax;
    int lowBinCount, highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMapping;

    std::string csvFileName = treeID + "-low_high_bin_mapping.csv";
    std::tie(lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, lowHighBinMapping) = loadBinMappingAndStats<T>(csvFileName);

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;


    std::vector<int> type(flattenedNodeLowHigh.size(), 0);
    std::vector<T> outLayerData;

    for (size_t i = 0; i < flattenedNodeLowHigh.size(); i++) {
        T curLow = flattenedNodeLowHigh[i].first;
        T curHigh = flattenedNodeLowHigh[i].second;

        // 通过 low 值找到对应的 low bin
        int lowBin = findLowBin(curLow, lowMin, lowMax, lowBinCount);
        
        // 查找 low bin 对应的 high bin
        T highBin = lowHighBinMapping[lowBin].second ;

        /*// 如果找到了 high bin，计算其中值
        T predictedHigh;  // 如果没有映射，使用原 high
        if (highBin != -1) {
            predictedHigh = getHighBinMidValue(highBin, highMin, highMax, highBinCount);
        }*/

        T predictedHigh = getHighBinMidValue(highBin);


        // 计算预测误差
        T predAbsErr = std::abs(predictedHigh - curHigh );

        if (predAbsErr < checkRadius) {
            int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
            if (predictedHigh >= curHigh) {
                type[i] = intvRadius + state;
               
            } else {
                type[i] = intvRadius - state;
              
            }
        } else {
            type[i] = 0;
      
            outLayerData.push_back(curHigh);
        }
    }

    return {type, outLayerData};
}


//分子树的bayes计算
template<typename T>
std::vector<int> computeArrayTypeBayesAVL(size_t k, const std::vector<std::pair<T, T>>& flattenedNodeLowHigh, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName, const std::string& csvFile) {
    // 从文件中读取 lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, 和 lowHighBinMapping
    T lowMin, lowMax, highMin, highMax;
    int lowBinCount, highBinCount;
    std::vector<std::pair<int, T>> lowHighBinMapping;
    std::tie(lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount, lowHighBinMapping) = loadBinMappingAndStats<T>(csvFile);

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;

    std::string varNamePreOut = std::to_string(k) + "_U_" + std::to_string(i);

    std::vector<int> type(flattenedNodeLowHigh.size(), 0);
    std::vector<T> outLayerData;

    //T predData = flattenedNodeLowHigh[0].second;  // 初始 high 值作为预测值
    //outLayerData.push_back(predData);

    for (size_t i = 0; i < flattenedNodeLowHigh.size(); i++) {
        T curLow = flattenedNodeLowHigh[i].first;
        T curHigh = flattenedNodeLowHigh[i].second;

        // 通过 low 值找到对应的 low bin
        int lowBin = findLowBin(curLow, lowMin, lowMax, lowBinCount);
        
        // 查找 low bin 对应的 high bin
        T highBin = lowHighBinMapping[lowBin].second ;

        // 如果找到了 high bin，计算其中值
        T predictedHigh;  // 如果没有映射，使用原 high
        if (highBin != -1) {
            predictedHigh = getHighBinMidValue(highBin, highMin, highMax, highBinCount);
        }

        // 计算预测误差
        T predAbsErr = std::abs(predictedHigh - curHigh );

        if (predAbsErr < checkRadius) {
            int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
            if (predictedHigh >= curHigh) {
                type[i] = intvRadius + state;
                //predData = predData + state * interval;
            } else {
                type[i] = intvRadius - state;
                //predData = predData - state * interval;
            }
        } else {
            type[i] = 0;
            //predData = predictedHigh;
            outLayerData.push_back(curHigh);
        }
    }

    adios2::Variable<T> bpOutlayer = io.DefineVariable<T>(varNamePreOut + variableName, {outLayerData.size()}, {0}, {outLayerData.size()}, adios2::ConstantDims);
    engine.Put(bpOutlayer, outLayerData.data(), adios2::Mode::Sync);

    return type;
}



//将树的min,max,max_high, id都打平返回
template<typename T>
std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<size_t>> 
flattenedTreeAttributes(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree) {
    
    std::vector<T> flattenedNodeMin;
    std::vector<T> flattenedNodeMax;
    std::vector<T> flattenedNodeMaxHigh;
    std::vector<size_t> flattenedNodeId;

    for (const auto &row : singleSubTree) {
        for (const auto &node : row) {
            flattenedNodeMin.push_back(node->interval.low);       // 最小值
            flattenedNodeMax.push_back(node->interval.high);      // 最大值
            flattenedNodeMaxHigh.push_back(node->max_high); // max_high
            flattenedNodeId.push_back(node->id);         // id
        }
    }

    return {flattenedNodeMin, flattenedNodeMax, flattenedNodeMaxHigh, flattenedNodeId};
}








template<typename T>
std::vector<T> flattenedTreeNodeMaxHigh(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree) {
    std::vector<T> flattenedNodeMaxHigh;

    for (const auto &row : singleSubTree) {
        for (const auto &node : row) {
            flattenedNodeMaxHigh.push_back(node->max_high);
        }
    }

    return flattenedNodeMaxHigh;
}

template<typename T>
std::vector<T> flattenedTreeNodeMin(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree) {
    std::vector<T> flattenedNodeMin;

    for (const auto &row : singleSubTree) {
        for (const auto &node : row) {
            flattenedNodeMin.push_back(node->interval.low);
        }
    }

    return flattenedNodeMin;
}

template<typename T>
std::vector<T> getLeafNodeMaxHigh(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree) {
    std::vector<T> leafNodeMaxHigh;

    if (singleSubTree.empty()) return leafNodeMaxHigh;

    for (size_t i = 0; i < singleSubTree.size() - 1; ++i) {
        for (ScidxAVLNode<T>* node : singleSubTree[i]) {
            if (node->left == nullptr && node->right == nullptr) {
                leafNodeMaxHigh.push_back(node->max_high);
            }
        }
    }

    for (ScidxAVLNode<T>* node : singleSubTree.back()) {
        leafNodeMaxHigh.push_back(node->max_high);
    }

    return leafNodeMaxHigh;
}

/*template<typename T>
auto preQuantiOptimizedMinBayes(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree, T error_bound) -> std::pair<std::vector<int>, std::vector<T>>  {
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;

    std::vector<int> type;
    T curData, predData, firstNodePredData;
    std::vector<T> outLayerData;

    for (size_t level = 0; level < singleSubTree.size(); ++level) {
        for (size_t index = 0; index < singleSubTree[level].size(); ++index) {
            curData = singleSubTree[level][index]->interval.low;

            if (level == 0 && index == 0) {
                type.push_back(0);
                firstNodePredData = curData;
                outLayerData.push_back(curData);
            } else if (index == 0) {
                T predAbsErr = std::abs(curData - firstNodePredData);

                if (predAbsErr < checkRadius) {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= firstNodePredData) {
                        type.push_back(intvRadius + state);
                        predData = firstNodePredData + state * interval;
                        firstNodePredData = firstNodePredData + state * interval;
                    } else {
                        type.push_back(intvRadius - state);
                        predData = firstNodePredData - state * interval;
                        firstNodePredData = firstNodePredData - state * interval;
                    }
                } else {
                    type.push_back(0);
                    predData = curData;
                    firstNodePredData = curData;
                    outLayerData.push_back(curData);
                }
            } else {
                T predAbsErr = std::abs(curData - predData);

                if (predAbsErr < checkRadius) {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= predData) {
                        type.push_back(intvRadius + state);
                        predData = predData + state * interval;
                    } else {
                        type.push_back(intvRadius - state);
                        predData = predData - state * interval;
                    }
                } else {
                    type.push_back(0);
                    predData = curData;
                    outLayerData.push_back(curData);
                }
            }
        }
    }

    return {type, outLayerData};
}*/

template<typename T>
auto preQuantiOptimizedMinBayes(
    const std::vector<std::vector<ScidxAVLNode<T>*>>& singleSubTree,
    T error_bound
) -> std::tuple<std::vector<int>, std::vector<T>, std::vector<MinMaxPair<T>>>
{
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;

    std::vector<int> type;
    std::vector<T> outLayerData;
    std::vector<MinMaxPair<T>> minMaxPairs;

    T curData, predData = 0;
    T firstNodePredData = 0;
    T reconstructedMin = 0;

    for (size_t level = 0; level < singleSubTree.size(); ++level) {
        bool foundFirstValidInThisLevel = false;

        for (size_t index = 0; index < singleSubTree[level].size(); ++index) {
            ScidxAVLNode<T>* node = singleSubTree[level][index];
            curData = node->interval.low;

            if (level == 0 && index == 0) {
                // 根节点：直接存
                type.push_back(0);
                outLayerData.push_back(curData);
                reconstructedMin = curData;
                predData = curData;
                firstNodePredData = curData;
                foundFirstValidInThisLevel = true;
            }
            else if (!foundFirstValidInThisLevel) {
                // 每层第一个节点，用上一层第一个有效节点预测
                T predAbsErr = std::abs(curData - firstNodePredData);
                if (predAbsErr < checkRadius) {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= firstNodePredData) {
                        type.push_back(intvRadius + state);
                        reconstructedMin = firstNodePredData + state * interval;
                        predData = reconstructedMin;
                        firstNodePredData = reconstructedMin;
                    } else {
                        type.push_back(intvRadius - state);
                        reconstructedMin = firstNodePredData - state * interval;
                        predData = reconstructedMin;
                        firstNodePredData = reconstructedMin;
                    }
                } else {
                    type.push_back(0);
                    outLayerData.push_back(curData);
                    reconstructedMin = curData;
                    predData = curData;
                    firstNodePredData = curData;
                }
                foundFirstValidInThisLevel = true;
            }
            else {
                // 同层中非首个节点：用前一个节点预测
                T predAbsErr = std::abs(curData - predData);
                if (predAbsErr < checkRadius) {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= predData) {
                        type.push_back(intvRadius + state);
                        reconstructedMin = predData + state * interval;
                        predData = reconstructedMin;
                    } else {
                        type.push_back(intvRadius - state);
                        reconstructedMin = predData - state * interval;
                        predData = reconstructedMin;
                    }
                } else {
                    type.push_back(0);
                    outLayerData.push_back(curData);
                    reconstructedMin = curData;
                    predData = curData;
                }
            }

            // 记录可解压 min 和原始 max
            minMaxPairs.push_back({reconstructedMin, node->interval.high});
        }
    }

    return {type, outLayerData, minMaxPairs};
}

template<typename T>
auto preQuantiOptimizedMinBayesNew(
    const std::vector<std::vector<ScidxAVLNode<T>*>>& singleSubTree,
    T error_bound,
    std::string treeID
) -> std::tuple<std::vector<int>, std::vector<T>, std::vector<MinMaxPair<T>>>
{
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;

    std::vector<int> type;
    std::vector<T> outLayerData;
    std::vector<MinMaxPair<T>> minMaxPairs;

    T reconstructedMin = 0;

    // 存储每层的二阶拟合参数 a, b, c (y = a*x^2 + b*x + c)
    std::vector<T> levelA, levelB, levelC;

    // ============================================================
    // 第一遍：对每层做二阶最小二乘拟合
    // ============================================================
    for (size_t level = 0; level < singleSubTree.size(); ++level) {
        size_t n = singleSubTree[level].size();

        if (n == 0) {
            levelA.push_back(0); levelB.push_back(0); levelC.push_back(0);
            continue;
        }
        if (n == 1) {
            T val = singleSubTree[level][0]->interval.low;
            levelA.push_back(0); levelB.push_back(0); levelC.push_back(val);
            continue;
        }
        if (n == 2) {
            // 只有两个点，退化为一阶
            T x0 = 0, y0 = singleSubTree[level][0]->interval.low;
            T x1 = 1, y1 = singleSubTree[level][1]->interval.low;
            T b = y1 - y0;
            levelA.push_back(0); levelB.push_back(b); levelC.push_back(y0);
            continue;
        }

        // 二阶最小二乘：y = a*x^2 + b*x + c
        // 需要解3x3线性方程组
        // sum(1)   sum(x)   sum(x^2)   | sum(y)
        // sum(x)   sum(x^2) sum(x^3)   | sum(xy)
        // sum(x^2) sum(x^3) sum(x^4)   | sum(x^2*y)
        T s0=0, s1=0, s2=0, s3=0, s4=0;
        T t0=0, t1=0, t2=0;

        for (size_t idx = 0; idx < n; ++idx) {
            T x = static_cast<T>(idx);
            T y = singleSubTree[level][idx]->interval.low;
            T x2 = x * x;
            T x3 = x2 * x;
            T x4 = x3 * x;
            s0 += 1;
            s1 += x;
            s2 += x2;
            s3 += x3;
            s4 += x4;
            t0 += y;
            t1 += x * y;
            t2 += x2 * y;
        }

        // 用高斯消元解3x3方程组
        // [s0 s1 s2] [c]   [t0]
        // [s1 s2 s3] [b] = [t1]
        // [s2 s3 s4] [a]   [t2]
        T mat[3][4] = {
            {s0, s1, s2, t0},
            {s1, s2, s3, t1},
            {s2, s3, s4, t2}
        };

        // 高斯消元
        for (int col = 0; col < 3; ++col) {
            // 找主元
            int maxRow = col;
            for (int row = col+1; row < 3; ++row) {
                if (std::abs(mat[row][col]) > std::abs(mat[maxRow][col]))
                    maxRow = row;
            }
            for (int k = 0; k < 4; ++k)
                std::swap(mat[col][k], mat[maxRow][k]);

            if (std::abs(mat[col][col]) < 1e-12) continue;

            for (int row = col+1; row < 3; ++row) {
                T factor = mat[row][col] / mat[col][col];
                for (int k = col; k < 4; ++k)
                    mat[row][k] -= factor * mat[col][k];
            }
        }

        // 回代
        T sol[3] = {0, 0, 0};
        for (int row = 2; row >= 0; --row) {
            if (std::abs(mat[row][row]) < 1e-12) continue;
            sol[row] = mat[row][3];
            for (int k = row+1; k < 3; ++k)
                sol[row] -= mat[row][k] * sol[k];
            sol[row] /= mat[row][row];
        }

        // sol[0]=c, sol[1]=b, sol[2]=a
        levelC.push_back(sol[0]);
        levelB.push_back(sol[1]);
        levelA.push_back(sol[2]);
    }

    // 将拟合参数存入文件
    {
        std::string paramFile = treeID + "-levelFitParams";
        std::ofstream ofs(paramFile, std::ios::binary);
        size_t numLevels = levelA.size();
        ofs.write(reinterpret_cast<const char*>(&numLevels), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(levelA.data()), numLevels * sizeof(T));
        ofs.write(reinterpret_cast<const char*>(levelB.data()), numLevels * sizeof(T));
        ofs.write(reinterpret_cast<const char*>(levelC.data()), numLevels * sizeof(T));
        ofs.close();
    }

    // ============================================================
    // 第二遍：用二阶拟合参数预测，量化压缩
    // ============================================================
    for (size_t level = 0; level < singleSubTree.size(); ++level) {
        T a = levelA[level];
        T b = levelB[level];
        T c = levelC[level];

        for (size_t index = 0; index < singleSubTree[level].size(); ++index) {
            ScidxAVLNode<T>* node = singleSubTree[level][index];
            T curData = node->interval.low;

            if (level == 0 && index == 0) {
                type.push_back(0);
                outLayerData.push_back(curData);
                reconstructedMin = curData;
                minMaxPairs.push_back({reconstructedMin, node->interval.high});
                continue;
            }

            // 二阶预测
            T x = static_cast<T>(index);
            T predictedValue = a * x * x + b * x + c;

            T predAbsErr = std::abs(curData - predictedValue);
            if (predAbsErr < checkRadius) {
                int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                if (curData >= predictedValue) {
                    type.push_back(intvRadius + state);
                    reconstructedMin = predictedValue + state * interval;
                } else {
                    type.push_back(intvRadius - state);
                    reconstructedMin = predictedValue - state * interval;
                }
            } else {
                type.push_back(0);
                outLayerData.push_back(curData);
                reconstructedMin = curData;
            }

            minMaxPairs.push_back({reconstructedMin, node->interval.high});
        }
    }

    return {type, outLayerData, minMaxPairs};
}


template<typename T>
auto preQuantiOptimizedMin(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree, T error_bound) -> std::pair<std::vector<int>, std::vector<T>>  {
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;

    std::vector<int> type;
    T curData, predData, firstNodePredData;
    std::vector<T> outLayerData;

    for (size_t level = 0; level < singleSubTree.size(); ++level) {
        for (size_t index = 0; index < singleSubTree[level].size(); ++index) {
            curData = singleSubTree[level][index]->interval.low;

            if (level == 0 && index == 0) {
                type.push_back(0);
                firstNodePredData = curData;
                outLayerData.push_back(curData);
            } else if (index == 0) {
                T predAbsErr = std::abs(curData - firstNodePredData);

                if (predAbsErr < checkRadius) {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= firstNodePredData) {
                        type.push_back(intvRadius + state);
                        predData = firstNodePredData + state * interval;
                        firstNodePredData = firstNodePredData + state * interval;
                    } else {
                        type.push_back(intvRadius - state);
                        predData = firstNodePredData - state * interval;
                        firstNodePredData = firstNodePredData - state * interval;
                    }
                } else {
                    type.push_back(0);
                    predData = curData;
                    firstNodePredData = curData;
                    outLayerData.push_back(curData);
                }
            } else {
                T predAbsErr = std::abs(curData - predData);

                if (predAbsErr < checkRadius) {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= predData) {
                        type.push_back(intvRadius + state);
                        predData = predData + state * interval;
                    } else {
                        type.push_back(intvRadius - state);
                        predData = predData - state * interval;
                    }
                } else {
                    type.push_back(0);
                    predData = curData;
                    outLayerData.push_back(curData);
                }
            }
        }
    }

    return {type, outLayerData};
}




/*template<typename T>
auto preQuantiOptimizedMin(
    const std::vector<std::vector<ScidxAVLNode<T>*>>& singleSubTree,
    T error_bound
) -> std::tuple<std::vector<int>, std::vector<T>, std::vector<MinMaxPair<T>>>
{
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;

    std::vector<int> type;
    std::vector<T> outLayerData;
    std::vector<MinMaxPair<T>> minMaxPairs;

    T curData, predData = 0;
    T firstNodePredData = 0;
    T reconstructedMin = 0;

    for (size_t level = 0; level < singleSubTree.size(); ++level) {
        bool foundFirstValidInThisLevel = false;

        for (size_t index = 0; index < singleSubTree[level].size(); ++index) {
            ScidxAVLNode<T>* node = singleSubTree[level][index];
            curData = node->interval.low;

            if (level == 0 && index == 0) {
                // 根节点：直接存
                type.push_back(0);
                outLayerData.push_back(curData);
                reconstructedMin = curData;
                predData = curData;
                firstNodePredData = curData;
                foundFirstValidInThisLevel = true;
            }
            else if (!foundFirstValidInThisLevel) {
                // 每层第一个节点，用上一层第一个有效节点预测
                T predAbsErr = std::abs(curData - firstNodePredData);
                if (predAbsErr < checkRadius) {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= firstNodePredData) {
                        type.push_back(intvRadius + state);
                        reconstructedMin = firstNodePredData + state * interval;
                        predData = reconstructedMin;
                        firstNodePredData = reconstructedMin;
                    } else {
                        type.push_back(intvRadius - state);
                        reconstructedMin = firstNodePredData - state * interval;
                        predData = reconstructedMin;
                        firstNodePredData = reconstructedMin;
                    }
                } else {
                    type.push_back(0);
                    outLayerData.push_back(curData);
                    reconstructedMin = curData;
                    predData = curData;
                    firstNodePredData = curData;
                }
                foundFirstValidInThisLevel = true;
            }
            else {
                // 同层中非首个节点：用前一个节点预测
                T predAbsErr = std::abs(curData - predData);
                if (predAbsErr < checkRadius) {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= predData) {
                        type.push_back(intvRadius + state);
                        reconstructedMin = predData + state * interval;
                        predData = reconstructedMin;
                    } else {
                        type.push_back(intvRadius - state);
                        reconstructedMin = predData - state * interval;
                        predData = reconstructedMin;
                    }
                } else {
                    type.push_back(0);
                    outLayerData.push_back(curData);
                    reconstructedMin = curData;
                    predData = curData;
                }
            }

            // 记录可解压 min 和原始 max
            minMaxPairs.push_back({reconstructedMin, node->interval.high});
        }
    }

    return {type, outLayerData, minMaxPairs};
}*/



template<typename T>
std::vector<int> preQuantiSingleTreeMin(size_t k, std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName) {
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;

    std::vector<int> type;
    T curData, predData, firstNodePredData;
    std::vector<T> outLayerData;

    for (size_t level = 0; level < singleSubTree.size(); ++level) {
        for (size_t index = 0; index < singleSubTree[level].size(); ++index) {
            curData = singleSubTree[level][index]->interval.low;

            if (level == 0 && index == 0) {
                type.push_back(0);
                firstNodePredData = curData;
                outLayerData.push_back(curData);
            } else if (index == 0) {
                T predAbsErr = std::abs(curData - firstNodePredData);

                if (predAbsErr < checkRadius) {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= firstNodePredData) {
                        type.push_back(intvRadius + state);
                        predData = firstNodePredData + state * interval;
                        firstNodePredData = firstNodePredData + state * interval;
                    } else {
                        type.push_back(intvRadius - state);
                        predData = firstNodePredData - state * interval;
                        firstNodePredData = firstNodePredData - state * interval;
                    }
                } else {
                    type.push_back(0);
                    predData = curData;
                    firstNodePredData = curData;
                    outLayerData.push_back(curData);
                }
            } else {
                T predAbsErr = std::abs(curData - predData);

                if (predAbsErr < checkRadius) {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= predData) {
                        type.push_back(intvRadius + state);
                        predData = predData + state * interval;
                    } else {
                        type.push_back(intvRadius - state);
                        predData = predData - state * interval;
                    }
                } else {
                    type.push_back(0);
                    predData = curData;
                    outLayerData.push_back(curData);
                }
            }
        }
    }

    std::string varNamePre = std::to_string(k) + "_U_" + std::to_string(i);

    adios2::Variable<T> bpOutlayer = io.DefineVariable<T>(varNamePre + variableName, {outLayerData.size()}, {0}, {outLayerData.size()}, adios2::ConstantDims);
    engine.Put(bpOutlayer, outLayerData.data(), adios2::Mode::Sync);
   
    return type;
}


//最naive的预测（pre+quanti)
template<typename T>
auto computeTypeForFlattenedArray(
    const std::vector<T>& flattenedNode, 
    T error_bound
) -> std::pair<std::vector<int>, std::vector<T>> 
{
    int quantization_intervals = 16384;  // 量化间隔
    int intvRadius = quantization_intervals / 2;  // 量化中心
    T checkRadius = (quantization_intervals - 1) * error_bound;  // 误差检查范围
    T interval = 2 * error_bound;  // 量化步长
    T recip_precision = 1 / error_bound;  // 计算量化索引

    std::vector<int> type;  // 存储量化索引
    std::vector<T> outLayerData;  // 存储越界数据
    T predData = 0; // 预测值，初始化为 0

    for (size_t i = 0; i < flattenedNode.size(); ++i) {
        T curData = flattenedNode[i];  // 当前数据点

        if (i == 0) {
            // 第一个数据直接存储
            type.push_back(0);
            predData = curData;
            outLayerData.push_back(curData);
        } else {
            T predAbsErr = std::abs(curData - predData);  // 计算误差

            if (predAbsErr < checkRadius) {
                int state = ((int)(predAbsErr * recip_precision + 1)) >> 1; // 计算量化索引
            

                if (curData >= predData) {
                    type.push_back(intvRadius + state);
                    predData = predData + state * interval;
                } else {
                    type.push_back(intvRadius - state);
                    predData = predData - state * interval;
                }
            } else {
                type.push_back(0);
                predData = curData;
                outLayerData.push_back(curData);
            }
        }
    }

    return {type, outLayerData};
}







// 对压缩结果进行 zstd 解压和 Huffman 的 decode，还原到 type
std::vector<int> singleHuffmanDecodeZstdAVL(std::vector<unsigned char> compressedByZstdData, size_t typeSize, unsigned char *s) {

    // zstd 的 decompress，compressedByZstdData 压缩后的数据
    std::vector<unsigned char> decompressedData = decompressWithZstdAVL(compressedByZstdData, typeSize * 4);

    // 解压后单子树大小，即与原始数据等长
    std::vector<int> decodedData(typeSize);

    // 使用全 Huffman tree，对单子树的 encode 值，进行 decode
    decode_withSubTreeAVL(s, decompressedData.data(), decodedData.size(), decodedData.data());

    // 其中 data 是实际的解压数据，size 是 type.size
    return decodedData;
}

// s 为序列化后的 HuffmanTree，encode 为需要解码的数据
void decode_withSubTreeAVL(unsigned char *s, unsigned char *encode, size_t targetLength, int *out) {
    int stateNum = 2 * 16384;
    HuffmanTree *decodeHuffmanTree = createHuffmanTree(stateNum);

    size_t nodeCount = bytesToInt_bigEndian(s);
    node root = reconstruct_HuffTree_from_bytes_anyStates(decodeHuffmanTree, s + 8, nodeCount);

    decode(encode, targetLength, root, out);
}

// 对 type 进行 Huffman encode 和 zstd 压缩
std::vector<unsigned char> singleHuffmanEncodeZstdAVL(std::vector<int> type, HuffmanTree *huffmanTree) {
    // 单个需要 encode 的子树
    std::vector<unsigned char> singleEncodeOut(type.size() * 4);
    size_t singleEncodeOutsize = 0;
   
    encode(huffmanTree, type.data(), type.size(), singleEncodeOut.data(), &singleEncodeOutsize);

    auto zstdBegin_time = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> compressedByZstdData = compressWithZstdAVL(singleEncodeOut, singleEncodeOutsize);
    
    auto zstdEnd_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> zstd_time = zstdEnd_time - zstdBegin_time;
    std::cout << "[Time new] zstd_time: " << zstd_time.count() << " seconds" << std::endl;



    return compressedByZstdData;
}


std::vector<unsigned char> compressWithZstdAVL(const std::vector<unsigned char> &data, size_t dataLength)
{
    // 默认的压缩级别为 3
    int level = 3;

    // 预估压缩后的大小，和sz保持一致
    size_t estimatedCompressedSize = (dataLength < 100) ? 200 : static_cast<size_t>(dataLength * 1.2);

    // 分配内存用于存储压缩后的数据
    std::vector<unsigned char> compressBytes(estimatedCompressedSize);

    // 使用 Zstandard 压缩算法进行压缩
    size_t compressedSize = ZSTD_compress(compressBytes.data(), estimatedCompressedSize,
                                          data.data(), dataLength, level);

   

    // 调整压缩后的数据大小
    compressBytes.resize(compressedSize);

    return compressBytes;
}

std::vector<unsigned char> decompressWithZstdAVL(const std::vector<unsigned char> &compressedData, uint64_t targetOriSize) {
    // 分配内存用于存储解压后的数据
    std::vector<unsigned char> oriData(targetOriSize);

    // 使用 Zstandard 解压缩算法进行解压
    size_t outSize = ZSTD_decompress(oriData.data(), targetOriSize, compressedData.data(), compressedData.size());

    // 调整解压后的数据大小
    oriData.resize(outSize);

    return oriData;
}

template<typename T>
std::vector<int> computeArrayTypeAVL(size_t k, const std::vector<T> &flattenedTreeNodeValue, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName) {
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;

    std::string varNamePreOut = std::to_string(k) +"_U_" + std::to_string(i);

    std::vector<int> type(flattenedTreeNodeValue.size(), 0);
    std::vector<T> outLayerData;

    T predData = flattenedTreeNodeValue[0];
    outLayerData.push_back(flattenedTreeNodeValue[0]);
    for (size_t i = 1; i < flattenedTreeNodeValue.size(); i++) {
        T curData = flattenedTreeNodeValue[i];
        T predAbsErr = std::abs(curData - predData);

        if (predAbsErr < checkRadius) {
            int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
            if (curData >= predData) {
                type[i] = intvRadius + state;
                predData = predData + state * interval;
            } else {
                type[i] = intvRadius - state;
                predData = predData - state * interval;
            }
        } else {
            type[i] = 0;
            predData = curData;
            outLayerData.push_back(curData);
        }
    }

    adios2::Variable<T> bpOutlayer = io.DefineVariable<T>(varNamePreOut + variableName, {outLayerData.size()}, {0}, {outLayerData.size()}, adios2::ConstantDims);
    engine.Put(bpOutlayer, outLayerData.data(), adios2::Mode::Sync);

    /*std::cout << "outLayerData size: " << outLayerData.size() << std::endl;

    std::cout << "outLayerData content: ";
    for (const T& value : outLayerData) {
        std::cout << value << " ";
    }


    std::cout << std::endl;*/

    return type;
}

//从type还原到原始值
template<typename T>
std::vector<T> decodeArrayType(const std::vector<int>& type, T error_bound, std::vector<T> outlayerData) {
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T interval = 2 * error_bound;

    std::vector<T> decodedValues;
    T predData = 0.0;
    size_t outlayerIndex = 0;

    for (size_t i = 0; i < type.size(); ++i) {
        if (type[i] == 0) {
            if (outlayerIndex >= outlayerData.size()) {
                throw std::runtime_error("outlayerdata index out of range");
            }
            T curData = outlayerData[outlayerIndex];
            outlayerIndex++;
            decodedValues.push_back(curData);
            predData = curData;
        } else {
            int state = std::abs(type[i] - intvRadius);
            T adjustment = state * interval;
            T curData = (type[i] >= intvRadius) ? predData + adjustment : predData - adjustment;

            decodedValues.push_back(curData);
            predData = curData;
        }
    }

    return decodedValues;
}



template<typename T>
std::vector<T> decodeArrayMaxBayesType(const std::vector<int>& type, T error_bound, std::vector<T> outlayerData) {
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T interval = 2 * error_bound;

    std::vector<T> decodedValues;
    T predData = 0.0;
    size_t outlayerIndex = 0;

    for (size_t i = 0; i < type.size(); ++i) {
        if (type[i] == 0) {
            if (outlayerIndex >= outlayerData.size()) {
                throw std::runtime_error("outlayerdata index out of range");
            }
            T curData = outlayerData[outlayerIndex];
            outlayerIndex++;
            decodedValues.push_back(curData);
            predData = curData;
        } else {
            int state = std::abs(type[i] - intvRadius);
            T adjustment = state * interval;
            T curData = (type[i] >= intvRadius) ? predData + adjustment : predData - adjustment;

            decodedValues.push_back(curData);
            predData = curData;
        }
    }

    return decodedValues;
}


template<typename T>
T dequantizeValue(int quantizedValue, T error_bound, int intvRadius, T interval, T& baseValue) {
    if (quantizedValue == 0) {
        return baseValue;
    }
    int state = std::abs(quantizedValue - intvRadius);
    T adjustment = state * interval;
    return (quantizedValue >= intvRadius) ? baseValue + adjustment : baseValue - adjustment;
}

template<typename T>
ScidxAVLNode<T>* reconstructAndDequantizeTree(const std::vector<int>& type, const std::vector<int>& structureVec, T error_bound, std::vector<T> outlayerLow) {
    if (type.empty() || structureVec.empty()) return nullptr;

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T interval = 2 * error_bound;

    if (outlayerLow.empty()) {
        std::cerr << "outlayerLow is empty\n";
        return nullptr;
    }

    T rootValue = outlayerLow[0];
    size_t outlayerIndex = 1;

    ScidxAVLNode<T>* root = new ScidxAVLNode<T>({ScidxInterval<T>{rootValue, rootValue}, 0});
    std::queue<ScidxAVLNode<T>**> nodesQueue;
    nodesQueue.push(&root);

    size_t typeIndex = 0;
    size_t structIndex = 0;
    T baseValue = rootValue;

    std::queue<T> baseValuesForNextLevel;
    int currentLevelNodeCount = 1;
    int processedNodeCount = 0;
    bool isLevelFirstNode = true;
    int readIndex = 0;

    while (!nodesQueue.empty() && structIndex < structureVec.size()) {
        ScidxAVLNode<T>** currentNodePtr = nodesQueue.front();
        nodesQueue.pop();
        
        if (isLevelFirstNode && !baseValuesForNextLevel.empty()) {
            baseValue = baseValuesForNextLevel.front();
            baseValuesForNextLevel.pop();
        }

        T dequantizedValue;
        if (structureVec[structIndex] == 1) {
            if (typeIndex != 0 && type[typeIndex] == 0) {
                if (outlayerIndex >= outlayerLow.size()) {
                    std::cerr << "outlayerLow index out of range\n";
                    return nullptr;
                }
                dequantizedValue = outlayerLow[outlayerIndex];
                outlayerIndex++;
                typeIndex++;
            } else {
                dequantizedValue = dequantizeValue(type[typeIndex], error_bound, intvRadius, interval, baseValue);
                typeIndex++;
            }
            *currentNodePtr = new ScidxAVLNode<T>({ScidxInterval<T>{dequantizedValue, dequantizedValue}, 0});
            if (isLevelFirstNode) {
                baseValuesForNextLevel.push(dequantizedValue);
                isLevelFirstNode = false;
            }
            baseValue = dequantizedValue;
            nodesQueue.push(&((*currentNodePtr)->left));
            nodesQueue.push(&((*currentNodePtr)->right));
        }

        if (++processedNodeCount == currentLevelNodeCount && !nodesQueue.empty()) {
            currentLevelNodeCount = nodesQueue.size();
            processedNodeCount = 0;
            isLevelFirstNode = true;
        }

        structIndex++;
    }

    return root;
}

/*template<typename T>
ScidxAVLNode<T>* reconstructTreeBayesMax(ScidxAVLNode<T>* reconstructMin, const std::vector<int>& maxtype, T error_bound, std::vector<T> outlayerHigh) {
    if (!root) return;

    std::queue<ScidxAVLNode<T>*> queue;
    queue.push(root);

    size_t index = 0;

    while (!queue.empty()) {
        ScidxAVLNode<T>* current = queue.front();
        queue.pop();

        // 先填入 high 值和 id，然后递增索引
        current->interval.high = decodeFlattenedMax[index];
        current->id = static_cast<size_t>(idOut[index]);

        // 打印当前节点的 id
        std::cout << "id: " << current->id << std::endl;

        index++; // 在使用后递增索引

        if (current->left != nullptr) {
            queue.push(current->left);
        }

        if (current->right != nullptr) {
            queue.push(current->right);
        }
    }
}*/

template <typename T>
void reconstructMax(ScidxAVLNode<T>* root, const std::vector<T>& decodeFlattenedMax, const std::vector<size_t>& idOut){
    
    if (!root) return;

    std::queue<ScidxAVLNode<T>*> queue;
    queue.push(root);

    size_t index = 0;

    while (!queue.empty() && index < decodeFlattenedMax.size() && index < idOut.size()) {
        ScidxAVLNode<T>* current = queue.front();
        queue.pop();

        // 先填入 high 值和 id，然后递增索引
        current->interval.high = decodeFlattenedMax[index];
        current->id = idOut[index];

        // 打印当前节点的 id
        //std::cout << "id: " << current->id << std::endl;

        index++; // 在使用后递增索引

        if (current->left != nullptr) {
            queue.push(current->left);
        }

        if (current->right != nullptr) {
            queue.push(current->right);
        }
    }
}


template<typename T>
void reconstractLeafMaxHigh(ScidxAVLNode<T>* root, const std::vector<T>& decodeFlattenedLeafMaxHigh) {
    if (!root) return;

    std::queue<ScidxAVLNode<T>*> nodesQueue;
    nodesQueue.push(root);
    size_t valueIndex = 0;

    bool lastLayerStarted = false;
    std::queue<ScidxAVLNode<T>*> nextLayerNodes;
    nextLayerNodes.push(root);

    while (!nodesQueue.empty()) {
        size_t layerSize = nodesQueue.size();
        lastLayerStarted = nextLayerNodes.empty();

        while (layerSize-- > 0) {
            ScidxAVLNode<T>* currentNode = nodesQueue.front();
            nodesQueue.pop();

            if ((currentNode->left == nullptr && currentNode->right == nullptr && !lastLayerStarted) || lastLayerStarted) {
                if (valueIndex < decodeFlattenedLeafMaxHigh.size()) {
                    currentNode->max_high = decodeFlattenedLeafMaxHigh[valueIndex++];
                } else {
                    std::cerr << "Error: Not enough values in decodeFlattenedLeafMaxHigh to update the node." << std::endl;
                    return;
                }
            }

            if (currentNode->left) {
                nodesQueue.push(currentNode->left);
                nextLayerNodes.push(currentNode->left);
            }
            if (currentNode->right) {
                nodesQueue.push(currentNode->right);
                nextLayerNodes.push(currentNode->right);
            }
        }

        if (nextLayerNodes.size() == nodesQueue.size()) {
            nextLayerNodes = std::queue<ScidxAVLNode<T>*>();
        }
    }

    if (valueIndex < decodeFlattenedLeafMaxHigh.size()) {
        std::cerr << "Warning: Not all values in decodeFlattenedLeafMaxHigh were used." << std::endl;
    }
}

template<typename T>
void updateMaxHighOfTree(ScidxAVLNode<T>* node) {
    if (node == nullptr) {
        return;
    }

    updateMaxHighOfTree(node->left);
    updateMaxHighOfTree(node->right);

    T maxHigh = node->interval.high;

    if (node->left == nullptr && node->right == nullptr) {
        maxHigh = std::max(maxHigh, node->max_high);
    }
    else {
        if (node->left != nullptr) {
            maxHigh = std::max(maxHigh, node->left->max_high);
        }
        if (node->right != nullptr) {
            maxHigh = std::max(maxHigh, node->right->max_high);
        }
    }

    node->max_high = maxHigh;
}

// 显式实例化模板函数
template void compressTree<double>(size_t k, std::vector<int>, std::vector<int>, std::vector<size_t>, std::vector<int>, double, adios2::Engine&, adios2::IO&, size_t, scidx::HuffmanTree*, scidx::HuffmanTree*, scidx::HuffmanTree*);
template ScidxAVLNode<double>* decompressTreeAVL<double>(size_t k, adios2::Engine&, adios2::IO&, size_t, std::vector<int>&, double, std::vector<unsigned char>, std::vector<unsigned char>, std::vector<unsigned char>);
template void computeTypeBufferAVL<double>(size_t k, std::vector<std::vector<ScidxAVLNode<double>*>>&, double, size_t, adios2::Engine&, adios2::IO&, std::vector<std::vector<int>>&, std::vector<std::vector<int>>&, std::vector<std::vector<size_t>>&, std::vector<std::vector<int>>&);

template void compressNaiveAVL<double>(std::vector<std::vector<ScidxAVLNode<double>*>>&, double, std::string);

template void computeOptimizedAVL<double>(
    std::vector<std::vector<ScidxAVLNode<double>*>>&,
    double,
    std::string, std::vector<SkippedNode<double>> 
);


template void computeOptimizedAVL<float>(
    std::vector<std::vector<ScidxAVLNode<float>*>>&,
    float,
    std::string,
    std::vector<SkippedNode<float>>
);

template void computeOptimizedAVLNew<double>(
    std::vector<std::vector<ScidxAVLNode<double>*>>&,
    double,
    std::string,
    std::vector<SkippedNode<double>>,
    std::vector<double>*
);

template void computeOptimizedAVLNew<float>(
    std::vector<std::vector<ScidxAVLNode<float>*>>&,
    float,
    std::string,
    std::vector<SkippedNode<float>>,
    std::vector<float>*
);

template void computeOptimizedAVLForStaggerWithCrossPrediction(
    std::vector<std::vector<ScidxAVLNode<double>*>>&,
    double,
    std::string,
    std::vector<SkippedNode<double>>,
    const std::vector<double>&,
    const std::vector<size_t>&,
    const std::vector<size_t>&,
    size_t
); 

template void computeOptimizedAVLForStaggerWithCrossPrediction(
    std::vector<std::vector<ScidxAVLNode<float>*>>&,
    float,
    std::string,
    std::vector<SkippedNode<float>>,
    const std::vector<float>&,
    const std::vector<size_t>&,
    const std::vector<size_t>&,
    size_t
); 


template void computeOptimizedSZ3<double>(
    std::vector<std::vector<ScidxAVLNode<double>*>>&,
    double,
    std::string, std::vector<SkippedNode<double>> 
);

template void computeOptimizedSZ3<float>(
    std::vector<std::vector<ScidxAVLNode<float>*>>&,
    float,
    std::string,
    std::vector<SkippedNode<float>>
);



template void computeOptimizedTestAVL<double>(
    std::vector<std::vector<ScidxAVLNode<double>*>>&,
    double,
    std::string
);

template std::tuple<ScidxAVLNode<double>*, std::vector<double>, std::vector<double>, std::vector<size_t>> decompressOptimizedAVL(const std::string& treeID ,  double error_bound);


template std::tuple<ScidxAVLNode<float>*, std::vector<float>, std::vector<float>, std::vector<size_t>> decompressOptimizedAVL(const std::string& treeID ,  float error_bound);

template ScidxAVLNode<double>* decompressOptimizedTestAVL(const std::string& treeID ,  double error_bound);




// 显式实例化模板函数
template void compressTree<float>(size_t k, std::vector<int>, std::vector<int>, std::vector<size_t>, std::vector<int>, float, adios2::Engine&, adios2::IO&, size_t, scidx::HuffmanTree*, scidx::HuffmanTree*, scidx::HuffmanTree*);
template ScidxAVLNode<float>* decompressTreeAVL<float>(size_t k, adios2::Engine&, adios2::IO&, size_t, std::vector<int>&, float, std::vector<unsigned char>, std::vector<unsigned char>, std::vector<unsigned char>);
template void computeTypeBufferAVL<float>(size_t k, std::vector<std::vector<ScidxAVLNode<float>*>>&, float, size_t, adios2::Engine&, adios2::IO&, std::vector<std::vector<int>>&, std::vector<std::vector<int>>&, std::vector<std::vector<size_t>>&, std::vector<std::vector<int>>&);


template std::vector<int> computeArrayTypeBayesAVL<float>(
    size_t, const std::vector<std::pair<float, float>>&, float, 
    adios2::Engine&, adios2::IO&, size_t, 
    const std::string&, const std::string&
);

template std::vector<int> computeArrayTypeBayesAVL<double>(
    size_t, const std::vector<std::pair<double, double>>&, double, 
    adios2::Engine&, adios2::IO&, size_t, 
    const std::string&, const std::string&
);

template std::pair<std::vector<int>, std::vector<float>> computeTypeForFlattenedArray<float>(
    const std::vector<float>&, float);

template std::pair<std::vector<int>, std::vector<double>> computeTypeForFlattenedArray<double>(
    const std::vector<double>&, double);


    template void computeOptimizedZFP<float>(
        std::vector<std::vector<ScidxAVLNode<float>*>>&,
        float,
        const std::string&,
        std::vector<SkippedNode<float>>
    );
    
    template void computeOptimizedZFP<double>(
        std::vector<std::vector<ScidxAVLNode<double>*>>&,
        double,
        const std::string&,
        std::vector<SkippedNode<double>>
    );
        





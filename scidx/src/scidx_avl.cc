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

using namespace scidx;

template<typename T>
void compressTree(std::vector<int> currentTypesLow, std::vector<int> currentTypesHigh, std::vector<size_t> currentIds, std::vector<int> currentTypesMaxHigh, T error_bound, adios2::Engine& engine, adios2::IO& io, int step, scidx::HuffmanTree* fullHuffmanTreeLow, scidx::HuffmanTree* fullHuffmanTreeHigh, scidx::HuffmanTree* fullHuffmanTreeMaxHigh);

template<typename T>
ScidxAVLNode<T>* decompressTreeAVL(adios2::Engine& engine, adios2::IO& io, int step, std::vector<int>& firstVector, T error_bound, std::vector<unsigned char> huffmanOut);

template<typename T>
std::vector<T> flattenedTreeNodeMax(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree);

template<typename T>
std::vector<T> flattenedTreeNodeMaxHigh(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree);

template<typename T>
std::vector<T> getLeafNodeMaxHigh(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree);

template<typename T>
std::vector<int> preQuantiSingleTreeMin(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName);

template<typename T>
std::vector<int> computeArrayTypeAVL(const std::vector<T> &flattenedTreeNodeValue, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName);

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

template<typename T>
void reconstructMax(ScidxAVLNode<T>* root, const std::vector<T>& decodeFlattenedMax);

template<typename T>
void reconstractLeafMaxHigh(ScidxAVLNode<T>* root, const std::vector<T>& decodeFlattenedLeafMaxHigh);

template<typename T>
void updateMaxHighOfTree(ScidxAVLNode<T>* node);

template<typename T>
void computeTypeBufferAVL(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleSubTree, 
    T error_bound, size_t i, adios2::Engine& engine, adios2::IO& io,
    std::vector<std::vector<int>>& allTreeTypesLow, 
    std::vector<std::vector<int>>& allTreeTypesHigh,
    std::vector<std::vector<size_t>>& allTreeId, 
    std::vector<std::vector<int>>& allTreeTypesMaxHigh
);

std::vector<scidx::HuffmanTree*> fullHuffmanAVL(const std::vector<std::vector<int>>& allTreeTypesLow,
                                          const std::vector<std::vector<int>>& allTreeTypesHigh,
                                          const std::vector<std::vector<int>>& allTreeTypesMaxHigh,
                                          adios2::Engine& engine, adios2::IO& io);

HuffmanTree* processHuffmanTreeAVL(const std::vector<std::vector<int>>& allTreeTypes, adios2::Engine& engine, adios2::IO& io, const std::string& variableName);

std::vector<scidx::HuffmanTree*> fullHuffmanAVL(const std::vector<std::vector<int>>& allTreeTypesLow,
                                          const std::vector<std::vector<int>>& allTreeTypesHigh,
                                          const std::vector<std::vector<int>>& allTreeTypesMaxHigh,
                                          adios2::Engine& engine, adios2::IO& io) {

    std::vector<HuffmanTree*> huffmanTrees;

    // 处理并写入不同的 Huffman 树
    HuffmanTree* huffmanTreeLow = processHuffmanTreeAVL(allTreeTypesLow, engine, io, "fullHuffmanTreeLow");
    huffmanTrees.push_back(huffmanTreeLow);

    HuffmanTree* huffmanTreeHigh = processHuffmanTreeAVL(allTreeTypesHigh, engine, io, "fullHuffmanTreeHigh");
    huffmanTrees.push_back(huffmanTreeHigh);

    HuffmanTree* huffmanTreeMaxHigh = processHuffmanTreeAVL(allTreeTypesMaxHigh, engine, io, "fullHuffmanTreeMaxHigh");
    huffmanTrees.push_back(huffmanTreeMaxHigh);

    return huffmanTrees;
}

HuffmanTree* processHuffmanTreeAVL(const std::vector<std::vector<int>>& allTreeTypes, adios2::Engine& engine, adios2::IO& io, const std::string& variableName) {
    std::vector<int> allTypes;
    for (const auto& type : allTreeTypes) {
        allTypes.insert(allTypes.end(), type.begin(), type.end());
    }

    int stateNum = 2 * 16384;
    HuffmanTree *huffmanTreeFull = createHuffmanTree(stateNum);

    unsigned char *huffmanOut = nullptr;
    size_t huffmanOutSize = 0;
    init_and_serialize_Huffmantree(huffmanTreeFull, allTypes.data(), allTypes.size(), &huffmanOut, &huffmanOutSize);

    /*std::cout << "huffmanOut content for " << variableName << ":" << std::endl;
    for (size_t i = 0; i < huffmanOutSize; ++i) {
        std::cout << +huffmanOut[i] << " ";
    }
    std::cout << std::endl;*/

    // Ensure huffmanOut and huffmanOutSize are valid
    if (huffmanOut == nullptr || huffmanOutSize == 0) {
        std::cerr << "Error: huffmanOut is null or huffmanOutSize is zero for " << variableName << std::endl;
        free(huffmanOut);
        return nullptr;
    }

    // Define the variable for Huffman tree output
    adios2::Variable<unsigned char> fullHuffmanTreeVar = io.DefineVariable<unsigned char>(
        variableName, {huffmanOutSize}, {0}, {huffmanOutSize}, adios2::ConstantDims);

    // Write the Huffman tree output to the file
    engine.Put(fullHuffmanTreeVar, huffmanOut, adios2::Mode::Sync);

    // Clean up the serialized output buffer
    free(huffmanOut);

    return huffmanTreeFull;
}

template<typename T>
void computeTypeBufferAVL(
    std::vector<std::vector<ScidxAVLNode<T>*>>& singleSubTree, 
    T error_bound, size_t i, adios2::Engine& engine, adios2::IO& io,
    std::vector<std::vector<int>>& allTreeTypesLow, 
    std::vector<std::vector<int>>& allTreeTypesHigh, 
    std::vector<std::vector<size_t>>& allTreeId,
    std::vector<std::vector<int>>& allTreeTypesMaxHigh
) {
    std::vector<T> treeNodeMaxArray = flattenedTreeNodeMax(singleSubTree);
    std::vector<size_t> treeNodeIdArray = flattenedTreeNodeId(singleSubTree);

    std::vector<T> treeNodeMaxHighArray = flattenedTreeNodeMaxHigh(singleSubTree);
    std::vector<T> leafNodeMaxHighArray = getLeafNodeMaxHigh(singleSubTree);
    
    std::vector<int> treeNodeMaxType = computeArrayTypeAVL(treeNodeMaxArray, error_bound, engine, io, i, "High");
    std::vector<int> leafNodeMaxHighType = computeArrayTypeAVL(leafNodeMaxHighArray, error_bound, engine, io, i, "Maxhigh");
    std::vector<int> treeNodeMinType = preQuantiSingleTreeMin(singleSubTree, error_bound, engine, io, i, "Low");
    
    /*std::cout << "treeNodeMaxType content: ";
    for (const int& val : treeNodeMaxType) {
        std::cout << val << " ";
    }
    std::cout << std::endl;*/

    allTreeTypesHigh.push_back(treeNodeMaxType);
    allTreeId.push_back(treeNodeIdArray);
    allTreeTypesMaxHigh.push_back(leafNodeMaxHighType);
    allTreeTypesLow.push_back(treeNodeMinType);
}

template<typename T>
void compressTree(std::vector<int> currentTypesLow, std::vector<int> currentTypesHigh, std::vector<size_t> currentIds, std::vector<int> currentTypesMaxHigh, T error_bound, adios2::Engine& engine, adios2::IO& io, int step, scidx::HuffmanTree* fullHuffmanTreeLow, scidx::HuffmanTree* fullHuffmanTreeHigh, scidx::HuffmanTree* fullHuffmanTreeMaxHigh) {
    //压缩前type的大小，解压后还原大小
    std::vector<size_t> typeSizes;
    typeSizes.push_back(currentTypesHigh.size());
    typeSizes.push_back(currentTypesMaxHigh.size());
    typeSizes.push_back(currentTypesLow.size());

    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstdAVL(currentTypesHigh, fullHuffmanTreeHigh);
    std::vector<unsigned char> compressedLeafMaxHigh = singleHuffmanEncodeZstdAVL(currentTypesMaxHigh, fullHuffmanTreeMaxHigh);
    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstdAVL(currentTypesLow, fullHuffmanTreeLow);

    std::string varNamePrefix = "compressed_data_" + std::to_string(step);

    adios2::Variable<unsigned char> bpCompressedMax = io.DefineVariable<unsigned char>(varNamePrefix + "_high", {compressedMax.size()}, {0}, {compressedMax.size()}, adios2::ConstantDims);
    adios2::Variable<size_t> bpCompressedIds = io.DefineVariable<size_t>(varNamePrefix + "_id", {currentIds.size()}, {0}, {currentIds.size()}, adios2::ConstantDims);
    adios2::Variable<unsigned char> bpCompressedLeafMaxHigh = io.DefineVariable<unsigned char>(varNamePrefix + "_max_high", {compressedLeafMaxHigh.size()}, {0}, {compressedLeafMaxHigh.size()}, adios2::ConstantDims);
    adios2::Variable<unsigned char> bpCompressedMin = io.DefineVariable<unsigned char>(varNamePrefix + "_low", {compressedMin.size()}, {0}, {compressedMin.size()}, adios2::ConstantDims);
    adios2::Variable<size_t> bpTypeSizes = io.DefineVariable<size_t>(varNamePrefix + "_type", {typeSizes.size()}, {0}, {typeSizes.size()}, adios2::ConstantDims);
    
    engine.Put(bpCompressedMax, compressedMax.data(), adios2::Mode::Sync);
    engine.Put(bpCompressedIds, currentIds.data(), adios2::Mode::Sync);
    engine.Put(bpCompressedLeafMaxHigh, compressedLeafMaxHigh.data(), adios2::Mode::Sync);
    engine.Put(bpCompressedMin, compressedMin.data(), adios2::Mode::Sync);
    engine.Put(bpTypeSizes, typeSizes.data(), adios2::Mode::Sync);
}

template<typename T>
ScidxAVLNode<T>* decompressTreeAVL(adios2::Engine& engine, adios2::IO& io, int step, std::vector<int>& firstVector, T error_bound, std::vector<unsigned char> huffmanOutLow, std::vector<unsigned char> huffmanOutHigh, std::vector<unsigned char> huffmanOutMaxHigh) {
    std::string varNamePrefix = "compressed_data_" + std::to_string(step);

    adios2::Variable<unsigned char> bpCompressedMax = io.InquireVariable<unsigned char>(varNamePrefix + "_high");
    adios2::Variable<unsigned char> bpCompressedLeafMaxHigh = io.InquireVariable<unsigned char>(varNamePrefix + "_max_high");
    adios2::Variable<unsigned char> bpCompressedMin = io.InquireVariable<unsigned char>(varNamePrefix + "_low");
    adios2::Variable<size_t> bpTypeSizes = io.InquireVariable<size_t>(varNamePrefix + "_type");

    std::vector<unsigned char> compressedMax;
    std::vector<unsigned char> compressedLeafMaxHigh;
    std::vector<unsigned char> compressedMin;
    std::vector<size_t> typeSizes;
 
    engine.Get(bpCompressedMax, compressedMax);
    engine.Get(bpCompressedLeafMaxHigh, compressedLeafMaxHigh);
    engine.Get(bpCompressedMin, compressedMin);
    engine.Get(bpTypeSizes, typeSizes);
   
    std::string varNameOutlayer = "OutLayer_" + std::to_string(step);

    adios2::Variable<T> bpOutlayerHigh = io.InquireVariable<T>(varNameOutlayer + "High");
    adios2::Variable<T> bpOutlayerMaxhigh = io.InquireVariable<T>(varNameOutlayer + "Maxhigh");
    adios2::Variable<T> bpOutlayerLow = io.InquireVariable<T>(varNameOutlayer + "Low");

    size_t varSizeHigh = bpOutlayerHigh.Shape()[0];
    size_t varSizeMaxHigh = bpOutlayerMaxhigh.Shape()[0];
    size_t varSizeLow = bpOutlayerLow.Shape()[0];
    
    std::vector<T> outlayerHigh(varSizeHigh);
    std::vector<T> outlayerMaxHigh(varSizeMaxHigh);
    std::vector<T> outlayerLow(varSizeLow);

    engine.Get(bpOutlayerHigh, outlayerHigh.data(), adios2::Mode::Sync);
    engine.Get(bpOutlayerMaxhigh, outlayerMaxHigh.data(), adios2::Mode::Sync);
    engine.Get(bpOutlayerLow, outlayerLow.data(), adios2::Mode::Sync);

    /*std::cout << "outlayerHigh content: ";
    for (const T& val : outlayerHigh) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "outlayerHigh size: " << outlayerHigh.size() << std::endl;

    std::cout << "outlayerMaxHigh content: ";
    for (const T& val : outlayerMaxHigh) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "outlayerMaxHigh size: " << outlayerMaxHigh.size() << std::endl;

    std::cout << "outlayerLow content: ";
    for (const T& val : outlayerLow) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "outlayerLow size: " << outlayerLow.size() << std::endl;


    std::cout << "typeSizes[0] = " << typeSizes[0] << std::endl;*/

    std::vector<int> decompressedTypeMax = singleHuffmanDecodeZstdAVL(compressedMax, typeSizes[0], huffmanOutHigh.data());
    std::vector<int> decompressedTypeLeafMaxHigh = singleHuffmanDecodeZstdAVL(compressedLeafMaxHigh, typeSizes[1], huffmanOutMaxHigh.data());
    std::vector<int> decompressedTypeMin = singleHuffmanDecodeZstdAVL(compressedMin, typeSizes[2], huffmanOutLow.data());

    /*std::cout << "decompressedTypeMax content: ";
    for (const int& val : decompressedTypeMax) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "decompressedTypeMax size: " << decompressedTypeMax.size() << std::endl;*/

    ScidxAVLNode<T>* reconstructMin = reconstructAndDequantizeTree(decompressedTypeMin, firstVector, error_bound, outlayerLow);
    std::vector<T> decodeFlattenedMax = decodeArrayType(decompressedTypeMax, error_bound, outlayerHigh);
    std::vector<T> decodeFlattenedLeafMaxHigh = decodeArrayType(decompressedTypeLeafMaxHigh, error_bound, outlayerMaxHigh);

    reconstructMax(reconstructMin, decodeFlattenedMax);

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
std::vector<size_t> flattenedTreeNodeId(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree) {
    std::vector<size_t> flattenedNodeId;

    for (const auto &row : singleSubTree) {
        for (const auto &node : row) {
            flattenedNodeId.push_back(node->id);
        }
    }

    return flattenedNodeId;
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

template<typename T>
std::vector<int> preQuantiSingleTreeMin(std::vector<std::vector<ScidxAVLNode<T>*>> singleSubTree, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName) {
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

    std::string varNamePre = "OutLayer_" + std::to_string(i);

    adios2::Variable<T> bpOutlayer = io.DefineVariable<T>(varNamePre + variableName, {outLayerData.size()}, {0}, {outLayerData.size()}, adios2::ConstantDims);
    engine.Put(bpOutlayer, outLayerData.data(), adios2::Mode::Sync);
   
    return type;
}

// 对压缩结果进行 zstd 解压和 Huffman 的 decode，还原到 type
std::vector<int> singleHuffmanDecodeZstdAVL(std::vector<unsigned char> compressedByZstdData, size_t typeSize, unsigned char *s) {

    // zstd 的 decompress，compressedByZstdData 压缩后的数据
    std::vector<unsigned char> decompressedData = decompressWithZstdAVL(compressedByZstdData, typeSize * 4);

    // 解压后单子树大小，即与原始数据等长
    std::vector<int> decodedData(typeSize);

    //std::cout << "Type size: " << typeSize << std::endl;

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

    std::vector<unsigned char> compressedByZstdData = compressWithZstdAVL(singleEncodeOut, singleEncodeOutsize);

    return compressedByZstdData;
}

std::vector<unsigned char> AVL(const std::vector<unsigned char> &data, size_t dataLength) {
    // 默认的压缩级别为 3
    int level = 3;

    // 预估压缩后的大小，和 sz 保持一致
    size_t estimatedCompressedSize = (dataLength < 100) ? 200 : static_cast<size_t>(dataLength * 1.2);

    // 分配内存用于存储压缩后的数据
    std::vector<unsigned char> compressBytes(estimatedCompressedSize);

    // 使用 Zstandard 压缩算法进行压缩
    size_t compressedSize = ZSTD_compress(compressBytes.data(), estimatedCompressedSize, data.data(), dataLength, level);

    // 调整压缩后的数据大小
    compressBytes.resize(compressedSize);

    return compressBytes;
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
std::vector<int> computeArrayTypeAVL(const std::vector<T> &flattenedTreeNodeValue, T error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName) {
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    T checkRadius = (quantization_intervals - 1) * error_bound;
    T interval = 2 * error_bound;
    T recip_precision = 1 / error_bound;

    std::string varNamePreOut = "OutLayer_" + std::to_string(i);

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

template<typename T>
void reconstructMax(ScidxAVLNode<T>* root, const std::vector<T>& decodeFlattenedMax) {
    if (!root) return;

    std::queue<ScidxAVLNode<T>*> queue;
    queue.push(root);

    size_t index = 0;

    while (!queue.empty() && index < decodeFlattenedMax.size()) {
        ScidxAVLNode<T>* current = queue.front();
        queue.pop();

        current->interval.high = decodeFlattenedMax[index++];

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
template void compressTree<double>(std::vector<int>, std::vector<int>, std::vector<size_t>, std::vector<int>, double, adios2::Engine&, adios2::IO&, int, scidx::HuffmanTree*, scidx::HuffmanTree*, scidx::HuffmanTree*);
template ScidxAVLNode<double>* decompressTreeAVL<double>(adios2::Engine&, adios2::IO&, int, std::vector<int>&, double, std::vector<unsigned char>, std::vector<unsigned char>, std::vector<unsigned char>);
template void computeTypeBufferAVL<double>(std::vector<std::vector<ScidxAVLNode<double>*>>&, double, size_t, adios2::Engine&, adios2::IO&, std::vector<std::vector<int>>&, std::vector<std::vector<int>>&, std::vector<std::vector<size_t>>&, std::vector<std::vector<int>>&);

// 显式实例化模板函数
template void compressTree<float>(std::vector<int>, std::vector<int>, std::vector<size_t>, std::vector<int>, float, adios2::Engine&, adios2::IO&, int, scidx::HuffmanTree*, scidx::HuffmanTree*, scidx::HuffmanTree*);
template ScidxAVLNode<float>* decompressTreeAVL<float>(adios2::Engine&, adios2::IO&, int, std::vector<int>&, float, std::vector<unsigned char>, std::vector<unsigned char>, std::vector<unsigned char>);
template void computeTypeBufferAVL<float>(std::vector<std::vector<ScidxAVLNode<float>*>>&, float, size_t, adios2::Engine&, adios2::IO&, std::vector<std::vector<int>>&, std::vector<std::vector<int>>&, std::vector<std::vector<size_t>>&, std::vector<std::vector<int>>&);


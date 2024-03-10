/**
 * Scientific Compact Data Index (scidx)
 * Authors: Ning Yan, Lipeng Wan, Sheng Di
 * */

#include <scidx.h>
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


ScidxRBNode<float>*  compressTree(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree, float error_bound, std::vector<int> firstVector);


std::vector<float> flattenedTreeNodeMax(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree);

std::vector<float> flattenedTreeNodeMaxHigh(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree);

std::vector<float> getLeafNodeMaxHigh(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree);


std::vector<int> preQuantiSingleTreeMin(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree, float error_bound);

std::vector<int> computeArrayType(const std::vector<float> &flattenedTreeNodeValue, float error_bound, const std::string& outputFileName);

std::vector<unsigned char> singleHuffmanEncodeZstd(std::vector<int> type, HuffmanTree *huffmanTree);

std::vector<int> singleHuffmanDecodeZstd(std::vector<unsigned char> compressedByZstdData, std::vector<int> type, unsigned char *s);

std::vector<unsigned char> compressWithZstd(const std::vector<unsigned char> &data, size_t dataLength);

std::vector<unsigned char> decompressWithZstd(const std::vector<unsigned char> &compressedData, uint64_t targetOriSize);

std::vector<int> compress_data(float *oriData, size_t dataLength, float error_bound);

void decode_withSubTree(unsigned char *s, unsigned char *encode, size_t targetLength, int *out);

std::vector<float> flattenTreeMaxHigh(const std::vector<std::vector<ScidxRBNode<float> *>> &tree);

std::vector<float> decodeArrayType(const std::vector<int>& type, const std::string& inputFileName, float error_bound);

float dequantizeValue(int quantizedValue, float error_bound, int intvRadius, float interval, float& baseValue);

ScidxRBNode<float>* reconstructAndDequantizeTree(const std::vector<int>& type, const std::vector<int>& structureVec, float error_bound);

void reconstructMax(ScidxRBNode<float>* root, const std::vector<float>& decodeFlattenedMax);

void reconstractLeafMaxHigh(ScidxRBNode<float>* root, const std::vector<float>& decodeFlattenedLeafMaxHigh);

void updateMaxHighOfTree(ScidxRBNode<float>* node);



// 对单个子tree进行进行pre+quanti
// min，max,max_high,id. min层级顺序预测，但每层第一个点由父节点预测. max,max_high层级预测
// 对单个子树进行pre+quanti,按顺序合并type,去构建大的HuffmanTree,然后用单个子tree的type去encode



ScidxRBNode<float>*  compressTree(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree, float error_bound, std::vector<int> firstVector){
    std::vector<float> treeNodeMaxArray = flattenedTreeNodeMax(singleSubTree);
    std::vector<float> treeNodeMaxHighArray = flattenedTreeNodeMaxHigh(singleSubTree);
    std::vector<float> leafNodeMaxHighArray = getLeafNodeMaxHigh(singleSubTree);
    
    std::cout << "leafNodeMaxHighArray contains: ";
    for (float value : leafNodeMaxHighArray) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    size_t bytesOfMax = treeNodeMaxArray.size() * sizeof(float);
    std::cout << "treeNodeMax occupies " << bytesOfMax << " bytes before" << std::endl;



    std::vector<int> treeNodeMaxType = computeArrayType(treeNodeMaxArray, error_bound, "OutlayerMax.txt");
    std::vector<int> treeNodeMaxHighType = computeArrayType(treeNodeMaxHighArray, error_bound, "OutlayerMaxHigh.txt");
    std::vector<int> leafNodeMaxHighType = computeArrayType(leafNodeMaxHighArray, error_bound, "OutlayerMaxHighLeafNode.txt");
    std::vector<int> treeNodeMinType = preQuantiSingleTreeMin(singleSubTree, error_bound);

    int stateNum = 2 * 16384;
    HuffmanTree *huffmanTreeMax = createHuffmanTree(stateNum);
    HuffmanTree *huffmanTreeMaxHigh = createHuffmanTree(stateNum);
    HuffmanTree *huffmanTreeLeafMaxHigh = createHuffmanTree(stateNum);
    HuffmanTree *huffmanTreeMin = createHuffmanTree(stateNum);

    unsigned char *huffmanOutMax, *huffmanOutMaxHigh, *huffmanOutLeafMaxHigh, *huffmanOutMin = nullptr;
    size_t huffmanOutMaxSize, huffmanOutMaxHighSize, huffmanOutLeafMaxHighSize, huffmanOutMinSize = 0;

    // 根据所有数据构建的全Huffmantree
    //TODO:是根据每种数据构建自己的Huffmantree,还是全部数据构建成一个
    //TODO:type替换成全部的总type
    init_and_serialize_Huffmantree(huffmanTreeMax, treeNodeMaxType.data(), treeNodeMaxType.size(), &huffmanOutMax, &huffmanOutMaxSize);
    init_and_serialize_Huffmantree(huffmanTreeMaxHigh, treeNodeMaxHighType.data(), treeNodeMaxHighType.size(), &huffmanOutMaxHigh, &huffmanOutMaxHighSize);
    init_and_serialize_Huffmantree(huffmanTreeLeafMaxHigh, leafNodeMaxHighType.data(), leafNodeMaxHighType.size(), &huffmanOutLeafMaxHigh, &huffmanOutLeafMaxHighSize);
    init_and_serialize_Huffmantree(huffmanTreeMin, treeNodeMinType.data(), treeNodeMinType.size(), &huffmanOutMin, &huffmanOutMinSize);

    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstd(treeNodeMaxType, huffmanTreeMax);
    std::vector<unsigned char> compressedMaxHigh = singleHuffmanEncodeZstd(treeNodeMaxHighType, huffmanTreeMaxHigh);
    std::vector<unsigned char> compressedLeafMaxHigh = singleHuffmanEncodeZstd(leafNodeMaxHighType, huffmanTreeLeafMaxHigh);
    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstd(treeNodeMinType, huffmanTreeMin);

    std::vector<int> decompressedTypeMax = singleHuffmanDecodeZstd(compressedMax, treeNodeMaxType, huffmanOutMax);
    std::vector<int> decompressedTypeMaxHigh =singleHuffmanDecodeZstd(compressedMaxHigh, treeNodeMaxHighType, huffmanOutMaxHigh);
    std::vector<int> decompressedTypeLeafMaxHigh =singleHuffmanDecodeZstd(compressedLeafMaxHigh, leafNodeMaxHighType, huffmanOutLeafMaxHigh);
    std::vector<int> decompressedTypeMin =singleHuffmanDecodeZstd(compressedMin, treeNodeMinType, huffmanOutMin);
    
    ScidxRBNode<float>* reconstructMin = reconstructAndDequantizeTree(decompressedTypeMin, firstVector, error_bound);
    std::vector<float> decodeFlattenedMax = decodeArrayType(decompressedTypeMax, "OutlayerMax.txt", error_bound);
    std::vector<float> decodeFlattenedLeafMaxHigh = decodeArrayType(decompressedTypeLeafMaxHigh, "OutlayerMaxHighLeafNode.txt", error_bound);

    std::cout << "decodeFlattenedLeafMaxHigh contains: ";
    for (float value : decodeFlattenedLeafMaxHigh) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    reconstructMax(reconstructMin,decodeFlattenedMax);

    reconstractLeafMaxHigh(reconstructMin, decodeFlattenedLeafMaxHigh);

    updateMaxHighOfTree(reconstructMin);

    return reconstructMin;
    
       

}

// 对单个子树的max,打平为层级顺序
std::vector<float> flattenedTreeNodeMax(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree)
{
    std::vector<float> flattenedNodeMax;

    for (const auto &row : singleSubTree)
    {
        for (const auto &node : row)
        {
            flattenedNodeMax.push_back(node->interval.high);
        }
    }

    return flattenedNodeMax;
}

// 对单个子树的max_high,打平为层级顺序
std::vector<float> flattenedTreeNodeMaxHigh(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree)
{
    std::vector<float> flattenedNodeMaxHigh;

    for (const auto &row : singleSubTree)
    {
        for (const auto &node : row)
        {
            flattenedNodeMaxHigh.push_back(node->max_high);
        }
    }

    return flattenedNodeMaxHigh;
}

std::vector<float> getLeafNodeMaxHigh(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree) {
    std::vector<float> leafNodeMaxHigh;

    if (singleSubTree.empty()) return leafNodeMaxHigh; // 确保输入不为空

    // 遍历所有层，除了最后一层
    for (size_t i = 0; i < singleSubTree.size() - 1; ++i) {
        for (ScidxRBNode<float>* node : singleSubTree[i]) {
            // 如果节点是叶子节点（即没有子节点），则添加其max_high值
            if (node->left == nullptr && node->right == nullptr) {
                leafNodeMaxHigh.push_back(node->max_high);
            }
        }
    }

    // 对于最后一层，添加所有节点的max_high值
    for (ScidxRBNode<float>* node : singleSubTree.back()) {
        leafNodeMaxHigh.push_back(node->max_high);
    }

    return leafNodeMaxHigh;
}

// 对单个子树的min,进行prediction+quantization,返回pre+quanti后的值
// 对于子树的min,用每一层的第一个点的解压值去预测下一层的第一个点，其他点层级顺序预测
// 由什么值去预测下一个点没关系，但是用的一定是解压值，因为解压的时候拿不到原始值
std::vector<int> preQuantiSingleTreeMin(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree, float error_bound)
{

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    float checkRadius = (quantization_intervals - 1) * error_bound;
    float interval = 2 * error_bound;
    float recip_precision = 1 / error_bound;

    std::vector<int> type;
    float curData, predData, firstNodePredData;

    std::ofstream originalDataFile("original_data.txt");
    // level是当前层
    for (size_t level = 0; level < singleSubTree.size(); ++level)
    {
        for (size_t index = 0; index < singleSubTree[level].size(); ++index)
        {
            // index是当前层的第n个点
            // curData为真实值
            curData = singleSubTree[level][index]->interval.low;

            if (level == 0 && index == 0)
            {

                // TODO：记录第一个点原始值
                type.push_back(0);
                firstNodePredData = curData;
                originalDataFile << curData << std::endl;
            }
            else if (index == 0)
            {
                // 每一层的第一个点通过上一层第一个点的解压值预测
                float predAbsErr = std::abs(curData - firstNodePredData);

                // 根据预测误差进行量化
                if (predAbsErr < checkRadius)
                {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= firstNodePredData)
                    {
                        type.push_back(intvRadius + state);
                        predData = firstNodePredData + state * interval;
                        firstNodePredData = firstNodePredData + state * interval;
                    }
                    else // curData<pred
                    {
                        type.push_back(intvRadius - state);
                        predData = firstNodePredData - state * interval;
                        firstNodePredData = firstNodePredData - state * interval;
                    }
                }
                else
                {
                    // 处理超范围数据
                    // TODO：记录当前原始值
                    std::cout << "outLayer aaaaaaaaa" << std::endl;
                    type.push_back(0);
                    predData = curData;
                    firstNodePredData = curData;
                    originalDataFile << curData << std::endl;
                }
            }
            else
            {
                float predAbsErr = std::abs(curData - predData);

                // 根据预测误差进行量化
                if (predAbsErr < checkRadius)
                {
                    int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
                    if (curData >= predData)
                    {
                        type.push_back(intvRadius + state);
                        predData = predData + state * interval;
                    }
                    else // curData<pred
                    {
                        type.push_back(intvRadius - state);
                        predData = predData - state * interval;
                    }
                }
                else
                {
                    std::cout << "outLayer aaaaaaaaa" << std::endl;
                    // 处理不可预测的数据
                    type.push_back(0);
                    predData = curData;
                    originalDataFile << curData << std::endl;
                }
            }
        }
    }
    originalDataFile.close();
    return type;
}

// 根据数组，通过前一个值的解压值预测后一个值，计算pre+quantization后的值
std::vector<int> computeArrayType(const std::vector<float> &flattenedTreeNodeValue, float error_bound, const std::string& outputFileName)
{
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    float checkRadius = (quantization_intervals - 1) * error_bound;
    float interval = 2 * error_bound;
    float recip_precision = 1 / error_bound;

    std::ofstream originalDataFile(outputFileName);

    // 创建一个与 flattenedTreeNodeValue 大小相同的、初始值为 0 的向量
    std::vector<int> type(flattenedTreeNodeValue.size(), 0);

    float predData = flattenedTreeNodeValue[0];
    originalDataFile << flattenedTreeNodeValue[0] << std::endl;
    for (size_t i = 1; i < flattenedTreeNodeValue.size(); i++)
    {
        float curData = flattenedTreeNodeValue[i];
        float predAbsErr = std::abs(curData - predData);

        // 根据预测误差进行量化
        if (predAbsErr < checkRadius)
        {
            int state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
            if (curData >= predData)
            {
                type[i] = intvRadius + state;
                predData = predData + state * interval;
            }
            else // curData<pred
            {
                type[i] = intvRadius - state;
                predData = predData - state * interval;
            }
        }
        else
        {
            // 处理超越范围的
            type[i] = 0;
            predData = curData;
            originalDataFile << curData << std::endl;
        }
    }

    return type;
}

std::vector<float> decodeArrayType(const std::vector<int>& type, const std::string& inputFileName, float error_bound)
{
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    float interval = 2 * error_bound;

    std::ifstream compressedDataFile(inputFileName);
    if (!compressedDataFile.is_open()) {
        // 处理文件打开失败的情况
        throw std::runtime_error("Failed to open input file: " + inputFileName);
    }

    std::vector<float> decodedValues;
    float predData = 0.0f;

    for (size_t i = 0; i < type.size(); ++i)
    {
        if (type[i] == 0)
        {
            // 处理超越范围的情况
            float curData;
            compressedDataFile >> curData;
            decodedValues.push_back(curData);
            predData = curData;
        }
        else
        {

            int state = std::abs(type[i] - intvRadius);
            float adjustment = state * interval;
            float curData = (type[i] >= intvRadius) ? predData  + adjustment : predData  - adjustment;

            decodedValues.push_back(curData);
            predData = curData;
        }
    }

    compressedDataFile.close();
    return decodedValues;
}


//对type进行Huffman encode和zstd压缩
std::vector<unsigned char> singleHuffmanEncodeZstd(std::vector<int> type, HuffmanTree *huffmanTree )
{
    // 单个需要encode的子树
    std::vector<unsigned char> singleEncodeOut(type.size() * 4);
    size_t singleEncodeOutsize = 0;

    encode(huffmanTree, type.data(), type.size(), singleEncodeOut.data(), &singleEncodeOutsize);

    std::vector<unsigned char> compressedByZstdData = compressWithZstd(singleEncodeOut, singleEncodeOutsize);
    std::cout << "Compressed size: " << compressedByZstdData.size() << " bytes" << std::endl;

    return compressedByZstdData;
}

//对压缩结果进行zstd解压和Huffman的decode，还原到type
std::vector<int> singleHuffmanDecodeZstd(std::vector<unsigned char> compressedByZstdData, std::vector<int> type, unsigned char *s)
{
    std::vector<unsigned char> decompressedData = decompressWithZstd(compressedByZstdData, type.size()*4);
    
    // 解压后单子树大小，即与原始数据等长
    //TODO：实际decode时候并不知道type.size()
    std::vector<int> decodedData(type.size());

    // 使用全Huffmantree，对单子树的encode值，进行decode
    decode_withSubTree(s, decompressedData.data(), decodedData.size(), decodedData.data());

    //其中data是实际的解压数据，size是type.size
    return decodedData;
}

std::vector<unsigned char> compressWithZstd(const std::vector<unsigned char> &data, size_t dataLength)
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

std::vector<unsigned char> decompressWithZstd(const std::vector<unsigned char> &compressedData, uint64_t targetOriSize)
{
    // TODO：zstd解压后数据大小应该如何赋值
    // 分配内存用于存储解压后的数据
    std::vector<unsigned char> oriData(targetOriSize);

    // 使用 Zstandard 解压缩算法进行解压
    size_t outSize = ZSTD_decompress(oriData.data(), targetOriSize, compressedData.data(), compressedData.size());

    // 调整解压后的数据大小
    oriData.resize(outSize);

    return oriData;
}

void decode_withSubTree(unsigned char *s, unsigned char *encode, size_t targetLength, int *out)
{
    int stateNum = 2 * 16384;
    HuffmanTree *decodeHuffmanTree = createHuffmanTree(stateNum);

    size_t nodeCount = bytesToInt_bigEndian(s);
    node root = reconstruct_HuffTree_from_bytes_anyStates(decodeHuffmanTree, s + 8, nodeCount);

    decode(encode, targetLength, root, out);
}

std::vector<int> compress_data(float *oriData, size_t dataLength, float error_bound)
{

    size_t i;

    // int* type = (int*) malloc(dataLength*sizeof(int));
    std::vector<int> type(dataLength, 0);

    type[0] = 0;

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;

    int state;
    float checkRadius;
    float curData;
    float predData = oriData[0];
    float predAbsErr;
    checkRadius = (quantization_intervals - 1) * error_bound;
    float interval = 2 * error_bound;
    float recip_precision = 1 / error_bound;

    for (i = 1; i < dataLength; i++)
    {
        curData = oriData[i];
        predAbsErr = std::abs(curData - predData);

        if (predAbsErr < checkRadius)
        {
            state = ((int)(predAbsErr * recip_precision + 1)) >> 1;

            if (curData >= predData)
            {
                type[i] = intvRadius + state;
                predData = predData + state * interval;
            }
            else // curData<pred
            {
                type[i] = intvRadius - state;
                predData = predData - state * interval;
            }
            continue;
        }

        // unpredictable data processing
        type[i] = 0;
        predData = oriData[i];
    }

    int stateNum = 2 * 16384;
    HuffmanTree *huffmanTree = createHuffmanTree(stateNum);

    unsigned char *out = nullptr;
    size_t outsize = 0;

    // 根据所有数据构建的全Huffmantree
    init_and_serialize_Huffmantree(huffmanTree, type.data(), type.size(), &out, &outsize);

    // 单个需要encode的子树
    std::vector<unsigned char> encodeOut(type.size() * 4);
    size_t encodeOutsize = 0;

    encode(huffmanTree, type.data(), type.size(), encodeOut.data(), &encodeOutsize);

    std::cout << "原始数据：" << std::endl;
    for (size_t i = 0; i < encodeOutsize; ++i)
    {
        std::cout << static_cast<int>(encodeOut.data()[i]) << " ";
    }
    std::cout << std::endl;

    std::vector<unsigned char> compressedByZstdData = compressWithZstd(encodeOut, encodeOutsize);

    std::vector<unsigned char> decompressedData = decompressWithZstd(compressedByZstdData, encodeOutsize);

    std::cout << "解压后数据：" << std::endl;
    for (size_t i = 0; i < decompressedData.size(); ++i)
    {
        std::cout << static_cast<int>(decompressedData.data()[i]) << " ";
    }
    std::cout << std::endl;

    // 解压后单子树大小，即与原始数据等长
    std::vector<int> decodedData(type.size());

    // 使用全Huffmantree，对单子树的encode值，进行decode
    decode_withSubTree(out, decompressedData.data(), decodedData.size(), decodedData.data());

    // 输出解码结果
    for (int value : decodedData)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    free(out);

    return type;
}

/* void init_and_serialize_FullHuffmantree(int *s, size_t length, unsigned char **out, size_t *outSize)
{
    int stateNum = 2*65536;
    HuffmanTree* huffmanTree = createHuffmanTree(stateNum);
    unsigned char* out = NULL;
    size_t outsize = 0;
    init_and_serialize_Huffmantree(huffmanTree, s, length, out, &outsize);
} */

// add single subtree encoded buffer to mainbuffer
void encode_and_append_to_buffer(HuffmanTree *huffmanTree, int *m, size_t mLength, std::vector<unsigned char> &buffer, std::vector<size_t> &encodedSizes)
{
    unsigned char *tempBuffer = nullptr;
    size_t enCodeSize = 0;

    encode(huffmanTree, m, mLength, tempBuffer, &enCodeSize);

    buffer.insert(buffer.end(), tempBuffer, tempBuffer + enCodeSize);

    // size of every encoded subtree
    encodedSizes.push_back(enCodeSize);

    free(tempBuffer);
}



void CollectMaxHighInOrder(const std::vector<std::vector<ScidxRBNode<float> *>> &tree, int level, int index, std::vector<float> &flattenedNodeMax_high)
{
    if (level >= tree.size() || index >= tree[level].size())
    {
        return; // 边界检查
    }

    // 计算左子节点的索引
    int leftIndex = 2 * index;
    // 递归遍历左子树
    CollectMaxHighInOrder(tree, level + 1, leftIndex, flattenedNodeMax_high);

    // 添加当前节点的 max_high 到 flattenedNodeMax_high
    ScidxRBNode<float> *node = tree[level][index];
    if (node != nullptr)
    {
        flattenedNodeMax_high.push_back(node->max_high);
    }

    // 计算右子节点的索引
    int rightIndex = 2 * index + 1;
    // 递归遍历右子树
    CollectMaxHighInOrder(tree, level + 1, rightIndex, flattenedNodeMax_high);
}

std::vector<float> flattenTreeMaxHigh(const std::vector<std::vector<ScidxRBNode<float> *>> &tree)
{
    std::vector<float> flattenedNodeMax_high;
    CollectMaxHighInOrder(tree, 0, 0, flattenedNodeMax_high);
    return flattenedNodeMax_high;
}

float dequantizeValue(int quantizedValue, float error_bound, int intvRadius, float interval, float& baseValue) {
    if (quantizedValue == 0) {
        // Special case handling: use the base value directly
        return baseValue;
    }
    int state = std::abs(quantizedValue - intvRadius);
    float adjustment = state * interval;
    return (quantizedValue >= intvRadius) ? baseValue + adjustment : baseValue - adjustment;
}




ScidxRBNode<float>* reconstructAndDequantizeTree(const std::vector<int>& type, const std::vector<int>& structureVec, float error_bound) {
    for (int elem : type) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    
    if (type.empty() || structureVec.empty()) return nullptr;

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    float interval = 2 * error_bound;

    // Read the root value from "original_data.txt"
    std::ifstream originalDataFile("original_data.txt");
    float rootValue;
    if (!(originalDataFile >> rootValue)) {
        std::cerr << "Failed to read root value from original_data.txt\n";
        return nullptr;
    }
    originalDataFile.close();

    ScidxRBNode<float>* root = new ScidxRBNode<float>({ScidxInterval<float>{rootValue, rootValue}, 0});
    std::queue<ScidxRBNode<float>**> nodesQueue;
    nodesQueue.push(&root);

    size_t typeIndex = 0;
    size_t structIndex = 0;
    float baseValue = rootValue; // Initialize base value with the root's value

    std::queue<float> baseValuesForNextLevel; // Queue to manage base values for the first node of each level
    int currentLevelNodeCount = 1;
    int processedNodeCount = 0;
    bool isLevelFirstNode = true; // Flag to indicate the first node of the current level
    int readIndex = 0;

    while (!nodesQueue.empty() && structIndex < structureVec.size()) {
        ScidxRBNode<float>** currentNodePtr = nodesQueue.front();
        nodesQueue.pop();
        
        // If this is the first node of a new level, update the base value accordingly
        if (isLevelFirstNode && !baseValuesForNextLevel.empty()) {
            baseValue = baseValuesForNextLevel.front();
            baseValuesForNextLevel.pop();
        }

        // Process the current node if it exists
        float dequantizedValue;
        if (structureVec[structIndex] == 1) {
            if (typeIndex != 0 && type[typeIndex] == 0) {
                std::cout << "Index: " << typeIndex << ", Value: " << type[typeIndex] << std::endl;
                std::ifstream dataFile("original_data.txt");
                // Set the file position to the correct position based on typeIndex
                dataFile.seekg(readIndex * sizeof(int), std::ios::beg);
                float readValue;
                if (!(dataFile >> readValue)) {
                    std::cerr << "Failed to read data from original_data.txt\n";
                    dataFile.close();
                    return nullptr;
                }
                dataFile.close();
                dequantizedValue = readValue;
                typeIndex++;

            }
            else {
                dequantizedValue = dequantizeValue(type[typeIndex], error_bound, intvRadius, interval, baseValue);
                typeIndex++;

            }
            *currentNodePtr = new ScidxRBNode<float>({ScidxInterval<float>{dequantizedValue, dequantizedValue}, 0});
            if (isLevelFirstNode) {
                baseValuesForNextLevel.push(dequantizedValue);
                isLevelFirstNode = false; // Reset flag for the current level
            }
            baseValue = dequantizedValue;
            // Add children to the queue
            nodesQueue.push(&((*currentNodePtr)->left));
            nodesQueue.push(&((*currentNodePtr)->right));
        }

        // Check if the current level is complete
        if (++processedNodeCount == currentLevelNodeCount && !nodesQueue.empty()) {
            // Prepare for the next level
            currentLevelNodeCount = nodesQueue.size(); // Update the count for the next level
            processedNodeCount = 0; // Reset processed node count for the new level
            isLevelFirstNode = true; // The next node processed will be the first node of a new level
        }

        structIndex++; // Move to the next structure vector element
    }

    return root;
}

void reconstructMax(ScidxRBNode<float>* root, const std::vector<float>& decodeFlattenedMax) {
    if (!root) return; 

    std::queue<ScidxRBNode<float>*> queue;
    queue.push(root);

    size_t index = 0; 

    while (!queue.empty() && index < decodeFlattenedMax.size()) {
        ScidxRBNode<float>* current = queue.front();
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




void reconstractLeafMaxHigh(ScidxRBNode<float>* root, const std::vector<float>& decodeFlattenedLeafMaxHigh) {
    if (!root) return; // 确保树不为空

    std::queue<ScidxRBNode<float>*> nodesQueue; // 用于BFS的队列
    nodesQueue.push(root);
    size_t valueIndex = 0; // 当前处理的decodeFlattenedLeafMaxHigh中的索引

    // 用于标记最后一层开始的标志
    bool lastLayerStarted = false;
    std::queue<ScidxRBNode<float>*> nextLayerNodes;
    nextLayerNodes.push(root);

    while (!nodesQueue.empty()) {
        size_t layerSize = nodesQueue.size();
        lastLayerStarted = nextLayerNodes.empty();

        while (layerSize-- > 0) {
            ScidxRBNode<float>* currentNode = nodesQueue.front();
            nodesQueue.pop();

            // 如果当前节点是叶子节点或已经开始处理最后一层
            if ((currentNode->left == nullptr && currentNode->right == nullptr && !lastLayerStarted) || lastLayerStarted) {
                if (valueIndex < decodeFlattenedLeafMaxHigh.size()) {
                    currentNode->max_high = decodeFlattenedLeafMaxHigh[valueIndex++];
                } else {
                    std::cerr << "Error: Not enough values in decodeFlattenedLeafMaxHigh to update the node." << std::endl;
                    return;
                }
            }

            // 将子节点加入队列
            if (currentNode->left) {
                nodesQueue.push(currentNode->left);
                nextLayerNodes.push(currentNode->left);
            }
            if (currentNode->right) {
                nodesQueue.push(currentNode->right);
                nextLayerNodes.push(currentNode->right);
            }
        }

        // 准备下一层节点
        if (nextLayerNodes.size() == nodesQueue.size()) {
            nextLayerNodes = std::queue<ScidxRBNode<float>*>(); // 清空下一层节点的标记
        }
    }

    // 检查是否所有的max_high值都已使用
    if (valueIndex < decodeFlattenedLeafMaxHigh.size()) {
        std::cerr << "Warning: Not all values in decodeFlattenedLeafMaxHigh were used." << std::endl;
    }
}

void updateMaxHighOfTree(ScidxRBNode<float>* node) {
    // 基本情况：如果节点为空，不需要更新
    if (node == nullptr) {
        return;
    }

    // 递归地更新左子树和右子树的max_high值
    updateMaxHighOfTree(node->left);
    updateMaxHighOfTree(node->right);

    // 当前节点的max_high值至少是它自身的max值（即区间的high值）
    float maxHigh = node->interval.high;

    // 如果节点是叶子节点，它可能已经有一个预设的max_high值，使用预设的max_high值或当前的max值中的较大者
    if (node->left == nullptr && node->right == nullptr) {
        maxHigh = std::max(maxHigh, node->max_high);
    }
    else {
        // 如果不是叶子节点，计算包括子节点的max_high值在内的最大max_high值
        if (node->left != nullptr) {
            maxHigh = std::max(maxHigh, node->left->max_high);
        }
        if (node->right != nullptr) {
            maxHigh = std::max(maxHigh, node->right->max_high);
        }
    }

    // 更新当前节点的max_high值
    node->max_high = maxHigh;
}
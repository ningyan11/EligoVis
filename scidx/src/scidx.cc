/**
 * Scientific Compact Data Index (scidx)
 * Authors: Ning Yan, Lipeng Wan, Sheng Di
 * */

#include <scidx.h>
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


void compressTree(std::vector<int> currentTypesLow, std::vector<int> currentTypesHigh, std::vector<int> currentTypesMaxHigh, float error_bound, adios2::Engine& engine, adios2::IO& io, int step, scidx::HuffmanTree*  fullHuffmanTreeLow, scidx::HuffmanTree*  fullHuffmanTreeHigh, scidx::HuffmanTree*  fullHuffmanTreeMaxHigh);

ScidxRBNode<float>* decompressTree(adios2::Engine& engine, adios2::IO& io, int step, std::vector<int>& firstVector, float error_bound, std::vector<unsigned char> huffmanOut);


std::vector<float> flattenedTreeNodeMax(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree);

std::vector<float> flattenedTreeNodeMaxHigh(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree);

std::vector<float> getLeafNodeMaxHigh(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree);


std::vector<int> preQuantiSingleTreeMin(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree, float error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName);

std::vector<int> computeArrayType(const std::vector<float> &flattenedTreeNodeValue, float error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName);
std::vector<unsigned char> singleHuffmanEncodeZstd(std::vector<int> type, HuffmanTree *huffmanTree);

std::vector<int> singleHuffmanDecodeZstd(std::vector<unsigned char> compressedByZstdData, size_t typeSize, unsigned char *s);


std::vector<unsigned char> compressWithZstd(const std::vector<unsigned char> &data, size_t dataLength);

std::vector<unsigned char> decompressWithZstd(const std::vector<unsigned char> &compressedData, uint64_t targetOriSize);

std::vector<int> compress_data(float *oriData, size_t dataLength, float error_bound);

void decode_withSubTree(unsigned char *s, unsigned char *encode, size_t targetLength, int *out);

std::vector<float> flattenTreeMaxHigh(const std::vector<std::vector<ScidxRBNode<float> *>> &tree);

std::vector<float> decodeArrayType(const std::vector<int>& type, float error_bound, std::vector<float> outlayerData);

float dequantizeValue(int quantizedValue, float error_bound, int intvRadius, float interval, float& baseValue);

ScidxRBNode<float>* reconstructAndDequantizeTree(const std::vector<int>& type, const std::vector<int>& structureVec, float error_bound, std::vector<float> outlayerLow);

void reconstructMax(ScidxRBNode<float>* root, const std::vector<float>& decodeFlattenedMax);

void reconstractLeafMaxHigh(ScidxRBNode<float>* root, const std::vector<float>& decodeFlattenedLeafMaxHigh);

void updateMaxHighOfTree(ScidxRBNode<float>* node);

void computeTypeBuffer(
    std::vector<std::vector<ScidxRBNode<float>*>>& singleSubTree, 
    float error_bound, size_t i, adios2::Engine& engine, adios2::IO& io,
    std::vector<std::vector<int>>& allTreeTypesLow, 
    std::vector<std::vector<int>>& allTreeTypesHigh, 
    std::vector<std::vector<int>>& allTreeTypesMaxHigh
);



std::vector<scidx::HuffmanTree*> fullHuffman(const std::vector<std::vector<int>>& allTreeTypesLow,
                                          const std::vector<std::vector<int>>& allTreeTypesHigh,
                                          const std::vector<std::vector<int>>& allTreeTypesMaxHigh,
                                          adios2::Engine& engine, adios2::IO& io);

HuffmanTree* processHuffmanTree(const std::vector<std::vector<int>>& allTreeTypes, adios2::Engine& engine, adios2::IO& io, const std::string& variableName);



void computeTypeBuffer(
    std::vector<std::vector<ScidxRBNode<float>*>>& singleSubTree, 
    float error_bound, size_t i, adios2::Engine& engine, adios2::IO& io,
    std::vector<std::vector<int>>& allTreeTypesLow, 
    std::vector<std::vector<int>>& allTreeTypesHigh, 
    std::vector<std::vector<int>>& allTreeTypesMaxHigh
) {
    // 将树状结构打平成一维数据
    std::vector<float> treeNodeMaxArray = flattenedTreeNodeMax(singleSubTree);
    std::vector<float> treeNodeMaxHighArray = flattenedTreeNodeMaxHigh(singleSubTree);
    std::vector<float> leafNodeMaxHighArray = getLeafNodeMaxHigh(singleSubTree);

    
    // 进行 quantization
    std::vector<int> treeNodeMaxType = computeArrayType(treeNodeMaxArray, error_bound, engine, io, i, "High");
    std::vector<int> leafNodeMaxHighType = computeArrayType(leafNodeMaxHighArray, error_bound, engine, io,  i, "Maxhigh");
    std::vector<int> treeNodeMinType = preQuantiSingleTreeMin(singleSubTree, error_bound, engine, io, i, "Low");
    

    std::cout << "treeNodeMaxType content: ";
    for (const int& val : treeNodeMaxType) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 将新计算的数据添加到相应的二维向量中
    allTreeTypesHigh.push_back(treeNodeMaxType);
    allTreeTypesMaxHigh.push_back(leafNodeMaxHighType);
    allTreeTypesLow.push_back(treeNodeMinType);
}


std::vector<scidx::HuffmanTree*> fullHuffman(const std::vector<std::vector<int>>& allTreeTypesLow,
                                          const std::vector<std::vector<int>>& allTreeTypesHigh,
                                          const std::vector<std::vector<int>>& allTreeTypesMaxHigh,
                                          adios2::Engine& engine, adios2::IO& io){

        std::vector<HuffmanTree*> huffmanTrees;

        

        // 处理并写入不同的 Huffman 树
        HuffmanTree* huffmanTreeLow = processHuffmanTree(allTreeTypesLow, engine, io, "fullHuffmanTreeLow");
        huffmanTrees.push_back(huffmanTreeLow);

        HuffmanTree* huffmanTreeHigh = processHuffmanTree(allTreeTypesHigh, engine, io, "fullHuffmanTreeHigh");
        huffmanTrees.push_back(huffmanTreeHigh);

        HuffmanTree* huffmanTreeMaxHigh = processHuffmanTree(allTreeTypesMaxHigh, engine, io, "fullHuffmanTreeMaxHigh");
        huffmanTrees.push_back(huffmanTreeMaxHigh);

        

        return huffmanTrees;
    }

    HuffmanTree* processHuffmanTree(const std::vector<std::vector<int>>& allTreeTypes, adios2::Engine& engine, adios2::IO& io, const std::string& variableName) {
        std::vector<int> allTypes;
        for (const auto& type : allTreeTypes) {
            allTypes.insert(allTypes.end(), type.begin(), type.end());
        }

        int stateNum = 2 * 16384;
        HuffmanTree *huffmanTreeFull = createHuffmanTree(stateNum);

        unsigned char *huffmanOut = nullptr;
        size_t huffmanOutSize = 0;
        init_and_serialize_Huffmantree(huffmanTreeFull, allTypes.data(), allTypes.size(), &huffmanOut, &huffmanOutSize);

        std::cout << "huffmanOut content for " << variableName << ":" << std::endl;
        for (size_t i = 0; i < huffmanOutSize; ++i) {
            std::cout << +huffmanOut[i] << " ";
        }
        std::cout << std::endl;

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

// 对单个子tree进行进行pre+quanti
// min，max,max_high,id. min层级顺序预测，但每层第一个点由父节点预测. max,max_high层级预测
// 对单个子树进行pre+quanti,按顺序合并type,去构建大的HuffmanTree,然后用单个子tree的type去encode
void compressTree(std::vector<int> currentTypesLow, std::vector<int> currentTypesHigh, std::vector<int> currentTypesMaxHigh, float error_bound, adios2::Engine& engine, adios2::IO& io, int step, scidx::HuffmanTree*  fullHuffmanTreeLow, scidx::HuffmanTree*  fullHuffmanTreeHigh, scidx::HuffmanTree*  fullHuffmanTreeMaxHigh){

    std::vector<size_t> typeSizes;
    typeSizes.push_back(currentTypesHigh.size());
    typeSizes.push_back(currentTypesMaxHigh.size());
    typeSizes.push_back(currentTypesLow.size());

    //进行Huffman的encode和zstd压缩
    std::vector<unsigned char> compressedMax = singleHuffmanEncodeZstd(currentTypesHigh, fullHuffmanTreeHigh);
    std::vector<unsigned char> compressedLeafMaxHigh = singleHuffmanEncodeZstd(currentTypesMaxHigh, fullHuffmanTreeMaxHigh);
    std::vector<unsigned char> compressedMin = singleHuffmanEncodeZstd(currentTypesLow, fullHuffmanTreeLow);

   // 每次调用 compressTree 函数创建不同的变量名
    std::string varNamePrefix = "compressed_data_" + std::to_string(step);

    // 定义 ADIOS 变量
    adios2::Variable<unsigned char> bpCompressedMax = io.DefineVariable<unsigned char>(varNamePrefix + "_high", {compressedMax.size()}, {0}, {compressedMax.size()}, adios2::ConstantDims);
    adios2::Variable<unsigned char> bpCompressedLeafMaxHigh = io.DefineVariable<unsigned char>(varNamePrefix + "_max_high", {compressedLeafMaxHigh.size()}, {0}, {compressedLeafMaxHigh.size()}, adios2::ConstantDims);
    adios2::Variable<unsigned char> bpCompressedMin = io.DefineVariable<unsigned char>(varNamePrefix + "_low", {compressedMin.size()}, {0}, {compressedMin.size()}, adios2::ConstantDims);
    adios2::Variable<size_t> bpTypeSizes = io.DefineVariable<size_t>(varNamePrefix + "_type", {typeSizes.size()}, {0}, {typeSizes.size()}, adios2::ConstantDims);
    
    // 写入数据
    engine.Put(bpCompressedMax, compressedMax.data(), adios2::Mode::Sync);
    engine.Put(bpCompressedLeafMaxHigh, compressedLeafMaxHigh.data(), adios2::Mode::Sync);
    engine.Put(bpCompressedMin, compressedMin.data(), adios2::Mode::Sync);
    engine.Put(bpTypeSizes, typeSizes.data(), adios2::Mode::Sync);

    
}


//单个树的解压重建
ScidxRBNode<float>* decompressTree(adios2::Engine& engine, adios2::IO& io, int step, std::vector<int>& firstVector, float error_bound, std::vector<unsigned char> huffmanOutLow, std::vector<unsigned char> huffmanOutHigh, std::vector<unsigned char> huffmanOutMaxHigh){
    
   
    std::string varNamePrefix = "compressed_data_" + std::to_string(step);


    // 获取定义的变量
    adios2::Variable<unsigned char> bpCompressedMax = io.InquireVariable<unsigned char>(varNamePrefix + "_high");
    adios2::Variable<unsigned char> bpCompressedLeafMaxHigh = io.InquireVariable<unsigned char>(varNamePrefix + "_max_high");
    adios2::Variable<unsigned char> bpCompressedMin = io.InquireVariable<unsigned char>(varNamePrefix + "_low");
    adios2::Variable<size_t> bpTypeSizes = io.InquireVariable<size_t>(varNamePrefix + "_type");


    // 读取数据
    std::vector<unsigned char> compressedMax;
    std::vector<unsigned char> compressedLeafMaxHigh;
    std::vector<unsigned char> compressedMin;
    std::vector<size_t> typeSizes;
 
    engine.Get(bpCompressedMax, compressedMax);
    engine.Get(bpCompressedLeafMaxHigh, compressedLeafMaxHigh);
    engine.Get(bpCompressedMin, compressedMin);
    engine.Get(bpTypeSizes, typeSizes);
   
    std::string varNameOutlayer = "OutLayer_" + std::to_string(step);

    // 定义 ADIOS 变量
    adios2::Variable<float> bpOutlayerHigh= io.InquireVariable<float>(varNameOutlayer + "High");
    adios2::Variable<float> bpOutlayerMaxhigh = io.InquireVariable<float>(varNameOutlayer + "Maxhigh");
    adios2::Variable<float> bpOutlayerLow = io.InquireVariable<float>(varNameOutlayer + "Low");

    size_t varSizeHigh = bpOutlayerHigh.Shape()[0];
    size_t varSizeMaxHigh = bpOutlayerMaxhigh.Shape()[0];
    size_t varSizeLow = bpOutlayerLow.Shape()[0];
    
    std::vector<float> outlayerHigh(varSizeHigh);
    std::vector<float> outlayerMaxHigh(varSizeMaxHigh);
    std::vector<float> outlayerLow(varSizeLow);

    engine.Get(bpOutlayerHigh, outlayerHigh.data(), adios2::Mode::Sync);
    engine.Get(bpOutlayerMaxhigh, outlayerMaxHigh.data(), adios2::Mode::Sync);
    engine.Get(bpOutlayerLow, outlayerLow.data(), adios2::Mode::Sync);

    std::cout << "outlayerHigh content: ";
    for (const float& val : outlayerHigh) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "outlayerHigh size: " << outlayerHigh.size() << std::endl;

    std::cout << "outlayerMaxHigh content: ";
    for (const float& val : outlayerMaxHigh) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "outlayerMaxHigh size: " << outlayerMaxHigh.size() << std::endl;

    std::cout << "outlayerLow content: ";
    for (const float& val : outlayerLow) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "outlayerLow size: " << outlayerLow.size() << std::endl;


    std::cout << "typeSizes[0] = " << typeSizes[0] << std::endl;

    std::vector<int> decompressedTypeMax = singleHuffmanDecodeZstd(compressedMax, typeSizes[0], huffmanOutHigh.data());
    std::vector<int> decompressedTypeLeafMaxHigh =singleHuffmanDecodeZstd(compressedLeafMaxHigh, typeSizes[1], huffmanOutMaxHigh.data());
    std::vector<int> decompressedTypeMin =singleHuffmanDecodeZstd(compressedMin, typeSizes[2], huffmanOutLow.data());

    std::cout << "decompressedTypeMax content: ";
    for (const int& val : decompressedTypeMax) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "decompressedTypeMax size: " << decompressedTypeMax.size() << std::endl;



            
    ScidxRBNode<float>* reconstructMin = reconstructAndDequantizeTree(decompressedTypeMin, firstVector, error_bound, outlayerLow );
    std::vector<float> decodeFlattenedMax = decodeArrayType(decompressedTypeMax, error_bound, outlayerHigh);
    std::vector<float> decodeFlattenedLeafMaxHigh = decodeArrayType(decompressedTypeLeafMaxHigh, error_bound, outlayerMaxHigh );

            
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
std::vector<int> preQuantiSingleTreeMin(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree, float error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName)
{

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    float checkRadius = (quantization_intervals - 1) * error_bound;
    float interval = 2 * error_bound;
    float recip_precision = 1 / error_bound;

    std::vector<int> type;
    float curData, predData, firstNodePredData;
    std::vector<float> outLayerData;

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
                type.push_back(0);
                firstNodePredData = curData;
                outLayerData.push_back(curData);
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
                    
                    type.push_back(0);
                    predData = curData;
                    firstNodePredData = curData;
                    outLayerData.push_back(curData);
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
                    // 处理不可预测的数据
                    type.push_back(0);
                    predData = curData;
                    outLayerData.push_back(curData);
                }
            }
        }
    }

    std::string varNamePre = "OutLayer_" + std::to_string(i);

    // 定义 ADIOS 变量
    adios2::Variable<float> bpOutlayer=  io.DefineVariable<float>(varNamePre + variableName, {outLayerData.size()}, {0}, {outLayerData.size()}, adios2::ConstantDims);
     
    engine.Put(bpOutlayer, outLayerData.data(), adios2::Mode::Sync);
   
    return type;
}

// 根据数组，通过前一个值的解压值预测后一个值，计算pre+quantization后的值
std::vector<int> computeArrayType(const std::vector<float> &flattenedTreeNodeValue, float error_bound, adios2::Engine& engine, adios2::IO& io, size_t i, const std::string& variableName)
{
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    float checkRadius = (quantization_intervals - 1) * error_bound;
    float interval = 2 * error_bound;
    float recip_precision = 1 / error_bound;

    std::string varNamePreOut = "OutLayer_" + std::to_string(i);

    // 创建一个与 flattenedTreeNodeValue 大小相同的、初始值为 0 的向量
    std::vector<int> type(flattenedTreeNodeValue.size(), 0);

    std::vector<float> outLayerData;

    float predData = flattenedTreeNodeValue[0];
    outLayerData.push_back(flattenedTreeNodeValue[0]);
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
            outLayerData.push_back(curData);
        }
    }

    // 定义 ADIOS 变量
    adios2::Variable<float> bpOutlayer= io.DefineVariable<float>(varNamePreOut + variableName, {outLayerData.size()}, {0}, {outLayerData.size()}, adios2::ConstantDims);
     
    engine.Put(bpOutlayer, outLayerData.data(), adios2::Mode::Sync);

    std::cout << "outLayerData size: " << outLayerData.size() << std::endl;

    std::cout << "outLayerData content: ";
    for (const float& value : outLayerData) {
        std::cout << value << " ";
    }
    std::cout << std::endl;



    return type;
}

std::vector<float> decodeArrayType(const std::vector<int>& type, float error_bound, std::vector<float> outlayerData)
{
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    float interval = 2 * error_bound;

 

    std::vector<float> decodedValues;
    float predData = 0.0f;
    size_t outlayerIndex = 0; 

    for (size_t i = 0; i < type.size(); ++i)
    {
        if (type[i] == 0)
        {
            if (outlayerIndex >= outlayerData.size()) {
                throw std::runtime_error("outlayerdata index out of range");
            }
            // 处理超越范围的情况
            float curData = outlayerData[outlayerIndex];
            outlayerIndex++;
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

    return compressedByZstdData;
}

//对压缩结果进行zstd解压和Huffman的decode，还原到type
std::vector<int> singleHuffmanDecodeZstd(std::vector<unsigned char> compressedByZstdData, size_t typeSize, unsigned char *s)
{

    //zstd的decompress，compressedByZstdData压缩后的数据
    std::vector<unsigned char> decompressedData = decompressWithZstd(compressedByZstdData, typeSize*4);

    // 解压后单子树大小，即与原始数据等长
    std::vector<int> decodedData(typeSize);

    std::cout << "Type size: " << typeSize << std::endl;

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

//s为序列化后的HuffmanTree，encode为需要解码的数据
void decode_withSubTree(unsigned char *s, unsigned char *encode, size_t targetLength, int *out)
{
    int stateNum = 2 * 16384;
    HuffmanTree *decodeHuffmanTree = createHuffmanTree(stateNum);

    size_t nodeCount = bytesToInt_bigEndian(s);
    node root = reconstruct_HuffTree_from_bytes_anyStates(decodeHuffmanTree, s + 8, nodeCount);

    decode(encode, targetLength, root, out);
}


/*std::vector<int> compress_data(float *oriData, size_t dataLength, float error_bound)
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
    decode_withSubTree(decompressedData.data(), decodedData.size(), decodedData.data());

    // 输出解码结果
    for (int value : decodedData)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    free(out);

    return type;
}*/

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




ScidxRBNode<float>* reconstructAndDequantizeTree(const std::vector<int>& type, const std::vector<int>& structureVec, float error_bound, std::vector<float> outlayerLow) {
    
    if (type.empty() || structureVec.empty()) return nullptr;

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    float interval = 2 * error_bound;

    if (outlayerLow.empty()) {
        std::cerr << "outlayerLow is empty\n";
        return nullptr;
    }

    float rootValue = outlayerLow[0];
    size_t outlayerIndex = 1; // 用来跟踪从outlayerLow中读取的索引

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
                // 从outlayerLow中读取原始数据
                if (outlayerIndex >= outlayerLow.size()) {
                    std::cerr << "outlayerLow index out of range\n";
                    return nullptr;
                }
                dequantizedValue = outlayerLow[outlayerIndex];
                outlayerIndex++;
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
/**
 * Scientific Compact Data Index (scidx)
 * Authors: Ning Yan, Lipeng Wan, Sheng Di
 * */

#include <cmath>
#include <cstdlib> 
#include <scidx_rb_interval_tree.h>
#include <scidx_defines.h>
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <zstd.h>
#include <scidx_BytesToolkit.h>

using namespace scidx;

std::vector<std::vector<int>> compress_index(std::vector<std::vector<ScidxrbNode<float>*>> subTreeNodesInLevels, float error_bound);

std::vector<int> compress_data_layered(std::vector<std::vector<ScidxrbNode<float>*>>& subTreeNodesInLevels, float error_bound);

int computeState(float curData, float predData, float error_bound, int intvRadius, float checkRadius, float interval, float recip_precision);

std::vector<int> compress_data(float *oriData, size_t dataLength, float error_bound);

void decode_withSubTree(unsigned char *s, unsigned char *encode, size_t targetLength, int *out);

std::vector<float> flattenTreeMaxHigh(const std::vector<std::vector<ScidxrbNode<float>*>>& tree);

//compress single subtree
std::vector<std::vector<int>> compress_index(std::vector<std::vector<ScidxrbNode<float>*>> subTreeNodesInLevels, float error_bound)
{
    std::vector<std::vector<int>> allCompressedType;
    std::vector<ScidxrbNode<float>*> flattenedNodes;
    std::vector<float> flattenedNodeLow;
    std::vector<float> flattenedNodeHigh;
    std::vector<size_t> flattenedNodeIds;
    std::vector<float> flattenedNodeMax_highByLevel;
    std::vector<float> flattenedNodeMax_high;

    for (const auto& row : subTreeNodesInLevels) {
        for (const auto& node : row) {
            flattenedNodes.push_back(node);
            flattenedNodeLow.push_back(node->rbInterval.low);
            flattenedNodeHigh.push_back(node->rbInterval.high);
            flattenedNodeIds.push_back(node->id);
            flattenedNodeMax_highByLevel.push_back(node-> max_high);
        }
    }

    std::cout << "LevelOrder max_high values: ";
    for (float val : flattenedNodeMax_highByLevel) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    flattenedNodeMax_high = flattenTreeMaxHigh(subTreeNodesInLevels);
    std::cout << "InOrder max_high values: ";
    for (float val : flattenedNodeMax_high) {
        std::cout << val << " ";
    }
    std::cout << std::endl;



    std::vector<int> nodeLowCompressedType = compress_data(flattenedNodeLow.data(), flattenedNodeLow.size(), error_bound);
    //std::vector<int> nodeHighCompressedType = compress_data(flattenedNodeHigh.data(), flattenedNodeHigh.size(), error_bound);
    std::vector<int> nodeMaxHighCompressedType = compress_data(flattenedNodeMax_high.data(), flattenedNodeMax_high.size(), error_bound);

    allCompressedType.push_back(nodeLowCompressedType);
    //allCompressedType.push_back(nodeHighCompressedType);
    allCompressedType.push_back(nodeMaxHighCompressedType);

    return allCompressedType;

}


std::vector<int> compress_data_layered(std::vector<std::vector<ScidxrbNode<float>*>> subTreeNodesInLevels, float error_bound)
{
    std::vector<int> type;
    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals / 2;
    float checkRadius = (quantization_intervals - 1) * error_bound;
    float interval = 2 * error_bound;
    float recip_precision = 1 / error_bound;
    float predData, curData, predAbsErr;
    int state;

    for (size_t level = 0; level < subTreeNodesInLevels.size(); ++level) {
        for (size_t index = 0; index < subTreeNodesInLevels[level].size(); ++index) {
            curData = subTreeNodesInLevels[level][index]->rbInterval.low; // Assuming we are compressing 'low' values

            if (level == 0 && index == 0) {
                // First data point in the tree
                predData = curData;
                type.push_back(0);
            } else if (index == 0) {
                // First data point of each level, use the first data point of the previous level to predict
                predData = subTreeNodesInLevels[level - 1][0]->rbInterval.low;
                type.push_back(computeState(curData, predData, error_bound, intvRadius, checkRadius, interval, recip_precision));
            } else {
                // Other data points, use the previous data point in the same level to predict
                predData = subTreeNodesInLevels[level][index - 1]->rbInterval.low;
                type.push_back(computeState(curData, predData, error_bound, intvRadius, checkRadius, interval, recip_precision));
            }
        }
    }

    return type;
}

int computeState(float curData, float predData, float error_bound, int intvRadius, float checkRadius, float interval, float recip_precision) {
    float predAbsErr = std::abs(curData - predData);
    int state;

    if (predAbsErr < checkRadius) {
        state = ((int)(predAbsErr * recip_precision + 1)) >> 1;
        if (curData >= predData) {
            return intvRadius + state;
        } else {
            return intvRadius - state;
        }
    } else {
        // Unpredictable data processing
        return 0;
    }
}


std::vector<int> compress_data(float *oriData, size_t dataLength, float error_bound)
{
    std::cout << "start to compress" << std::endl;

    size_t i;

    // int* type = (int*) malloc(dataLength*sizeof(int));
    std::vector<int> type(dataLength, 0); 

    type[0] = 0;

    int quantization_intervals = 16384;
    int intvRadius = quantization_intervals/2; 

    int state;
    float checkRadius;
    float curData;
    float predData = oriData[0];
    float predAbsErr;
    checkRadius = (quantization_intervals-1)*error_bound;
    float interval = 2*error_bound;
    float recip_precision = 1/error_bound;
    

    for (i = 1; i < dataLength; i++) {
        curData = oriData[i];
        predAbsErr = std::abs(curData - predData);
        
        if(predAbsErr < checkRadius){
            state = ((int)(predAbsErr*recip_precision+1))>>1;

            if(curData >= predData)
                {
                    type[i] = intvRadius+state;
                    predData = predData + state*interval;
                }
                else //curData<pred
                {
                    type[i] = intvRadius-state;
                    predData = predData - state*interval;
                }
                continue;
        }

        //unpredictable data processing
		type[i] = 0;
        predData = oriData[i];
    }
    // free(type);
    //return type;

    //HuffmanTree and SZ

int stateNum = 2 * 16384;
HuffmanTree* huffmanTree = createHuffmanTree(stateNum);

unsigned char* out = nullptr;
size_t outsize = 0;

//根据所有数据构建的全Huffmantree
init_and_serialize_Huffmantree(huffmanTree, type.data(), type.size(), &out, &outsize);


//std::cout << "type.size() = " << type.size() << std::endl; 

//单个需要encode的子树
std::vector<unsigned char> encodeOut(type.size() * 4);
size_t encodeOutsize = 0;

encode(huffmanTree, type.data(), type.size(), encodeOut.data(), &encodeOutsize);


// 解压后单子树大小，即与原始数据等长
std::vector<int> decodedData(type.size()); 

//使用全Huffmantree，对单子树的encode值，进行decode
decode_withSubTree(out, encodeOut.data(), decodedData.size(), decodedData.data());

// 输出解码结果
for (int value : decodedData) {
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

    //size of every encoded subtree
    encodedSizes.push_back(enCodeSize);

    free(tempBuffer);
}


void decode_withSubTree(unsigned char *s, unsigned char *encode, size_t targetLength, int *out)
{
	int stateNum = 2 * 16384;
    HuffmanTree* decodeHuffmanTree = createHuffmanTree(stateNum);

    size_t nodeCount = bytesToInt_bigEndian(s);
	node root = reconstruct_HuffTree_from_bytes_anyStates(decodeHuffmanTree,s+8, nodeCount);


	decode(encode, targetLength, root, out);
}



void CollectMaxHighInOrder(const std::vector<std::vector<ScidxrbNode<float>*>>& tree, int level, int index, std::vector<float>& flattenedNodeMax_high) {
    if (level >= tree.size() || index >= tree[level].size()) {
        return; // 边界检查
    }

    // 计算左子节点的索引
    int leftIndex = 2 * index;
    // 递归遍历左子树
    CollectMaxHighInOrder(tree, level + 1, leftIndex, flattenedNodeMax_high);

    // 添加当前节点的 max_high 到 flattenedNodeMax_high
    ScidxrbNode<float>* node = tree[level][index];
    if (node != nullptr) {
        flattenedNodeMax_high.push_back(node->max_high);
    }

    // 计算右子节点的索引
    int rightIndex = 2 * index + 1;
    // 递归遍历右子树
    CollectMaxHighInOrder(tree, level + 1, rightIndex, flattenedNodeMax_high);
}

std::vector<float> flattenTreeMaxHigh(const std::vector<std::vector<ScidxrbNode<float>*>>& tree) {
    std::vector<float> flattenedNodeMax_high;
    CollectMaxHighInOrder(tree, 0, 0, flattenedNodeMax_high);
    return flattenedNodeMax_high;
}






#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <scidx_avl.h> 
#include <SZ3/api/sz.hpp> 
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <utility> 
#include <scidx_avl_interval_tree.h>  
#include <scidx_block_min_max.h>      
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem> 
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include "../miniIsosurface/marchingCubes/util/Image3D.h"
#include "../miniIsosurface/marchingCubes/util/TriangleMesh.h"
#include "../miniIsosurface/marchingCubes/util/SaveTriangleMesh.h"
#include "../miniIsosurface/marchingCubes/util/util.h"
#include "../miniIsosurface/marchingCubes/util/MarchingCubesTables.h"
#include "../miniIsosurface/marchingCubes/util/Timer.h"
#include "../miniIsosurface/marchingCubes/util/LoadImage.h"
#include "../miniIsosurface/marchingCubes/mantevoCommon/YAML_Doc.hpp"
#include <unordered_map>
#include <unordered_set>


void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
}


std::vector<ScidxInterval<double>> loadBigBlockIndexFile(const std::string& filename) {
    std::vector<ScidxInterval<double>> bigBlockIndices;

    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return bigBlockIndices;  // 返回空
    }

    double minVal, maxVal;
    while (inFile.read(reinterpret_cast<char*>(&minVal), sizeof(double)) &&
           inFile.read(reinterpret_cast<char*>(&maxVal), sizeof(double))) {
        bigBlockIndices.push_back({minVal, maxVal});
    }

    inFile.close();
    return bigBlockIndices;
}


std::vector<size_t> safeQueryOverlapIds(ScidxAVLNode<double>* root, const ScidxInterval<double>& query) {
    if (root && root->id == static_cast<size_t>(-1)) {
        return {};  // dummy 树，返回空
    }
    return queryOverlapIds(root, query);
}


// ============================================================
// 直接查询步级 index 树
// 去掉大块层后，每步只有一棵树，文件名格式为 {stepNum}-0
// 返回该步内的原始 flat 小块 ID（均匀 index）
// ============================================================
std::vector<size_t> queryIndexRawIds(
    size_t actualStepNum,
    const std::string& indexDir,
    const std::vector<double>& queryRange,
    double error_bound)
{
    // 无大块层，localBlockID 固定为 0
    std::string treeID = indexDir + std::to_string(actualStepNum) + "-0";

    std::cout << "[queryIndexRawIds] treeID: " << treeID << std::endl;

    auto [decompressedTree, decodedSkippedMin, decodedSkippedMax, skippedIdOut] =
        decompressOptimizedAVL(treeID, error_bound);
    
    auto time_before_query = std::chrono::high_resolution_clock::now();

    adjustAndUpdateMaxHigh(decompressedTree, error_bound);

    ScidxInterval<double> query;
    query.low  = queryRange[0];
    query.high = queryRange[1];

    auto ids = safeQueryOverlapIds(decompressedTree, query);

    for (size_t i = 0; i < decodedSkippedMin.size(); ++i) {
        if (skippedIdOut[i] == static_cast<size_t>(-1)) continue;
        double minWithError = decodedSkippedMin[i] - error_bound;
        double maxWithError = decodedSkippedMax[i] + error_bound;
        if (!(minWithError > query.high || maxWithError < query.low)) {
            ids.push_back(skippedIdOut[i]);
        }
    }

    auto time_after_query = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> adjustAndQueryTime = time_after_query - time_before_query;
    std::cerr << "[Time5] adjustAndQueryInTree (" << treeID << "): "
              << adjustAndQueryTime.count() << " seconds" << std::endl;


    std::cout << "[queryIndexRawIds] Step " << actualStepNum
              << " -> " << ids.size() << " raw block IDs" << std::endl;
    return ids;
}


//blockSize每小块数据个数大小，totalBlocks全部原数据小块大小
// 自包含的通用解压函数

// 内联工具函数：计算分块配置参数
struct BlockConfig {
    size_t blockSize;
    size_t unpredSizeBits;
    size_t dataSizeBytes;
    size_t signBytesPerBlock;
    
    BlockConfig(size_t bs) : blockSize(bs) {
        // 根据分块大小计算参数
        if (bs == 64) {           // 4³
            unpredSizeBits = 6;
            dataSizeBytes = 1;
            signBytesPerBlock = 8;
        } else if (bs == 4096) {  // 16³
            unpredSizeBits = 12;
            dataSizeBytes = 2;
            signBytesPerBlock = 512;
        } else if (bs == 262144) { // 64³
            unpredSizeBits = 18;
            dataSizeBytes = 3;
            signBytesPerBlock = 32768;
        } else if (bs == 16777216) { // 256³
            unpredSizeBits = 24;
            dataSizeBytes = 4;
            signBytesPerBlock = 2097152;
        } else if (bs == 1073741824) { // 1024³
            unpredSizeBits = 30;
            dataSizeBytes = 4;
            signBytesPerBlock = 134217728;
        } else {
            throw std::runtime_error("Unsupported block size");
        }
    }
};

// 内联工具函数：解包位数据
void unpackBitsInline(const std::vector<uint8_t>& input, std::vector<size_t>& output, 
                     size_t bitsPerValue, size_t numValues) {
    output.clear();
    output.reserve(numValues);
    
    if (input.empty() || numValues == 0) return;
    
    size_t byteIndex = 0;
    size_t bitIndex = 0;
    uint64_t mask = (1ULL << bitsPerValue) - 1;
    
    for (size_t i = 0; i < numValues; i++) {
        uint64_t value = 0;
        size_t bitsRead = 0;
        
        while (bitsRead < bitsPerValue) {
            if (byteIndex >= input.size()) {
                throw std::runtime_error("Insufficient data for unpacking");
            }
            
            size_t bitsToRead = std::min(bitsPerValue - bitsRead, 8 - bitIndex);
            uint8_t byteMask = ((1 << bitsToRead) - 1) << bitIndex;
            uint8_t bits = (input[byteIndex] & byteMask) >> bitIndex;
            
            value |= (static_cast<uint64_t>(bits) << bitsRead);
            bitsRead += bitsToRead;
            bitIndex += bitsToRead;
            
            if (bitIndex >= 8) {
                bitIndex = 0;
                byteIndex++;
            }
        }
        
        output.push_back(static_cast<size_t>(value & mask));
    }
}

// 内联工具函数：读取多字节数值
uint64_t readValueInline(const uint8_t* data, size_t numBytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < numBytes; i++) {
        value |= (static_cast<uint64_t>(data[i]) << (i * 8));
    }
    return value;
}

// 通用解压函数 - 选择性读取版本
std::vector<std::vector<double>> universalBatchDecompressBlocks(
    const std::string& subDir,
    const std::vector<size_t>& blockIds,
    size_t blockSize,
    size_t totalBlocks,
    double error_bound) {
    
    int rank = 0;
    std::cout << "universal read: " << std::endl;
    
    // 获取块配置
    BlockConfig config(blockSize);
    
    auto beginReadOriginalBlocks = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> decompressedOriginalresult;

    std::cout << "subDir = " << subDir << "\n";
    std::cout << "universal read1: " << std::endl;


    
    // 文件名
    std::string unpredDataFileName = subDir + "universal_unpredData.bin";
    std::string compressedFileName = subDir + "universal_compressed_data.bin";
    std::string signFileName = subDir + "universal_sign_data.bin";
    
    std::ifstream unpredStream(unpredDataFileName, std::ios::binary);
    std::ifstream compStream(compressedFileName, std::ios::binary);
    std::ifstream signStream(signFileName, std::ios::binary);
    
    if (!unpredStream || !compStream || !signStream) {
        throw std::runtime_error("Failed to open one or more input files.");
    }

    std::cout << "universal read2: " << std::endl;
    
    // 获取文件大小
    unpredStream.seekg(0, std::ios::end);
    size_t unpredFileSize = unpredStream.tellg();
    unpredStream.seekg(0, std::ios::beg);
    
    compStream.seekg(0, std::ios::end);
    size_t compFileSize = compStream.tellg();
    compStream.seekg(0, std::ios::beg);
    
    signStream.seekg(0, std::ios::end);
    size_t signFileSize = signStream.tellg();
    signStream.seekg(0, std::ios::beg);

    std::cout << "universal read3: " << std::endl;
    
    // 读取所有文件到内存
    std::vector<uint8_t> allUnpredDataRead(unpredFileSize);
    std::vector<uint8_t> allCompDataRead(compFileSize);
    std::vector<uint8_t> allSignDataRead(signFileSize);

    
    unpredStream.read(reinterpret_cast<char*>(allUnpredDataRead.data()), unpredFileSize);
    compStream.read(reinterpret_cast<char*>(allCompDataRead.data()), compFileSize);
    signStream.read(reinterpret_cast<char*>(allSignDataRead.data()), signFileSize);

    std::cout << "universal read4: " << std::endl;
    
    // 解析unpredSizes
    size_t unpredSizeBytes = (totalBlocks * config.unpredSizeBits + 7) / 8;
    std::vector<uint8_t> compressedUnpredSizes(allUnpredDataRead.begin(),
                                              allUnpredDataRead.begin() + unpredSizeBytes);
    std::vector<size_t> unpredSizes;
    unpackBitsInline(compressedUnpredSizes, unpredSizes, config.unpredSizeBits, totalBlocks);
    
    // 读取bitCounts
    std::vector<uint8_t> bitCounts(totalBlocks);
    std::memcpy(bitCounts.data(), allCompDataRead.data(), totalBlocks);
    
    // 读取compSizes
    std::vector<size_t> compSizes(totalBlocks);
    for (size_t i = 0; i < totalBlocks; i++) {
        size_t offset = totalBlocks + i * config.dataSizeBytes;
        compSizes[i] = static_cast<size_t>(readValueInline(
            allCompDataRead.data() + offset, config.dataSizeBytes));
    }
    
    auto endReadOriginalBlocks = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readOriginalBlocks = endReadOriginalBlocks - beginReadOriginalBlocks;
    std::cerr << "[Rank " << rank << "][Time6]: read all compressed small blocks time: " 
              << readOriginalBlocks.count() << " seconds" << std::endl;
    
    // 计算偏移量
    std::vector<size_t> unpredOffsets(totalBlocks, 0);
    std::vector<size_t> compOffsets(totalBlocks, 0);
    
    for (size_t i = 1; i < totalBlocks; ++i) {
        unpredOffsets[i] = unpredOffsets[i - 1] + unpredSizes[i - 1];
        compOffsets[i] = compOffsets[i - 1] + compSizes[i - 1];
    }
    
    // 预分配缓冲区
    std::vector<double> unpredData;
    std::vector<uint8_t> compData;
    std::vector<uint8_t> signBits(config.signBytesPerBlock);
    
    for (size_t bid : blockIds) {
        // 读取unpredData
        size_t unpredSize = unpredSizes[bid];
        size_t unpredOffset = unpredSizeBytes + sizeof(double) * unpredOffsets[bid];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredDataRead.data() + unpredOffset,
                   unpredSize * sizeof(double));
        
        // 读取压缩数据
        uint8_t bitCount = bitCounts[bid];
        size_t dataSize = compSizes[bid];
        size_t compOffset = totalBlocks + totalBlocks * config.dataSizeBytes + compOffsets[bid];
        compData.resize(dataSize);
        std::memcpy(compData.data(), allCompDataRead.data() + compOffset, dataSize);
        
        // 读取符号数据
        size_t signOffset = config.signBytesPerBlock * bid;
        std::memcpy(signBits.data(), allSignDataRead.data() + signOffset, config.signBytesPerBlock);
        
        // 解码量化索引
        int radius = 512;
        std::vector<int> quant_inds(blockSize);
        size_t bitPos = 0;
        
        for (size_t i = 0; i < blockSize; ++i) {
            uint32_t val = 0, bitsRead = 0;
            
            while (bitsRead < bitCount && (bitPos / 8) < compData.size()) {
                size_t byteIndex = bitPos / 8;
                size_t bitOffset = bitPos % 8;
                size_t available = std::min(static_cast<size_t>(bitCount - bitsRead), 8u - bitOffset);
                
                val |= ((compData[byteIndex] >> bitOffset) & ((1 << available) - 1)) << bitsRead;
                bitsRead += available;
                bitPos += available;
            }
            
            int sign = (signBits[i / 8] >> (i % 8)) & 1;
            quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);
            quant_inds[i] += radius;
        }
        
        // 重建metadata
        std::vector<SZ3::uchar> metadataBuffer;
        metadataBuffer.push_back(0b00000010);
        
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&error_bound),
            reinterpret_cast<SZ3::uchar*>(&error_bound) + sizeof(double));
        
        int correctRadius = radius;
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&correctRadius),
            reinterpret_cast<SZ3::uchar*>(&correctRadius) + sizeof(int));
        
        size_t correctUnpredSize = static_cast<size_t>(unpredSize);
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize) + sizeof(size_t));
        
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data()),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data() + unpredSize));
        
        // SZ3解压配置
        SZ3::Config conf(blockSize);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;
        
        auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);
        
        std::vector<double> decompressedBlock(blockSize);
        const SZ3::uchar* metaPtr = metadataBuffer.data();
        size_t metaSize = metadataBuffer.size();
        
        decompose.load(metaPtr, metaSize);
        decompose.decompress(conf, quant_inds, decompressedBlock.data());
        
        decompressedOriginalresult.push_back(std::move(decompressedBlock));
    }
    
    auto endDecompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompressSmallBlocks = endDecompress - endReadOriginalBlocks;
    std::cerr << "[Rank " << rank << "][Time7]: decompress small blocks time: " 
              << decompressSmallBlocks.count() << " seconds" << std::endl;
    
    return decompressedOriginalresult;
}


// ============================================================================
// 辅助函数1（double版）：获取邻居块指针
// 说明：即使不再做26邻居扩展预解压，Marching Cubes内部区域2-8（面/棱/角边界）
//       仍然会尝试查邻居块指针 —— 如果邻居块没有被解压（不在blockMap中），
//       这里会返回nullptr，对应边界区域会被自动跳过（不生成三角形），
//       这就是"块边界处理会出现缝隙"的来源，是本次简化后接受的代价。
// ============================================================================
inline const double* getNeighborBlockPointer(
    size_t blockId,
    int dx, int dy, int dz,
    const std::unordered_map<size_t, const double*>& blockMap,
    const std::vector<size_t>& blockCountPerDim)
{
    // 计算当前块的3D坐标
    size_t bz = blockId % blockCountPerDim[2];
    size_t remaining = blockId / blockCountPerDim[2];
    size_t by = remaining % blockCountPerDim[1];
    size_t bx = remaining / blockCountPerDim[1];    
    
    // 计算邻居块的3D坐标
    int nbr_x = static_cast<int>(bx) + dx;
    int nbr_y = static_cast<int>(by) + dy;
    int nbr_z = static_cast<int>(bz) + dz;
    
    // 边界检查：邻居是否在有效范围内
    if (nbr_x < 0 || nbr_x >= static_cast<int>(blockCountPerDim[0]) ||
        nbr_y < 0 || nbr_y >= static_cast<int>(blockCountPerDim[1]) ||
        nbr_z < 0 || nbr_z >= static_cast<int>(blockCountPerDim[2])) {
        return nullptr;
    }
    
    // 计算邻居块的全局ID（row-major顺序）
    size_t nbrId = static_cast<size_t>(nbr_x) * blockCountPerDim[1] * blockCountPerDim[2] + 
                   static_cast<size_t>(nbr_y) * blockCountPerDim[2] + 
                   static_cast<size_t>(nbr_z);
    
    // 从映射中查找邻居块数据
    auto it = blockMap.find(nbrId);
    return (it != blockMap.end()) ? it->second : nullptr;
}

// ============================================================================
// 辅助函数2（double版）：处理内部cube
// ============================================================================
inline void processInternalCube(
    const double* blockData,
    size_t local_x, size_t local_y, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t base = local_x + local_y * Bx + local_z * Bx * By;
    
    std::array<double, 8> cubeValues = {{
        blockData[base],
        blockData[base + 1],
        blockData[base + 1 + Bx],
        blockData[base + Bx],
        blockData[base + Bx * By],
        blockData[base + 1 + Bx * By],
        blockData[base + 1 + Bx + Bx * By],
        blockData[base + Bx + Bx * By]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    std::array<std::array<double, 3>, 8> cubeGradients;
    for (int i = 0; i < 8; ++i) {
        size_t px = local_x + (i & 1);
        size_t py = local_y + ((i >> 1) & 1);
        size_t pz = local_z + ((i >> 2) & 1);
        
        size_t px_safe = std::min(px, Bx - 1);
        size_t py_safe = std::min(py, By - 1);
        size_t pz_safe = std::min(pz, Bx - 1);
        
        std::array<double, 3> grad = {0.0, 0.0, 0.0};
        
        if (px_safe == 0) {
            grad[0] = blockData[px_safe + py_safe*Bx + pz_safe*Bx*By] - 
                      blockData[(px_safe+1) + py_safe*Bx + pz_safe*Bx*By];
        } else if (px_safe == Bx - 1) {
            grad[0] = blockData[(px_safe-1) + py_safe*Bx + pz_safe*Bx*By] - 
                      blockData[px_safe + py_safe*Bx + pz_safe*Bx*By];
        } else {
            grad[0] = (blockData[(px_safe-1) + py_safe*Bx + pz_safe*Bx*By] - 
                       blockData[(px_safe+1) + py_safe*Bx + pz_safe*Bx*By]) / 2.0;
        }
        
        if (py_safe == 0) {
            grad[1] = blockData[px_safe + py_safe*Bx + pz_safe*Bx*By] - 
                      blockData[px_safe + (py_safe+1)*Bx + pz_safe*Bx*By];
        } else if (py_safe == By - 1) {
            grad[1] = blockData[px_safe + (py_safe-1)*Bx + pz_safe*Bx*By] - 
                      blockData[px_safe + py_safe*Bx + pz_safe*Bx*By];
        } else {
            grad[1] = (blockData[px_safe + (py_safe-1)*Bx + pz_safe*Bx*By] - 
                       blockData[px_safe + (py_safe+1)*Bx + pz_safe*Bx*By]) / 2.0;
        }
        
        if (pz_safe == 0) {
            grad[2] = blockData[px_safe + py_safe*Bx + pz_safe*Bx*By] - 
                      blockData[px_safe + py_safe*Bx + (pz_safe+1)*Bx*By];
        } else if (pz_safe == Bx - 1) {
            grad[2] = blockData[px_safe + py_safe*Bx + (pz_safe-1)*Bx*By] - 
                      blockData[px_safe + py_safe*Bx + pz_safe*Bx*By];
        } else {
            grad[2] = (blockData[px_safe + py_safe*Bx + (pz_safe-1)*Bx*By] - 
                       blockData[px_safe + py_safe*Bx + (pz_safe+1)*Bx*By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx = (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                
                double w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数3（double版）：处理X+边界cube
// ============================================================================
inline void processBoundaryXPlusCube(
    const double* blockData,
    const double* neighbor_xplus,
    size_t local_y, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr = 0 + local_y * Bx + local_z * Bx * By;
    
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],
        neighbor_xplus[base_nbr],
        neighbor_xplus[base_nbr + Bx],
        blockData[base_curr + Bx],
        blockData[base_curr + Bx * By],
        neighbor_xplus[base_nbr + Bx * By],
        neighbor_xplus[base_nbr + Bx + Bx * By],
        blockData[base_curr + Bx + Bx * By]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x,   local_y,   local_z,   false},
        {0,         local_y,   local_z,   true},
        {0,         local_y+1, local_z,   true},
        {local_x,   local_y+1, local_z,   false},
        {local_x,   local_y,   local_z+1, false},
        {0,         local_y,   local_z+1, true},
        {0,         local_y+1, local_z+1, true},
        {local_x,   local_y+1, local_z+1, false}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const double* dataPtr = vi.in_neighbor ? neighbor_xplus : blockData;
        
        std::array<double, 3> grad;
        
        if (vi.in_neighbor) {
            if (vi.px == 0) {
                double val_left  = blockData[local_x + vi.py * Bx + vi.pz * Bx * By];
                double val_right = (vi.px + 1 < Bx) ? 
                    neighbor_xplus[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By] : val_left;
                grad[0] = (val_left - val_right) / 2.0;
            } else {
                grad[0] = (neighbor_xplus[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                           neighbor_xplus[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
            }
        } else {
            if (vi.px == Bx - 1) {
                double val_left  = (vi.px > 0) ?
                    blockData[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] :
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                double val_right = neighbor_xplus[0 + vi.py * Bx + vi.pz * Bx * By];
                grad[0] = (val_left - val_right) / 2.0;
            } else {
                grad[0] = (blockData[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                           blockData[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
            }
        }
        
        if (vi.py == 0) {
            grad[1] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                       dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.pz == 0) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                       dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx = (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                
                double w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数4（double版）：处理Y+边界cube
// ============================================================================
inline void processBoundaryYPlusCube(
    const double* blockData,
    const double* neighbor_yplus,
    size_t local_x, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_y = By - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr  = local_x + 0       * Bx + local_z * Bx * By;
    
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],
        blockData[base_curr + 1],
        neighbor_yplus[base_nbr + 1],
        neighbor_yplus[base_nbr],
        blockData[base_curr + Bx * By],
        blockData[base_curr + 1 + Bx * By],
        neighbor_yplus[base_nbr + 1 + Bx * By],
        neighbor_yplus[base_nbr + Bx * By]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x,   local_y,   local_z,   false},
        {local_x+1, local_y,   local_z,   false},
        {local_x+1, 0,         local_z,   true},
        {local_x,   0,         local_z,   true},
        {local_x,   local_y,   local_z+1, false},
        {local_x+1, local_y,   local_z+1, false},
        {local_x+1, 0,         local_z+1, true},
        {local_x,   0,         local_z+1, true}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const double* dataPtr = vi.in_neighbor ? neighbor_yplus : blockData;
        
        std::array<double, 3> grad;
        
        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                       dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.in_neighbor) {
            if (vi.py == 0) {
                double val_left  = blockData[vi.px + local_y * Bx + vi.pz * Bx * By];
                double val_right = (vi.py + 1 < By) ? 
                    neighbor_yplus[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By] : val_left;
                grad[1] = (val_left - val_right) / 2.0;
            } else {
                grad[1] = (neighbor_yplus[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                           neighbor_yplus[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
            }
        } else {
            if (vi.py == By - 1) {
                double val_left  = (vi.py > 0) ? 
                    blockData[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] :
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                double val_right = neighbor_yplus[vi.px + 0 * Bx + vi.pz * Bx * By];
                grad[1] = (val_left - val_right) / 2.0;
            } else {
                grad[1] = (blockData[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                           blockData[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
            }
        }
        
        if (vi.pz == 0) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                       dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx = (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                
                double w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数5（double版）：处理Z+边界cube
// ============================================================================
inline void processBoundaryZPlusCube(
    const double* blockData,
    const double* neighbor_zplus,
    size_t local_x, size_t local_y,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_z = Bx - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr  = local_x + local_y * Bx + 0 * Bx * By;
    
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],
        blockData[base_curr + 1],
        blockData[base_curr + 1 + Bx],
        blockData[base_curr + Bx],
        neighbor_zplus[base_nbr],
        neighbor_zplus[base_nbr + 1],
        neighbor_zplus[base_nbr + 1 + Bx],
        neighbor_zplus[base_nbr + Bx]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x,   local_y,   local_z,   false},
        {local_x+1, local_y,   local_z,   false},
        {local_x+1, local_y+1, local_z,   false},
        {local_x,   local_y+1, local_z,   false},
        {local_x,   local_y,   0,         true},
        {local_x+1, local_y,   0,         true},
        {local_x+1, local_y+1, 0,         true},
        {local_x,   local_y+1, 0,         true}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const double* dataPtr = vi.in_neighbor ? neighbor_zplus : blockData;
        
        std::array<double, 3> grad;
        
        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                       dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.py == 0) {
            grad[1] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                       dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.in_neighbor) {
            if (vi.pz == 0) {
                double val_left  = blockData[vi.px + vi.py * Bx + local_z * Bx * By];
                double val_right = (vi.pz + 1 < Bx) ? 
                    neighbor_zplus[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By] : val_left;
                grad[2] = (val_left - val_right) / 2.0;
            } else {
                grad[2] = (neighbor_zplus[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                           neighbor_zplus[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
            }
        } else {
            if (vi.pz == Bx - 1) {
                double val_left  = (vi.pz > 0) ? 
                    blockData[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] :
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                double val_right = neighbor_zplus[vi.px + vi.py * Bx + 0 * Bx * By];
                grad[2] = (val_left - val_right) / 2.0;
            } else {
                grad[2] = (blockData[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                           blockData[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
            }
        }
        
        cubeGradients[i] = grad;
    }
    
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx = (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                
                double w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}


// ============================================================================
// 辅助函数6（double版）：处理XY边棱cube
// ============================================================================
inline void processEdgeXYCube(
    const double* blockData,
    const double* neighbor_xplus,
    const double* neighbor_yplus,
    const double* neighbor_xy_diagonal,
    size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;
    size_t local_y = By - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_xp = 0 + local_y * Bx + local_z * Bx * By;
    size_t base_yp = local_x + 0 * Bx + local_z * Bx * By;
    size_t base_xy = 0 + 0 * Bx + local_z * Bx * By;
    
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],
        neighbor_xplus[base_xp],
        neighbor_xy_diagonal[base_xy],
        neighbor_yplus[base_yp],
        blockData[base_curr + Bx * By],
        neighbor_xplus[base_xp + Bx * By],
        neighbor_xy_diagonal[base_xy + Bx * By],
        neighbor_yplus[base_yp + Bx * By]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        const double* dataPtr;
        const double* neighbor_x;
        const double* neighbor_y;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x, local_y, local_z, blockData,          neighbor_xplus,      neighbor_yplus},
        {0,       local_y, local_z, neighbor_xplus,     blockData,           neighbor_xy_diagonal},
        {0,       0,       local_z, neighbor_xy_diagonal, neighbor_yplus,   neighbor_xplus},
        {local_x, 0,       local_z, neighbor_yplus,     neighbor_xy_diagonal, blockData},
        {local_x, local_y, local_z+1, blockData,        neighbor_xplus,      neighbor_yplus},
        {0,       local_y, local_z+1, neighbor_xplus,   blockData,           neighbor_xy_diagonal},
        {0,       0,       local_z+1, neighbor_xy_diagonal, neighbor_yplus, neighbor_xplus},
        {local_x, 0,       local_z+1, neighbor_yplus,   neighbor_xy_diagonal, blockData}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<double, 3> grad;
        
        if (vi.px == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            double val_right = (vi.px + 1 < Bx)
                ? vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[0] = (val_left - val_right) / 2.0;
        } else if (vi.px == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = (vi.px > 0)
                ? vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0;
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.py == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            double val_up   = (vi.py + 1 < By)
                ? vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[1] = (val_down - val_up) / 2.0;
        } else if (vi.py == By - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = (vi.py > 0)
                ? vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_up   = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0;
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.pz == 0) {
            grad[2] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数7（double版）：处理XZ边棱cube
// ============================================================================
inline void processEdgeXZCube(
    const double* blockData,
    const double* neighbor_xplus,
    const double* neighbor_zplus,
    const double* neighbor_xz_diagonal,
    size_t local_y,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;
    size_t local_z = Bx - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_xp   = 0       + local_y * Bx + local_z * Bx * By;
    size_t base_zp   = local_x + local_y * Bx + 0 * Bx * By;
    size_t base_xz   = 0       + local_y * Bx + 0 * Bx * By;
    
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],
        neighbor_xplus[base_xp],
        neighbor_xplus[base_xp + Bx],
        blockData[base_curr + Bx],
        neighbor_zplus[base_zp],
        neighbor_xz_diagonal[base_xz],
        neighbor_xz_diagonal[base_xz + Bx],
        neighbor_zplus[base_zp + Bx]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        const double* dataPtr;
        const double* neighbor_x;
        const double* neighbor_z;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x, local_y, local_z, blockData,          neighbor_xplus,    neighbor_zplus},
        {0,       local_y, local_z, neighbor_xplus,     blockData,         neighbor_xz_diagonal},
        {0,       local_y+1, local_z, neighbor_xplus,   blockData,         neighbor_xz_diagonal},
        {local_x, local_y+1, local_z, blockData,        neighbor_xplus,    neighbor_zplus},
        {local_x, local_y,   0,       neighbor_zplus,   neighbor_xz_diagonal, blockData},
        {0,       local_y,   0,       neighbor_xz_diagonal, neighbor_zplus, neighbor_xplus},
        {0,       local_y+1, 0,       neighbor_xz_diagonal, neighbor_zplus, neighbor_xplus},
        {local_x, local_y+1, 0,       neighbor_zplus,   neighbor_xz_diagonal, blockData}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<double, 3> grad;
        
        if (vi.px == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            double val_right = (vi.px + 1 < Bx)
                ? vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[0] = (val_left - val_right) / 2.0;
        } else if (vi.px == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = (vi.px > 0)
                ? vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0;
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.py == 0) {
            grad[1] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.pz == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            double val_front = (vi.pz + 1 < Bx)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]
                : val_curr;
            grad[2] = (val_back - val_front) / 2.0;
        } else if (vi.pz == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = (vi.pz > 0)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By]
                : val_curr;
            double val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数8（double版）：处理YZ边棱cube
// ============================================================================
inline void processEdgeYZCube(
    const double* blockData,
    const double* neighbor_yplus,
    const double* neighbor_zplus,
    const double* neighbor_yz_diagonal,
    size_t local_x,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_y = By - 1;
    size_t local_z = Bx - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_yp   = local_x + 0 * Bx       + local_z * Bx * By;
    size_t base_zp   = local_x + local_y * Bx + 0 * Bx * By;
    size_t base_yz   = local_x + 0 * Bx       + 0 * Bx * By;
    
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],
        blockData[base_curr + 1],
        neighbor_yplus[base_yp + 1],
        neighbor_yplus[base_yp],
        neighbor_zplus[base_zp],
        neighbor_zplus[base_zp + 1],
        neighbor_yz_diagonal[base_yz + 1],
        neighbor_yz_diagonal[base_yz]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        const double* dataPtr;
        const double* neighbor_y;
        const double* neighbor_z;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x,   local_y,   local_z,   blockData,        neighbor_yplus,   neighbor_zplus},
        {local_x+1, local_y,   local_z,   blockData,        neighbor_yplus,   neighbor_zplus},
        {local_x+1, 0,         local_z,   neighbor_yplus,   blockData,        neighbor_yz_diagonal},
        {local_x,   0,         local_z,   neighbor_yplus,   blockData,        neighbor_yz_diagonal},
        {local_x,   local_y,   0,         neighbor_zplus,   neighbor_yz_diagonal, blockData},
        {local_x+1, local_y,   0,         neighbor_zplus,   neighbor_yz_diagonal, blockData},
        {local_x+1, 0,         0,         neighbor_yz_diagonal, neighbor_zplus, neighbor_yplus},
        {local_x,   0,         0,         neighbor_yz_diagonal, neighbor_zplus, neighbor_yplus}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<double, 3> grad;
        
        if (vi.px == 0) {
            grad[0] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.py == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            double val_up   = (vi.py + 1 < By)
                ? vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[1] = (val_down - val_up) / 2.0;
        } else if (vi.py == By - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = (vi.py > 0)
                ? vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_up   = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0;
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.pz == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            double val_front = (vi.pz + 1 < Bx)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]
                : val_curr;
            grad[2] = (val_back - val_front) / 2.0;
        } else if (vi.pz == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = (vi.pz > 0)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By]
                : val_curr;
            double val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数9（double版）：处理XYZ角点cube
// ============================================================================
inline void processCornerXYZCube(
    const double* blockData,
    const double* neighbor_xplus,
    const double* neighbor_yplus,
    const double* neighbor_zplus,
    const double* neighbor_xy_diagonal,
    const double* neighbor_xz_diagonal,
    const double* neighbor_yz_diagonal,
    const double* neighbor_xyz_diagonal,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;
    size_t local_y = By - 1;
    size_t local_z = Bx - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_xp   = 0       + local_y * Bx + local_z * Bx * By;
    size_t base_yp   = local_x + 0       * Bx + local_z * Bx * By;
    size_t base_zp   = local_x + local_y * Bx + 0       * Bx * By;
    size_t base_xy   = 0       + 0       * Bx + local_z * Bx * By;
    size_t base_xz   = 0       + local_y * Bx + 0       * Bx * By;
    size_t base_yz   = local_x + 0       * Bx + 0       * Bx * By;
    size_t base_xyz  = 0       + 0       * Bx + 0       * Bx * By;
    
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],
        neighbor_xplus[base_xp],
        neighbor_xy_diagonal[base_xy],
        neighbor_yplus[base_yp],
        neighbor_zplus[base_zp],
        neighbor_xz_diagonal[base_xz],
        neighbor_xyz_diagonal[base_xyz],
        neighbor_yz_diagonal[base_yz]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        const double* dataPtr;
        const double* neighbor_x;
        const double* neighbor_y;
        const double* neighbor_z;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x, local_y, local_z, blockData,
         neighbor_xplus, neighbor_yplus, neighbor_zplus},
        
        {0,       local_y, local_z, neighbor_xplus,
         blockData, neighbor_xy_diagonal, neighbor_xz_diagonal},
        
        {0, 0, local_z, neighbor_xy_diagonal,
         neighbor_yplus, neighbor_xplus, neighbor_xyz_diagonal},
        
        {local_x, 0, local_z, neighbor_yplus,
         neighbor_xy_diagonal, blockData, neighbor_yz_diagonal},
        
        {local_x, local_y, 0, neighbor_zplus,
         neighbor_xz_diagonal, neighbor_yz_diagonal, blockData},
        
        {0, local_y, 0, neighbor_xz_diagonal,
         neighbor_zplus, neighbor_xyz_diagonal, neighbor_xplus},
        
        {0, 0, 0, neighbor_xyz_diagonal,
         neighbor_yz_diagonal, neighbor_xz_diagonal, neighbor_xy_diagonal},
        
        {local_x, 0, 0, neighbor_yz_diagonal,
         neighbor_xyz_diagonal, neighbor_zplus, neighbor_yplus}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<double, 3> grad;
        
        if (vi.px == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            double val_right = (vi.px + 1 < Bx)
                ? vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[0] = (val_left - val_right) / 2.0;
        } else if (vi.px == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = (vi.px > 0)
                ? vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0;
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.py == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            double val_up   = (vi.py + 1 < By)
                ? vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[1] = (val_down - val_up) / 2.0;
        } else if (vi.py == By - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = (vi.py > 0)
                ? vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_up   = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0;
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        if (vi.pz == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            double val_front = (vi.pz + 1 < Bx)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]
                : val_curr;
            grad[2] = (val_back - val_front) / 2.0;
        } else if (vi.pz == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = (vi.pz > 0)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By]
                : val_curr;
            double val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}


// ============================================================================
// 主函数：RunMarchingCubesOnDecompressedBlocks （double 版本）
// 分区域处理：内部cube（快速路径）+ 面/棱/角边界（需要邻居块指针）
//
// 说明：本简化版不再做26邻居预扩展解压。blockMap 里只包含"索引命中的核心块"。
// 因此 getNeighborBlockPointer 在邻居块未被解压时会返回 nullptr，
// 对应的边界区域（面/棱/角）会被自动跳过，不生成三角形 —— 这就是块与块
// 交界处可能出现缝隙/空洞的原因，是本次简化后接受的代价。
// 如果两个相邻块都恰好被索引命中（都在 blockMap 中），则它们之间的边界
// 仍然会被正确处理，不会有缝隙。
// ============================================================================
util::TriangleMesh<double> RunMarchingCubesOnDecompressedBlocks(
    const std::vector<std::vector<double>>& decompressedBlocks,
    const std::vector<size_t>& coreGlobalSmallBlockIds,
    double isovalue,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& dataShape)
{
    // ========== 初始化：计算块的布局信息 ==========
    std::vector<size_t> blockCountOnEachDim(3);
    for (size_t i = 0; i < 3; i++) {
        blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];
    }

    size_t Bx = blockShape[0];
    size_t By = blockShape[1];
    size_t Bz = blockShape[2];

    // ========== 步骤1：建立块ID到数据指针的映射（一次性，O(n)）==========
    // 只包含核心命中块（没有26邻居扩展）
    std::unordered_map<size_t, const double*> blockMap;
    blockMap.reserve(coreGlobalSmallBlockIds.size());
    for (size_t i = 0; i < coreGlobalSmallBlockIds.size(); i++) {
        blockMap[coreGlobalSmallBlockIds[i]] = decompressedBlocks[i].data();
    }

    // ========== 步骤2：排序块ID以提升Cache命中率 ==========
    std::vector<size_t> sortedBlockIds = coreGlobalSmallBlockIds;
    std::sort(sortedBlockIds.begin(), sortedBlockIds.end());

    // ========== 全局结果容器 ==========
    std::vector<std::array<double, 3>> globalPoints;
    std::vector<std::array<double, 3>> globalNormals;
    std::vector<std::array<int, 3>> globalTriangles;

    size_t estimatedPoints = decompressedBlocks.size() * (Bx-1) * (By-1) * (Bz-1) * 6;
    globalPoints.reserve(estimatedPoints);
    globalNormals.reserve(estimatedPoints);
    globalTriangles.reserve(estimatedPoints / 3);

    int totalVertexOffset = 0;

    // ========== 步骤3：遍历每个核心块 ==========
    for (size_t globalBlockId : sortedBlockIds) {
        
        const double* blockData = blockMap[globalBlockId];
        
        size_t block_z = globalBlockId % blockCountOnEachDim[2];
        size_t remaining = globalBlockId / blockCountOnEachDim[2];
        size_t block_y = remaining % blockCountOnEachDim[1];
        size_t block_x = remaining / blockCountOnEachDim[1];

        size_t global_x_start = block_x * Bx;
        size_t global_y_start = block_y * By;
        size_t global_z_start = block_z * Bz;

        // 预获取邻居块指针（若邻居没被解压，返回nullptr，对应边界区域会被跳过）
        const double* neighbor_xplus = getNeighborBlockPointer(globalBlockId, 1, 0, 0, blockMap, blockCountOnEachDim);
        const double* neighbor_yplus = getNeighborBlockPointer(globalBlockId, 0, 1, 0, blockMap, blockCountOnEachDim);
        const double* neighbor_zplus = getNeighborBlockPointer(globalBlockId, 0, 0, 1, blockMap, blockCountOnEachDim);

        size_t max_x = std::min(Bx, dataShape[0] - global_x_start);
        size_t max_y = std::min(By, dataShape[1] - global_y_start);
        size_t max_z = std::min(Bz, dataShape[2] - global_z_start);

        std::vector<std::array<double, 3>> localPoints;
        std::vector<std::array<double, 3>> localNormals;
        std::vector<std::array<int, 3>> localTriangles;
        std::unordered_map<size_t, int> localPointMap;
        int localPtIdx = 0;

        // ========== 区域1：内部cube（占绝大多数工作量）==========
        size_t internal_max_x = (max_x > 1) ? (max_x - 1) : 0;
        size_t internal_max_y = (max_y > 1) ? (max_y - 1) : 0;
        size_t internal_max_z = (max_z > 1) ? (max_z - 1) : 0;

        for (size_t lz = 0; lz < internal_max_z; ++lz) {
            for (size_t ly = 0; ly < internal_max_y; ++ly) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + lz;

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processInternalCube(blockData, lx, ly, lz, Bx, By, gx, gy, gz,
                                       isovalue, localPoints, localNormals, 
                                       localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域2：X+ 面边界（仅当邻居块也被解压时才处理）==========
        if (neighbor_xplus != nullptr && max_x == Bx) {
            for (size_t lz = 0; lz < internal_max_z; ++lz) {
                for (size_t ly = 0; ly < internal_max_y; ++ly) {
                    
                    size_t gx = global_x_start + (Bx - 1);
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + lz;

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processBoundaryXPlusCube(blockData, neighbor_xplus, ly, lz, Bx, By,
                                            gx, gy, gz, isovalue, localPoints, localNormals,
                                            localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域3：Y+ 面边界 ==========
        if (neighbor_yplus != nullptr && max_y == By) {
            for (size_t lz = 0; lz < internal_max_z; ++lz) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + (By - 1);
                    size_t gz = global_z_start + lz;

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processBoundaryYPlusCube(blockData, neighbor_yplus, lx, lz, Bx, By,
                                            gx, gy, gz, isovalue, localPoints, localNormals,
                                            localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域4：Z+ 面边界 ==========
        if (neighbor_zplus != nullptr && max_z == Bz) {
            for (size_t ly = 0; ly < internal_max_y; ++ly) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + (Bz - 1);

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processBoundaryZPlusCube(blockData, neighbor_zplus, lx, ly, Bx, By,
                                            gx, gy, gz, isovalue, localPoints, localNormals,
                                            localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域5：XY边棱 ==========
        if (neighbor_xplus != nullptr && neighbor_yplus != nullptr && 
            max_x == Bx && max_y == By) {
            
            const double* neighbor_xy = getNeighborBlockPointer(globalBlockId, 1, 1, 0, 
                                                               blockMap, blockCountOnEachDim);
            
            if (neighbor_xy != nullptr) {
                for (size_t lz = 0; lz < internal_max_z; ++lz) {
                    
                    size_t gx = global_x_start + (Bx - 1);
                    size_t gy = global_y_start + (By - 1);
                    size_t gz = global_z_start + lz;

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processEdgeXYCube(blockData, neighbor_xplus, neighbor_yplus, neighbor_xy,
                                     lz, Bx, By, gx, gy, gz, isovalue,
                                     localPoints, localNormals, localTriangles, 
                                     localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域6：XZ边棱 ==========
        if (neighbor_xplus != nullptr && neighbor_zplus != nullptr && 
            max_x == Bx && max_z == Bz) {
            
            const double* neighbor_xz = getNeighborBlockPointer(globalBlockId, 1, 0, 1, 
                                                               blockMap, blockCountOnEachDim);
            
            if (neighbor_xz != nullptr) {
                for (size_t ly = 0; ly < internal_max_y; ++ly) {
                    
                    size_t gx = global_x_start + (Bx - 1);
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + (Bz - 1);

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processEdgeXZCube(blockData, neighbor_xplus, neighbor_zplus, neighbor_xz,
                                     ly, Bx, By, gx, gy, gz, isovalue,
                                     localPoints, localNormals, localTriangles, 
                                     localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域7：YZ边棱 ==========
        if (neighbor_yplus != nullptr && neighbor_zplus != nullptr && 
            max_y == By && max_z == Bz) {
            
            const double* neighbor_yz = getNeighborBlockPointer(globalBlockId, 0, 1, 1, 
                                                               blockMap, blockCountOnEachDim);
            
            if (neighbor_yz != nullptr) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + (By - 1);
                    size_t gz = global_z_start + (Bz - 1);

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processEdgeYZCube(blockData, neighbor_yplus, neighbor_zplus, neighbor_yz,
                                     lx, Bx, By, gx, gy, gz, isovalue,
                                     localPoints, localNormals, localTriangles, 
                                     localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域8：XYZ角点 ==========
        if (neighbor_xplus != nullptr && neighbor_yplus != nullptr && neighbor_zplus != nullptr &&
            max_x == Bx && max_y == By && max_z == Bz) {
            
            const double* neighbor_xy = getNeighborBlockPointer(globalBlockId, 1, 1, 0, 
                                                               blockMap, blockCountOnEachDim);
            const double* neighbor_xz = getNeighborBlockPointer(globalBlockId, 1, 0, 1, 
                                                               blockMap, blockCountOnEachDim);
            const double* neighbor_yz = getNeighborBlockPointer(globalBlockId, 0, 1, 1, 
                                                               blockMap, blockCountOnEachDim);
            const double* neighbor_xyz = getNeighborBlockPointer(globalBlockId, 1, 1, 1, 
                                                                blockMap, blockCountOnEachDim);
            
            if (neighbor_xy != nullptr && neighbor_xz != nullptr && 
                neighbor_yz != nullptr && neighbor_xyz != nullptr) {
                
                size_t gx = global_x_start + (Bx - 1);
                size_t gy = global_y_start + (By - 1);
                size_t gz = global_z_start + (Bz - 1);

                if (gx + 1 < dataShape[0] && gy + 1 < dataShape[1] && gz + 1 < dataShape[2]) {
                    processCornerXYZCube(blockData, neighbor_xplus, neighbor_yplus, neighbor_zplus,
                                        neighbor_xy, neighbor_xz, neighbor_yz, neighbor_xyz,
                                        Bx, By, gx, gy, gz, isovalue,
                                        localPoints, localNormals, localTriangles, 
                                        localPointMap, localPtIdx);
                }
            }
        }

        // === 合并局部结果到全局 ===
        globalPoints.insert(globalPoints.end(), localPoints.begin(), localPoints.end());
        globalNormals.insert(globalNormals.end(), localNormals.begin(), localNormals.end());

        for (const auto& tri : localTriangles) {
            std::array<int, 3> adjustedTri = {{
                tri[0] + totalVertexOffset,
                tri[1] + totalVertexOffset,
                tri[2] + totalVertexOffset
            }};
            globalTriangles.push_back(adjustedTri);
        }

        if (localPoints.size() > std::numeric_limits<int>::max() - totalVertexOffset) {
            throw std::runtime_error("Too many vertices for int indexing");
        }
        
        totalVertexOffset += static_cast<int>(localPoints.size());
    }

    return util::TriangleMesh<double>(globalPoints, globalNormals, globalTriangles);
}


void RunAndSaveIsosurfaceMesh(
    const std::vector<std::vector<double>>& decompressedBlocks,  
    const std::vector<size_t>& coreGlobalSmallBlockIds,
    double isovalue,  
    const std::vector<size_t>& smallBlockShape,
    const std::vector<size_t>& stepDataShape,
    const std::string& outFile
) {
    std::cout << "[Info] Running Marching Cubes on decompressed blocks\n";
    std::cout << "       Volume shape: (" << stepDataShape[0] << ", " << stepDataShape[1] << ", " << stepDataShape[2] << ")\n";
    std::cout << "       Block shape:  (" << smallBlockShape[0] << ", " << smallBlockShape[1] << ", " << smallBlockShape[2] << ")\n";
    std::cout << "       Isovalue:     " << isovalue << "\n";
    std::cout << "       Output file:  " << outFile << "\n";

    util::Timer timer;
    timer.start();

    util::TriangleMesh<double> mesh = RunMarchingCubesOnDecompressedBlocks(  
        decompressedBlocks,
        coreGlobalSmallBlockIds,
        isovalue,
        smallBlockShape,
        stepDataShape 
    );

    timer.stop();

    std::cout << "[Result] Mesh vertices:  " << mesh.numberOfVertices() << "\n";
    std::cout << "[Result] Mesh triangles: " << mesh.numberOfTriangles() << "\n";
    std::cout << "[Timing] CPU time (sec):  " << timer.getCPUtime() << "\n";
    std::cout << "[Timing] Wall time (sec): " << timer.getWallTime() << "\n";

    util::saveTriangleMesh(mesh, outFile.c_str());
}


int main(int argc, char* argv[]) {

    auto totalStart = std::chrono::high_resolution_clock::now(); 

    MPI_Init(&argc, &argv);

    int mpi_rank = 0;
    int mpi_size = 1;

    std::string inputFileName, variableName, indexDir;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, blockShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0, smallBlockSize = 1, totalBlocksNumber = 1; 
    std::vector<double> queryRange;
    double relative_error_bound = 1E-3;
    size_t extraValue = 0;

    // **解析命令行参数**
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc) {
            inputFileName = argv[++i];
        } else if (arg == "--variable_name" && i + 1 < argc) {
            variableName = argv[++i];
        } else if (arg == "--dimensions" && i + 1 < argc)
        {
            nDim = std::stoul(argv[++i]);
        }
        else if (arg == "--begin_step" && i + 1 < argc) {
            beginStepNum = std::stoul(argv[++i]);
        } else if (arg == "--end_step" && i + 1 < argc) {
            endStepNum = std::stoul(argv[++i]);
        } 

        else if (arg == "--stepData_shape" && i + nDim < argc)
        {
            for (size_t j = 0; j < nDim; j++)
            {
                stepDataShape.push_back(std::stoul(argv[++i]));
            }
        }

        else if (arg == "--bigBlock_shape" && i + nDim < argc)
        {
            // 保留此参数解析以兼容旧脚本，实际不再使用大块层
            for (size_t j = 0; j < nDim; j++)
            {
                blockShape.push_back(std::stoul(argv[++i]));
            }
        }
        else if (arg == "--small_block_shape")
        {
            if (nDim)
            {
                if ((int)(i + nDim) < argc)
                {
                    for (size_t j = i + 1; j < i + 1 + nDim; j++)
                    {
                        smallBlockShape.push_back(atoi(argv[j]));
                    }
                }
                else
                {
                    std::cerr << "--small_block_shape option requires [# of dimensions] argument." << std::endl;
                    return 1;
                }
            }
        }else if (arg == "--query_range" && i + 2 < argc) {
            queryRange.push_back(std::stod(argv[++i]));
            queryRange.push_back(std::stod(argv[++i]));
        }else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        } 
    }

    // === 去掉大块层，直接用 smallBlockShape 计算分块数量 ===
    size_t smallBlocksPerStep = 1;
    for (size_t d = 0; d < nDim; d++) {
        smallBlocksPerStep *= stepDataShape[d] / smallBlockShape[d];
        smallBlockSize     *= smallBlockShape[d];
    }
    size_t nSteps = endStepNum - beginStepNum + 1;
    totalBlocksNumber = nSteps * smallBlocksPerStep;

    if (queryRange.size() != 2) {
        std::cerr << "Error: Invalid query range. Use --query_range <low> <high>" << std::endl;
        return 1;
    }

    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    // === 只保留 uniform 索引目录 ===
    indexDir = "/expanse/lustre/scratch/sdi/temp_project/nyx_index_compress/" + inputFileBaseName + "_" + safeVarName + "_"
             + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
             + std::to_string(extraValue) + "_newour_stagger1_index/";

    std::string mySubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + safeVarName + "_"
             + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
             + std::to_string(extraValue) + "_blockZFP_originalDataCompression/";

    auto afterProcessTime = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> preProcessTime = afterProcessTime - totalStart;
    std::cerr << "[Rank " << mpi_rank << "[Time1] : preProcessTime: " << preProcessTime.count() << " seconds" << std::endl;

    // === 只加载 uniform 索引的 big_block_minmax（step级粗筛）===
    std::vector<ScidxInterval<double>> allBigBlockIndices = loadBigBlockIndexFile(indexDir + "big_block_minmax");

    std::cout << "[Debug] uniform big_block_minmax entries: " << allBigBlockIndices.size() << "\n";

    double globalMin = std::numeric_limits<double>::max();
    double globalMax = std::numeric_limits<double>::lowest();

    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }

    std::cout << "Global Min: " << globalMin << std::endl;
    std::cout << "Global Max: " << globalMax << std::endl;

    double error_bound = relative_error_bound * (globalMax - globalMin);
    std::cout << "Computed error_bound: " << error_bound << std::endl;

    double isovalue = queryRange[0];

    // === 只筛 uniform 命中的 step（不再取 stagger 并集）===
    std::vector<size_t> selectedSteps;
    for (size_t i = 0; i < allBigBlockIndices.size(); i++) {
        if (isovalue <= allBigBlockIndices[i].high && isovalue >= allBigBlockIndices[i].low) {
            selectedSteps.push_back(i);
        }
    }
    std::cout << "Selected Steps (uniform only): " << selectedSteps.size() << std::endl;

    auto afterQueryBigBlock = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> readAndQueryBigBlock = afterQueryBigBlock - afterProcessTime;
    std::cerr << "[Rank " << mpi_rank << "[Time2] : read and query bigBlocks: " << readAndQueryBigBlock.count() << " seconds" << std::endl;

    // === 只查 uniform 索引，得到核心命中块（去掉 stagger 查询 + 坐标转换）===
    std::unordered_set<size_t> hitUniformBlocksSet;
    for (size_t stepRelIdx : selectedSteps) {
        size_t actualStepNum = beginStepNum + stepRelIdx;
        size_t stepOffset    = stepRelIdx * smallBlocksPerStep;

        std::vector<size_t> uniformRawIds =
            queryIndexRawIds(actualStepNum, indexDir, queryRange, error_bound);

        auto time_before_merge = std::chrono::high_resolution_clock::now();

        for (size_t rawId : uniformRawIds)
            if (rawId < smallBlocksPerStep)
                hitUniformBlocksSet.insert(stepOffset + rawId);

        auto time_after_merge = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> mergeTime = time_after_merge - time_before_merge;
        std::cerr << "[Time_merge] insert + dedup: "
                << mergeTime.count() << " seconds" << std::endl;
    }

    auto time_before_sort = std::chrono::high_resolution_clock::now();

    std::vector<size_t> localGlobalSmallBlockIds(
        hitUniformBlocksSet.begin(), hitUniformBlocksSet.end());
    std::sort(localGlobalSmallBlockIds.begin(), localGlobalSmallBlockIds.end());

    auto time_after_sort = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sortTime = time_after_sort - time_before_sort;
    std::cerr << "[Time_sort] set->vector + sort: "
            << sortTime.count() << " seconds" << std::endl;

    std::cerr << "[CHECK] Uniform hit blocks (deduped): "
            << localGlobalSmallBlockIds.size() << std::endl;

    for (size_t bid : localGlobalSmallBlockIds) {
        if (bid >= totalBlocksNumber) {
            std::cerr << "[BUG] bid " << bid << " >= totalBlocksNumber = " << totalBlocksNumber << std::endl;
            exit(1);
        }
    }

    // ============================================================
    // 说明：已去掉 26 邻居扩展逻辑。
    // 直接用 localGlobalSmallBlockIds（uniform索引命中的核心块）去解压，
    // 不再额外解压周围邻居块。代价：块与块交界处的cube如果两侧不是
    // 同时命中，会缺少邻居数据而被跳过，mesh可能在块边界处出现缝隙。
    // ============================================================

    // === 直接解压核心命中块 ===
    std::vector<std::vector<double>> decompressedTargetOriginalBlockdata =
        universalBatchDecompressBlocks(
            mySubDir,
            localGlobalSmallBlockIds,
            smallBlockSize,
            totalBlocksNumber,
            error_bound
        );

    auto finishDecompress = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> decompressTime = finishDecompress - time_after_sort;
    std::cerr << "[Time_decompress] decompress core blocks: " << decompressTime.count() << " seconds" << std::endl;

    // ============================================================
    // 为每个step分别提取等值面
    // ============================================================

    std::cout << "\n========================================" << std::endl;
    std::cout << "Extracting isosurfaces for " << nSteps << " steps..." << std::endl;
    std::cout << "========================================\n" << std::endl;

    double ratio_block = (double)decompressedTargetOriginalBlockdata.size() / smallBlocksPerStep * 100.0;

    std::cout << "[Info] Blocks per step: " << smallBlocksPerStep << std::endl;
    std::cout << "[Info] Total decompressed blocks: " << decompressedTargetOriginalBlockdata.size() << std::endl;
    std::cout << "[Info] Decompression ratio: " << ratio_block << "%" << std::endl;

    for (size_t step = 0; step < nSteps; ++step) {
        auto stepStart = std::chrono::high_resolution_clock::now();
        
        size_t currentStepNum = beginStepNum + step;
        size_t stepOffset = step * smallBlocksPerStep;
        
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Processing Step " << currentStepNum << std::endl;
        
        // === 提取当前step的核心块ID ===
        std::vector<size_t> stepCoreBlockIds;
        for (size_t id : localGlobalSmallBlockIds) {
            if (id >= stepOffset && id < stepOffset + smallBlocksPerStep) {
                stepCoreBlockIds.push_back(id);
            }
        }
        
        std::cout << "[Info] Step " << currentStepNum << " core blocks: " 
                << stepCoreBlockIds.size() << std::endl;
        
        // === 提取当前step的解压数据 ===
        std::vector<std::vector<double>> stepBlocks;
        stepBlocks.reserve(stepCoreBlockIds.size());
        
        std::unordered_map<size_t, size_t> blockIdToIndex;
        for (size_t i = 0; i < localGlobalSmallBlockIds.size(); ++i) {
            blockIdToIndex[localGlobalSmallBlockIds[i]] = i;
        }
        
        for (size_t blockId : stepCoreBlockIds) {
            auto it = blockIdToIndex.find(blockId);
            if (it != blockIdToIndex.end()) {
                stepBlocks.push_back(decompressedTargetOriginalBlockdata[it->second]);
            } else {
                std::cerr << "Error: Block " << blockId << " not found in decompressed data!" << std::endl;
            }
        }
        
        std::cout << "[Info] Extracted " << stepBlocks.size() << " blocks for Step " 
                << currentStepNum << std::endl;
        
        // === 生成输出文件名 ===
        std::string outFile = "/expanse/lustre/scratch/sdi/temp_project/iso_results/isosurface_step_" 
            + std::to_string(currentStepNum) 
            + "_iso_" + std::to_string(queryRange[0]) + "_uniformOnly.vtk";

        double iso_value = queryRange[0];

        // === 执行Marching Cubes（只用核心块，无邻居扩展）===
        RunAndSaveIsosurfaceMesh(
            stepBlocks,
            stepCoreBlockIds,
            iso_value,
            smallBlockShape,
            stepDataShape,
            outFile
        );
        
        auto stepEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> stepTime = stepEnd - stepStart;
        
        std::cout << "Step " << currentStepNum << " completed in " 
                << stepTime.count() << " seconds" << std::endl;
        std::cout << "   Output: " << outFile << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "All " << nSteps << " isosurfaces generated!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    MPI_Finalize();
    return 0;
}
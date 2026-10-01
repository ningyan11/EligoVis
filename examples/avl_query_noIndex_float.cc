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
#include <mpi.h> 
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
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <array>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>


// 辅助函数：将位置转换为多维索引
std::vector<size_t> positionToIndices(size_t position, const std::vector<size_t>& shape) {
    std::vector<size_t> indices;
    size_t remainingPosition = position;
    
    for (auto dimensionSize = shape.rbegin(); dimensionSize != shape.rend(); ++dimensionSize) {
        indices.insert(indices.begin(), remainingPosition % *dimensionSize);
        remainingPosition /= *dimensionSize;
    }
    
    return indices;
}

// 辅助函数：将多维索引转换为位置
size_t indicesToPosition(const std::vector<size_t>& shape, const std::vector<size_t>& indices) {
    size_t position = indices[0];
    size_t multiplier = 1;
    
    for (size_t i = 1; i < shape.size(); ++i) {
        multiplier *= shape[i - 1];
        position += indices[i] * multiplier;
    }
    
    return position;
}



void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
}


std::vector<ScidxInterval<float>> loadBigBlockIndexFile(const std::string& filename) {
    std::vector<ScidxInterval<float>> bigBlockIndices;

    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return bigBlockIndices;  // 返回空
    }

    float minVal, maxVal;
    while (inFile.read(reinterpret_cast<char*>(&minVal), sizeof(float)) &&
           inFile.read(reinterpret_cast<char*>(&maxVal), sizeof(float))) {
        bigBlockIndices.push_back({minVal, maxVal});
    }

    inFile.close();
    return bigBlockIndices;
}



std::vector<std::vector<float>> batchDecompressBlocksAllRead(const std::string& subDir,
    const std::vector<size_t>& blockIds,
    size_t blockSize,
    size_t totalBlocks,
    float error_bound) {

    auto start_time_read = std::chrono::high_resolution_clock::now();      
    std::vector<std::vector<float>> decompressedOriginalresult;

    //std::cout << "[Debug] Number of blocks to decompress: " << blockIds.size() << std::endl;
    //std::cout << "[Debug] subDir: " << subDir << std::endl;


    std::string unpredDataFileName = subDir + "fixed_unpredData.bin";
    std::string compressedFileName = subDir + "fixed_compressed_data.bin";
    std::string signFileName = subDir + "fixed_sign_data.bin";


    // 打开文件流
    std::ifstream unpredStream(unpredDataFileName, std::ios::binary);
    std::ifstream compStream(compressedFileName, std::ios::binary);
    std::ifstream signStream(signFileName, std::ios::binary);

    if (!unpredStream || !compStream || !signStream) {
        throw std::runtime_error("Failed to open one or more input files.");
    }

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

    // 分配足够大的内存
    std::vector<unsigned char> allUnpredData(unpredFileSize);
    std::vector<unsigned char> allCompData(compFileSize);
    std::vector<unsigned char> allSignData(signFileSize);

    // 读取文件数据到内存
    unpredStream.read(reinterpret_cast<char*>(allUnpredData.data()), unpredFileSize);
    compStream.read(reinterpret_cast<char*>(allCompData.data()), compFileSize);
    signStream.read(reinterpret_cast<char*>(allSignData.data()), signFileSize);

    //std::cout << "[Debug] after read all: " << std::endl;


    // 读取 unpredSizes 数据 (文件的头部数据)
    std::vector<unsigned char> unpredSizes(totalBlocks);
    std::memcpy(unpredSizes.data(), allUnpredData.data(), totalBlocks);

    // 分配 bitCounts 和 compSizes
    std::vector<unsigned char> bitCounts(totalBlocks);
    std::vector<unsigned char> compSizes(totalBlocks);

    // 将 allCompData 中的前 totalBlocks 个字节分配给 bitCounts
    std::memcpy(bitCounts.data(), allCompData.data(), totalBlocks);

    // 将 allCompData 中的接下来的 totalBlocks 个字节分配给 compSizes
    std::memcpy(compSizes.data(), allCompData.data() + totalBlocks, totalBlocks);

    const size_t signBytesPerBlock = (blockSize + 7) / 8;

    auto begin_decompress = std::chrono::high_resolution_clock::now(); 


    // === 提前准备所有偏移 ===
    //偏移的字节数
    std::vector<size_t> unpredOffsets(totalBlocks , 0);
    std::vector<size_t> compOffsets(totalBlocks , 0);

    //先记录所有偏移，后续不用每次累加
    for (size_t i = 1; i < totalBlocks; ++i) {
        unpredOffsets[i] = unpredOffsets[i - 1] + unpredSizes[i - 1];
        compOffsets[i] = compOffsets[i - 1] + compSizes[i - 1];
    }

    // === 循环外预分配缓冲区（复用）===
    std::vector<float> unpredData;
    std::vector<unsigned char> compData;
    std::vector<unsigned char> signBits(signBytesPerBlock);
    std::vector<int> quant_inds(blockSize);
    std::vector<float> decompressedBlock(blockSize);

    for (size_t bid : blockIds) {
        size_t local_id = bid ; 
        //auto t1 = std::chrono::high_resolution_clock::now();
        

        // Unpred
        unsigned char unpredSize = unpredSizes[local_id];
        size_t unpredOffset = totalBlocks + sizeof(float) * unpredOffsets[local_id];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredData.data() + unpredOffset, unpredSize * sizeof(float));

        // Compressed
        unsigned char bitCount = bitCounts[local_id];
        unsigned char dataSize = compSizes[local_id];
        size_t compOffset = 2 * totalBlocks + compOffsets[local_id];
        compData.resize(dataSize);
        std::memcpy(compData.data(), allCompData.data() + compOffset, dataSize);

        // Sign
        size_t signOffset = signBytesPerBlock * local_id;
        std::memcpy(signBits.data(), allSignData.data() + signOffset, signBytesPerBlock);

        //auto t2 = std::chrono::high_resolution_clock::now();
        //memory_access_time += std::chrono::duration<float>(t2 - t1).count();

        
    // 解码数据
    int radius = 512;
    std::vector<int> quant_inds(blockSize);

    size_t bitPos = 0;

        for (size_t i = 0; i < blockSize; ++i) {
            unsigned int val = 0, bitsRead = 0;

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


        //必须转换
        double error_bound_double = static_cast<double>(error_bound);

        // === Metadata + decompression ===
        std::vector<SZ3::uchar> metadataBuffer;
        metadataBuffer.push_back(0b00000010);
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&error_bound_double),
            reinterpret_cast<SZ3::uchar*>(&error_bound_double) + sizeof(double));
    
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&radius),
            reinterpret_cast<SZ3::uchar*>(&radius) + sizeof(int));
        size_t correctUnpredSize = static_cast<size_t>(unpredSize);
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize) + sizeof(size_t));
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data()),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data() + unpredSize));

        // 解压缩配置
        SZ3::Config conf(blockSize);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;

        auto predictor = SZ3::LorenzoPredictor<float, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<float, 1>(conf, quantizer);

        const SZ3::uchar* metaPtr = metadataBuffer.data();
        size_t metaSize = metadataBuffer.size();

        decompose.load(metaPtr, metaSize);
        decompose.decompress(conf, quant_inds, decompressedBlock.data());
        decompressedOriginalresult.push_back(decompressedBlock);  
    }

    auto end_decompress = std::chrono::high_resolution_clock::now(); 
    

    std::chrono::duration<double> readTime = begin_decompress - start_time_read;
    std::chrono::duration<double> compressTime = end_decompress - begin_decompress;

    
    std::cout <<"[Time2]:Total read time: " << readTime.count() << " seconds" << std::endl;
    std::cout << "Time3]:Total decompress time: " << compressTime.count() << " seconds" << std::endl;

    return decompressedOriginalresult;
}


// ============================================================
// 新增：对解压后的数据做 log10 变换
// ============================================================
// 只在解压完成之后、跑 Marching Cubes 之前调用，不改变压缩/解压逻辑本身。
// 密度场里如果出现 <= 0 的值（正常情况下密度不应为负，但压缩误差/量化
// 可能导致极少数点略微越过0），用 floorValue 做一个下限保护，避免 log(0)
// 或 log(负数) 产生 -inf / nan。
void applyLog10ToDecompressedData(
    std::vector<std::vector<float>>& decompressedBlocks,
    float floorValue = 1e-6f)
{
    for (auto& block : decompressedBlocks) {
        for (auto& v : block) {
            v = std::log10(std::max(v, floorValue));
        }
    }
}


    
/**
 * 预计算映射表
 * BlockDataAccessor类：用于快速访问分块数据
 * 
 * 核心思想：预计算一个映射表，将每个全局坐标(x,y,z)映射到对应的(块索引, 块内索引)
 * 这样后续访问数据时只需要O(1)时间，而不需要重复计算坐标转换
 */
/*class BlockDataAccessor {
private:
    const std::vector<std::vector<float>>& blockData;
    std::vector<std::pair<size_t, size_t>> positionToBlockMapping; // 映射表：线性位置 -> (块索引, 块内索引)
    std::vector<size_t> dataShape;

public:
    /**
     * 构造函数：预计算映射表
     * 这里会模拟原始分块过程，为每个数据点建立索引映射
     */
    /*BlockDataAccessor(const std::vector<std::vector<float>>& data,
                    const std::vector<size_t>& dShape,
                    const std::vector<size_t>& blockShape) 
        : blockData(data), dataShape(dShape) {


        // 预计算映射表
        size_t totalElements = dataShape[0] * dataShape[1] * dataShape[2];
        positionToBlockMapping.resize(totalElements);

        // 计算每个维度有多少个块
        std::vector<size_t> blockCountOnEachDim(3);
        for (size_t i = 0; i < 3; i++) {
            blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];
        }

        // 用于记录每个块当前已经放入了多少个数据点
        std::vector<size_t> blockDataCounts(blockData.size(), 0);

        for (size_t p = 0; p < totalElements; p++) {
            // 将线性位置p转换为3D坐标
            std::vector<size_t> elem_global_id = positionToIndices(p, dataShape);
            // 计算这个点属于哪个块（通过坐标除以块大小）
            std::vector<size_t> block_global_id(3);
            for (size_t i = 0; i < 3; i++) {
                block_global_id[i] = elem_global_id[i] / blockShape[i];
            }
        
            // 将块的3D坐标转换为块的线性索引
            size_t block_position = indicesToPosition(blockCountOnEachDim, block_global_id);
            // 这个点在其所属块中的索引（按照分块时的插入顺序）
            size_t indexInBlock = blockDataCounts[block_position];

            // 建立映射：线性位置p -> (块索引, 块内索引)
            positionToBlockMapping[p] = {block_position, indexInBlock};
            // 更新该块的数据计数
            blockDataCounts[block_position]++;
        }
        blockCountOnEachDim.resize(3);
        for (size_t i = 0; i < 3; i++) {
            blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];
        }
    }


    /**
    * 核心函数：根据全局坐标快速获取数据值
    * 输入：全局3D坐标(global_x, global_y, global_z)
    * 输出：该位置的数据值
    * 
    * 时间复杂度：O(1) - 直接查表，无需重复计算
    */
    /*inline float getValue(size_t global_x, size_t global_y, size_t global_z) const {
        // 将3D坐标转换为线性位置
        size_t linear_pos = global_z * dataShape[1] * dataShape[0] + global_y * dataShape[0] + global_x;

        if (linear_pos >= positionToBlockMapping.size()) {
            return 0.0f;
        }

        // 从映射表中获取块索引和块内索引
        size_t blockIndex = positionToBlockMapping[linear_pos].first;
        size_t indexInBlock = positionToBlockMapping[linear_pos].second;

        static int debug_count = 0;

        if (blockIndex < blockData.size() && indexInBlock < blockData[blockIndex].size()) {
            return blockData[blockIndex][indexInBlock];
        }

        return 0.0f;
    }

    // 检查坐标是否在有效范围内
    bool isValidCoord(size_t x, size_t y, size_t z) const {
        return x < dataShape[0] && y < dataShape[1] && z < dataShape[2];
    }

    const std::vector<size_t>& getDataShape() const { return dataShape; }
};*/


//在分块存储的数据里，根据全局坐标 (global_x, global_y, global_z) 找到对应的数值
//压缩解压后的数据是按照块存储的
//全局坐标是全局的空间坐标
inline float getBlockValue(
    const std::vector<std::vector<float>>& blockData,
    size_t global_x, size_t global_y, size_t global_z,
    const std::vector<size_t>& blockShape,
    size_t blocks_x, size_t blocks_y, size_t blocks_z) {
    
    size_t block_x = global_x / blockShape[0];
    size_t block_y = global_y / blockShape[1];
    size_t block_z = global_z / blockShape[2];
    
    // 块索引按照 x+y*X+z*X*Y 顺序
    size_t blockIndex = block_z + block_y * blocks_z + block_x * blocks_z * blocks_y;
    
    
    size_t local_x = global_x % blockShape[0];
    size_t local_y = global_y % blockShape[1];
    size_t local_z = global_z % blockShape[2];
    
    // 块内索引按照 x+y*X+z*X*Y 顺序
    size_t indexInBlock = local_x + local_y * blockShape[0] + local_z * blockShape[0] * blockShape[1];

    
    return blockData[blockIndex][indexInBlock];
}

// 直接从当前块中取值，块内局部坐标访问函数
//输入是当前块的数据+块内的局部空间坐标
inline float getBlockValueLocal(
    const std::vector<float>& blockData,
    size_t local_x, size_t local_y, size_t local_z,
    const std::vector<size_t>& blockShape
) {
    size_t localOffset = local_x + local_y * blockShape[0] + local_z * blockShape[0] * blockShape[1];
    return blockData[localOffset];
}


//块内cube遍历，不跨块。对“按块解压”的数据逐块运行Marching Cubes，但只遍历块内的cubes（不跨块取值/合并）
//输入：解压后的全部数据，按照一维存储
//在解压后的数据中，在每一个块内的每个cube,(根据local坐标，在解压块中找到对应的数据），进行每个cube的mc)
util::TriangleMesh<float> RunMarchingCubesOnDecompressedBlocks(
    
    const std::vector<std::vector<float>>& decompressedBlocks,
    float isovalue,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& dataShape
)
{
    // 计算每个维度的块数量
    std::vector<size_t> blockCountOnEachDim(3);
    for (size_t i = 0; i < 3; i++) {
        blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];
    }

    // 全局结果容器 
    std::vector<std::array<float, 3>> globalPoints;
    std::vector<std::array<float, 3>> globalNormals;
    std::vector<std::array<int, 3>> globalTriangles;  

    // 预分配内存
    size_t estimatedPoints = decompressedBlocks.size() * (blockShape[0]-1) * (blockShape[1]-1) * (blockShape[2]-1) * 6;
    globalPoints.reserve(estimatedPoints);
    globalNormals.reserve(estimatedPoints);
    globalTriangles.reserve(estimatedPoints / 3);

    int totalVertexOffset = 0;  
    size_t processedCubes = 0;

    // 按块处理，但只处理块内部的cube
    for (size_t blockIdx = 0; blockIdx < decompressedBlocks.size(); ++blockIdx) {
        

        // 计算当前块的3D坐标
        size_t block_z = blockIdx % blockCountOnEachDim[2];  
        size_t remaining = blockIdx / blockCountOnEachDim[2];
        size_t block_y = remaining % blockCountOnEachDim[1]; 
        size_t block_x = remaining / blockCountOnEachDim[1]; 


        // 计算当前块在全局坐标系中的起始位置
        size_t global_x_start = block_x * blockShape[0];
        size_t global_y_start = block_y * blockShape[1];
        size_t global_z_start = block_z * blockShape[2];

        // 计算当前块的有效范围（避免越界）
        size_t max_x = std::min(blockShape[0] - 1, dataShape[0] - global_x_start - 1);
        size_t max_y = std::min(blockShape[1] - 1, dataShape[1] - global_y_start - 1);
        size_t max_z = std::min(blockShape[2] - 1, dataShape[2] - global_z_start - 1);

        // 当前块的局部结果 
        std::vector<std::array<float, 3>> localPoints;
        std::vector<std::array<float, 3>> localNormals;
        std::vector<std::array<int, 3>> localTriangles;  

        std::unordered_map<size_t, int> localPointMap; 
        int localPtIdx = 0;  

        const std::vector<float>& currentBlockData = decompressedBlocks[blockIdx];

        // 遍历块内的每个cube
        for (size_t local_z = 0; local_z < max_z; ++local_z) {
            for (size_t local_y = 0; local_y < max_y; ++local_y) {
                for (size_t local_x = 0; local_x < max_x; ++local_x) {

                    // 计算cube的全局坐标
                    size_t global_x = global_x_start + local_x;
                    size_t global_y = global_y_start + local_y;
                    size_t global_z = global_z_start + local_z;

                    // 检查cube的8个顶点是否都在有效范围内
                    if (global_x + 1 >= dataShape[0] || global_y + 1 >= dataShape[1] || global_z + 1 >= dataShape[2]) {
                        continue; // 跳过边界cube
                    }

                    //从当前块中，根据块内坐标找到对应的值
                    std::array<float, 8> cubeValues = {{
                        getBlockValueLocal(currentBlockData, local_x, local_y, local_z, blockShape),         // 0
                        getBlockValueLocal(currentBlockData, local_x+1, local_y, local_z, blockShape),       // 1
                        getBlockValueLocal(currentBlockData, local_x+1, local_y+1, local_z, blockShape),     // 2
                        getBlockValueLocal(currentBlockData, local_x, local_y+1, local_z, blockShape),       // 3
                        getBlockValueLocal(currentBlockData, local_x, local_y, local_z+1, blockShape),       // 4
                        getBlockValueLocal(currentBlockData, local_x+1, local_y, local_z+1, blockShape),     // 5
                        getBlockValueLocal(currentBlockData, local_x+1, local_y+1, local_z+1, blockShape),   // 6
                        getBlockValueLocal(currentBlockData, local_x, local_y+1, local_z+1, blockShape)      // 7
                    }};


                    /*std::array<float, 8> cubeValues = {{
                        accessor.getValue(global_x, global_y, global_z),         // 0
                        accessor.getValue(global_x+1, global_y, global_z),       // 1
                        accessor.getValue(global_x+1, global_y+1, global_z),     // 2
                        accessor.getValue(global_x, global_y+1, global_z),       // 3
                        accessor.getValue(global_x, global_y, global_z+1),       // 4
                        accessor.getValue(global_x+1, global_y, global_z+1),     // 5
                        accessor.getValue(global_x+1, global_y+1, global_z+1),   // 6
                        accessor.getValue(global_x, global_y+1, global_z+1)      // 7
                    }};*/

                    // 计算marching cubes的配置ID
                    int cellCaseId = util::findCaseId(cubeValues, isovalue);

                    // 如果配置ID是0或255，说明等值面不穿过这个cube
                    if (cellCaseId == 0 || cellCaseId == 255) {
                        continue;
                    }

                    processedCubes++;

                    // 计算cube的8个顶点global坐标（用于后续插值）
                    std::array<std::array<float, 3>, 8> cubePositions = {{
                        {float(global_x), float(global_y), float(global_z)},         // 0
                        {float(global_x+1), float(global_y), float(global_z)},       // 1
                        {float(global_x+1), float(global_y+1), float(global_z)},     // 2
                        {float(global_x), float(global_y+1), float(global_z)},       // 3
                        {float(global_x), float(global_y), float(global_z+1)},       // 4
                        {float(global_x+1), float(global_y), float(global_z+1)},     // 5
                        {float(global_x+1), float(global_y+1), float(global_z+1)},   // 6
                        {float(global_x), float(global_y+1), float(global_z+1)}      // 7
                    }};

                    // 计算梯度（用于生成法向量）
                    std::array<std::array<float, 3>, 8> cubeGradients;
                    for (int i = 0; i < 8; ++i) {
                        size_t px = global_x + (i & 1);
                        size_t py = global_y + ((i >> 1) & 1);
                        size_t pz = global_z + ((i >> 2) & 1);
                        
                        std::array<float, 3> grad = {0.0f, 0.0f, 0.0f};
                        
                        

                        // X方向梯度 (块内计算)
                        if (px == 0) {
                            grad[0] = (getBlockValueLocal(currentBlockData, px, py, pz, blockShape) - 
                                      getBlockValueLocal(currentBlockData, px+1, py, pz, blockShape)) / 1.0f;
                        }
                        else if (px == blockShape[0] - 1) {
                            grad[0] = (getBlockValueLocal(currentBlockData, px-1, py, pz, blockShape) - 
                                      getBlockValueLocal(currentBlockData, px, py, pz, blockShape)) / 1.0f;
                        }
                        else {
                            grad[0] = (getBlockValueLocal(currentBlockData, px-1, py, pz, blockShape) - 
                                      getBlockValueLocal(currentBlockData, px+1, py, pz, blockShape)) / 2.0f;
                        }
                        
                        // Y方向梯度 (块内计算)
                        if (py == 0) {
                            grad[1] = (getBlockValueLocal(currentBlockData, px, py, pz, blockShape) - 
                                      getBlockValueLocal(currentBlockData, px, py+1, pz, blockShape)) / 1.0f;
                        }
                        else if (py == blockShape[1] - 1) {
                            grad[1] = (getBlockValueLocal(currentBlockData, px, py-1, pz, blockShape) - 
                                      getBlockValueLocal(currentBlockData, px, py, pz, blockShape)) / 1.0f;
                        }
                        else {
                            grad[1] = (getBlockValueLocal(currentBlockData, px, py-1, pz, blockShape) - 
                                      getBlockValueLocal(currentBlockData, px, py+1, pz, blockShape)) / 2.0f;
                        }
                        
                        // Z方向梯度 (块内计算)
                        if (pz == 0) {
                            grad[2] = (getBlockValueLocal(currentBlockData, px, py, pz, blockShape) - 
                                      getBlockValueLocal(currentBlockData, px, py, pz+1, blockShape)) / 1.0f;
                        }
                        else if (pz == blockShape[2] - 1) {
                            grad[2] = (getBlockValueLocal(currentBlockData, px, py, pz-1, blockShape) - 
                                      getBlockValueLocal(currentBlockData, px, py, pz, blockShape)) / 1.0f;
                        }
                        else {
                            grad[2] = (getBlockValueLocal(currentBlockData, px, py, pz-1, blockShape) - 
                                      getBlockValueLocal(currentBlockData, px, py, pz+1, blockShape)) / 2.0f;
                        }
                        
                        cubeGradients[i] = grad;
                    }

                    

                    // 生成三角形
                    const int *triEdges = util::caseTrianglesEdges[cellCaseId];

                    // 处理每个三角形（每3个边索引定义一个三角形）
                    for (; *triEdges != -1; triEdges += 3) {
                        std::array<int, 3> tri;  
                        
                        for (int i = 0; i < 3; ++i) {
                            int edgeIdx = triEdges[i];
                            
                            // 使用简单的局部边索引
                            size_t localEdgeIdx = (local_z * max_y * max_x + local_y * max_x + local_x) * 12 + edgeIdx;
                            
                            auto it = localPointMap.find(localEdgeIdx);
                            if (it != localPointMap.end()) {
                                tri[i] = it->second;
                            } else {
                                const int *vs = util::edgeVertices[edgeIdx];
                                int v1 = vs[0];
                                int v2 = vs[1];
                                
                                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                                
                                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                                
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
            }
        }

        // 合并局部结果到全局
        globalPoints.insert(globalPoints.end(), localPoints.begin(), localPoints.end());
        globalNormals.insert(globalNormals.end(), localNormals.begin(), localNormals.end());

        // 合并三角形（需要调整顶点索引）
        for (const auto& tri : localTriangles) {
            std::array<int, 3> adjustedTri = {{  
                tri[0] + totalVertexOffset,
                tri[1] + totalVertexOffset,
                tri[2] + totalVertexOffset
            }};
            globalTriangles.push_back(adjustedTri);
        }

        // 更新全局顶点偏移量 
        if (localPoints.size() > std::numeric_limits<int>::max() - totalVertexOffset) {
            throw std::runtime_error("Too many vertices for int indexing");
        }
        totalVertexOffset += static_cast<int>(localPoints.size());
    }

    return util::TriangleMesh<float>(globalPoints, globalNormals, globalTriangles);
}





//全局cube遍历，跨块
//有按块解压的数据，有全部的数据
//直接按MC的方式遍历每个cube，根据全局坐标，从解压后的数据中找到对应的数据进行mc处理
util::TriangleMesh<float> RunMarchingCubesOnDecompressedBlocksOptimized(
    const std::vector<std::vector<float>>& decompressedBlocks,
    float isovalue,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& dataShape
)
{
    // 预计算块维度信息（避免重复计算）
    size_t blocks_x = (dataShape[0] + blockShape[0] - 1) / blockShape[0];
    size_t blocks_y = (dataShape[1] + blockShape[1] - 1) / blockShape[1];
    size_t blocks_z = (dataShape[2] + blockShape[2] - 1) / blockShape[2];

    // 全局结果容器 
    std::vector<std::array<float, 3>> globalPoints;
    std::vector<std::array<float, 3>> globalNormals;
    std::vector<std::array<int, 3>> globalTriangles; 

    // 全局点映射和索引计数 
    std::unordered_map<size_t, int> globalPointMap; 
    int ptIdx = 0; 

    size_t processedCubes = 0;

    // 直接按MC的方式遍历每个cube，参照源码顺序
    for (size_t z = 0; z < dataShape[2] - 1; ++z) {
        for (size_t y = 0; y < dataShape[1] - 1; ++y) {
            for (size_t x = 0; x < dataShape[0] - 1; ++x) {
                
                // 获取cube的8个顶点值
                std::array<float, 8> cubeValues = {{
                    getBlockValue(decompressedBlocks, x, y, z, blockShape, blocks_x, blocks_y, blocks_z),         // 0
                    getBlockValue(decompressedBlocks, x+1, y, z, blockShape, blocks_x, blocks_y, blocks_z),       // 1
                    getBlockValue(decompressedBlocks, x+1, y+1, z, blockShape, blocks_x, blocks_y, blocks_z),     // 2
                    getBlockValue(decompressedBlocks, x, y+1, z, blockShape, blocks_x, blocks_y, blocks_z),       // 3
                    getBlockValue(decompressedBlocks, x, y, z+1, blockShape, blocks_x, blocks_y, blocks_z),       // 4
                    getBlockValue(decompressedBlocks, x+1, y, z+1, blockShape, blocks_x, blocks_y, blocks_z),     // 5
                    getBlockValue(decompressedBlocks, x+1, y+1, z+1, blockShape, blocks_x, blocks_y, blocks_z),   // 6
                    getBlockValue(decompressedBlocks, x, y+1, z+1, blockShape, blocks_x, blocks_y, blocks_z)      // 7
                }};

                // 计算marching cubes的配置ID
                int cellCaseId = util::findCaseId(cubeValues, isovalue);

                // 如果配置ID是0或255，说明等值面不穿过这个cube
                if (cellCaseId == 0 || cellCaseId == 255) {
                    continue;
                }

                processedCubes++;

                // 计算cube的8个顶点坐标（用于后续插值）
                std::array<std::array<float, 3>, 8> cubePositions = {{
                    {float(x), float(y), float(z)},         // 0
                    {float(x+1), float(y), float(z)},       // 1
                    {float(x+1), float(y+1), float(z)},     // 2
                    {float(x), float(y+1), float(z)},       // 3
                    {float(x), float(y), float(z+1)},       // 4
                    {float(x+1), float(y), float(z+1)},     // 5
                    {float(x+1), float(y+1), float(z+1)},   // 6
                    {float(x), float(y+1), float(z+1)}      // 7
                }};

                // 计算梯度（用于生成法向量）
                std::array<std::array<float, 3>, 8> cubeGradients;
                for (int i = 0; i < 8; ++i) {
                    size_t px = x + (i & 1);
                    size_t py = y + ((i >> 1) & 1);
                    size_t pz = z + ((i >> 2) & 1);
                    
                    std::array<float, 3> grad = {0.0f, 0.0f, 0.0f};
                    
                    // X方向梯度
                    if (px == 0) {
                        grad[0] = (getBlockValue(decompressedBlocks, px, py, pz, blockShape, blocks_x, blocks_y, blocks_z) - 
                                  getBlockValue(decompressedBlocks, px+1, py, pz, blockShape, blocks_x, blocks_y, blocks_z)) / 1.0f;
                    }
                    else if (px == dataShape[0] - 1) {
                        grad[0] = (getBlockValue(decompressedBlocks, px-1, py, pz, blockShape, blocks_x, blocks_y, blocks_z) - 
                                  getBlockValue(decompressedBlocks, px, py, pz, blockShape, blocks_x, blocks_y, blocks_z)) / 1.0f;
                    }
                    else {
                        grad[0] = (getBlockValue(decompressedBlocks, px-1, py, pz, blockShape, blocks_x, blocks_y, blocks_z) - 
                                  getBlockValue(decompressedBlocks, px+1, py, pz, blockShape, blocks_x, blocks_y, blocks_z)) / 2.0f;
                    }
                    
                    // Y方向梯度
                    if (py == 0) {
                        grad[1] = (getBlockValue(decompressedBlocks, px, py, pz, blockShape, blocks_x, blocks_y, blocks_z) - 
                                  getBlockValue(decompressedBlocks, px, py+1, pz, blockShape, blocks_x, blocks_y, blocks_z)) / 1.0f;
                    }
                    else if (py == dataShape[1] - 1) {
                        grad[1] = (getBlockValue(decompressedBlocks, px, py-1, pz, blockShape, blocks_x, blocks_y, blocks_z) - 
                                  getBlockValue(decompressedBlocks, px, py, pz, blockShape, blocks_x, blocks_y, blocks_z)) / 1.0f;
                    }
                    else {
                        grad[1] = (getBlockValue(decompressedBlocks, px, py-1, pz, blockShape, blocks_x, blocks_y, blocks_z) - 
                                  getBlockValue(decompressedBlocks, px, py+1, pz, blockShape, blocks_x, blocks_y, blocks_z)) / 2.0f;
                    }
                    
                    // Z方向梯度
                    if (pz == 0) {
                        grad[2] = (getBlockValue(decompressedBlocks, px, py, pz, blockShape, blocks_x, blocks_y, blocks_z) - 
                                  getBlockValue(decompressedBlocks, px, py, pz+1, blockShape, blocks_x, blocks_y, blocks_z)) / 1.0f;
                    }
                    else if (pz == dataShape[2] - 1) {
                        grad[2] = (getBlockValue(decompressedBlocks, px, py, pz-1, blockShape, blocks_x, blocks_y, blocks_z) - 
                                  getBlockValue(decompressedBlocks, px, py, pz, blockShape, blocks_x, blocks_y, blocks_z)) / 1.0f;
                    }
                    else {
                        grad[2] = (getBlockValue(decompressedBlocks, px, py, pz-1, blockShape, blocks_x, blocks_y, blocks_z) - 
                                  getBlockValue(decompressedBlocks, px, py, pz+1, blockShape, blocks_x, blocks_y, blocks_z)) / 2.0f;
                    }
                    
                    cubeGradients[i] = grad;
                }

                // 生成三角形
                const int *triEdges = util::caseTrianglesEdges[cellCaseId];

                // 处理每个三角形（每3个边索引定义一个三角形）
                for (; *triEdges != -1; triEdges += 3) {
                    std::array<int, 3> tri;  
                    
                    for (int i = 0; i < 3; ++i) {
                        int edgeIdx = triEdges[i];
                        
                        // 计算全局边索引（类似MC源码的globalEdgeIndex）
                        size_t globalEdgeIdx = ((z * (dataShape[1]-1) + y) * (dataShape[0]-1) + x) * 12 + edgeIdx;
                        
                        auto it = globalPointMap.find(globalEdgeIdx);
                        if (it != globalPointMap.end()) {
                            tri[i] = it->second;
                        } else {
                            const int *vs = util::edgeVertices[edgeIdx];
                            int v1 = vs[0];
                            int v2 = vs[1];
                            
                            float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                            
                            std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                            std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                            
                            globalPoints.push_back(newPt);
                            globalNormals.push_back(newNorm);
                            
                            globalPointMap[globalEdgeIdx] = ptIdx;
                            tri[i] = ptIdx++;
                        }
                    }
                    
                    if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
                        globalTriangles.push_back(tri);
                    }
                }
            }
        }
    }

    return util::TriangleMesh<float>(globalPoints, globalNormals, globalTriangles);
}



void RunAndSaveIsosurfaceMesh(
    const std::vector<std::vector<float>>& decompressedBlocks,
    float isovalue,
    const std::vector<size_t>& smallBlockShape,
    const std::vector<size_t>& stepDataShape,
    const std::string& outFile
) {
        // 打印基本信息
        std::cout << "[Info] Running Marching Cubes on decompressed blocks\n";
        std::cout << "       Volume shape: (" << stepDataShape[0] << ", " << stepDataShape[1] << ", " << stepDataShape[2] << ")\n";
        std::cout << "       Block shape:  (" << smallBlockShape[0] << ", " << smallBlockShape[1] << ", " << smallBlockShape[2] << ")\n";
        std::cout << "       Isovalue (log10 space):     " << isovalue << "\n";
        std::cout << "       Output file:  " << outFile << "\n";


        //auto mappingStart = std::chrono::high_resolution_clock::now();
        
        //BlockDataAccessor accessor(decompressedBlocks, stepDataShape, smallBlockShape);
        
        //auto mappingEnd = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> mappingTime = mappingEnd - mappingStart;
        //std::cout << "[Info] Mapping table built in " << mappingTime.count() << " seconds.\n";


        // 启动计时器
        util::Timer timer;
        timer.start();

        // 执行 Marching Cubes 提取
        util::TriangleMesh<float> mesh = RunMarchingCubesOnDecompressedBlocksOptimized(
            
            decompressedBlocks,
            isovalue,
            smallBlockShape,
            stepDataShape 
        );

        timer.stop();

        // 打印结果信息
        std::cout << "[Result] Mesh vertices:  " << mesh.numberOfVertices() << "\n";
        std::cout << "[Result] Mesh triangles: " << mesh.numberOfTriangles() << "\n";
        std::cout << "[Timing] CPU time (sec):  " << timer.getCPUtime() << "\n";
        std::cout << "[Timing] Wall time (sec): " << timer.getWallTime() << "\n";

        // 写入 VTK
        util::saveTriangleMesh(mesh, outFile.c_str());

    }



//将分块解压的数据恢复为分块前一维结构（仅做验证）
std::vector<float> restoreVolumeFromBlocks(
    const std::vector<std::vector<float>>& decompressedTargetOriginalBlockdata,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& dataShape
) {
    size_t nDim = dataShape.size();
    size_t totalElements = 1;
    for (size_t dim : dataShape) {
        totalElements *= dim;
    }
    
    // 计算每个维度的块数量
    std::vector<size_t> blockCountOnEachDim(nDim);
    for (size_t i = 0; i < nDim; i++) {
        blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];
    }
    
    // 初始化还原后的数据
    std::vector<float> restoredData(totalElements);
    
    // 重新模拟原始的分块过程来确定数据对应关系
    // 创建一个映射：原始位置 -> (块索引, 块内索引)
    std::vector<std::pair<size_t, size_t>> positionToBlockMapping(totalElements);
    std::vector<size_t> blockDataCounts(decompressedTargetOriginalBlockdata.size(), 0);
    
    // 按照原始分块逻辑建立映射
    for (size_t p = 0; p < totalElements; p++) {
        std::vector<size_t> elem_global_id = positionToIndices(p, dataShape);
        std::vector<size_t> block_global_id(nDim);
        for (size_t i = 0; i < nDim; i++) {
            block_global_id[i] = elem_global_id[i] / blockShape[i];
        }
        
        size_t block_position = indicesToPosition(blockCountOnEachDim, block_global_id);
        size_t indexInBlock = blockDataCounts[block_position];
        
        positionToBlockMapping[p] = {block_position, indexInBlock};
        blockDataCounts[block_position]++;
    }
    
    // 根据映射还原数据
    for (size_t p = 0; p < totalElements; p++) {
        size_t blockIndex = positionToBlockMapping[p].first;
        size_t indexInBlock = positionToBlockMapping[p].second;
        
        if (blockIndex < decompressedTargetOriginalBlockdata.size() && 
            indexInBlock < decompressedTargetOriginalBlockdata[blockIndex].size()) {
            restoredData[p] = decompressedTargetOriginalBlockdata[blockIndex][indexInBlock];
        }
    }
    
    // 打印前300个值进行调试
    /*std::cout << "=== Restored Data Debug (First 300 values) ===" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(300, restoredData.size()); ++i) {
        std::cout << "restoredData[" << i << "] = " << std::setprecision(6) << restoredData[i] << std::endl;
    }
    std::cout << "=== End of Restored Data Debug ===" << std::endl;*/
    
    return restoredData;
}



void writeVolumeToVTK(
    const std::string& filename,
    const std::vector<float>& volumeData,
    const std::vector<size_t>& dataShape
    // [X, Y, Z]
) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    size_t X = dataShape[0], Y = dataShape[1], Z = dataShape[2];
    size_t total = X * Y * Z;

    out << "# vtk DataFile Version 3.0\n";
    out << "Volume example\n";
    out << "BINARY\n";
    out << "DATASET STRUCTURED_POINTS\n";
    
    // VTK期望的维度顺序是 [X, Y, Z]，但数据存储顺序可能是 [Z, Y, X]
    // 为了不改变volumeData的写入顺序，我们需要调整DIMENSIONS声明
    // 如果数据是按照 z-y-x 顺序存储的，那么在VTK中应该声明为 Z Y X
    out << "DIMENSIONS " << Z << " " << Y << " " << X << "\n";
    out << "ORIGIN 0 0 0\n";
    out << "SPACING 1 1 1\n";
    out << "POINT_DATA " << total << "\n";
    out << "SCALARS volume_scalars float\n";
    out << "LOOKUP_TABLE default\n";

    // 保持volumeData的原始顺序写入
    for (float val : volumeData) {
        float be_val = val;
        uint8_t* p = reinterpret_cast<uint8_t*>(&be_val);
        std::reverse(p, p + sizeof(float));  // 转换为 big-endian
        out.write(reinterpret_cast<char*>(p), sizeof(float));
    }

    /*for (size_t i = 0; i < std::min<size_t>(300, volumeData.size()); ++i) {
        std::cout << "volumeData[" << i << "] = " << std::setprecision(6) << volumeData[i] << std::endl;
    }*/
        
    out.close();
}

 


int main(int argc, char* argv[]) {

    auto totalStart = std::chrono::high_resolution_clock::now(); 



    int mpi_size = 1;
    int mpi_rank = 0;

    std::string inputFileName, variableName, indexDir;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, blockShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0, bigBlocksPerStep = 1, smallBlocksPerBig = 1 ,  smallBlockSize = 1 , totalBlocksNumber = 1, smallBlocksPerStep = 1; 
    std::vector<float> queryRange;
    float relative_error_bound = 1E-3;
    size_t extraValue =0;
    


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

    for (size_t d = 0; d < nDim; d++) {
        smallBlocksPerStep *= stepDataShape[d] / smallBlockShape[d];            
        // 每维小块数量
        smallBlockSize *= smallBlockShape[d];                             // 每个小块的数据量
    }

    size_t nSteps = 1; 
    //小块总数
    totalBlocksNumber = nSteps * smallBlocksPerStep;



    if (queryRange.size() != 2) {
        //if (mpi_rank == 0) 
        std::cerr << "Error: Invalid query range. Use --query_range <low> <high>" << std::endl;
        //MPI_Finalize();
        return 1;
    }

    

    
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    //std::string subDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +   "_iso_fixlength_originalDataCompression/";

    std::string safeDatasetName = variableName;
    std::replace(safeDatasetName.begin(), safeDatasetName.end(), '/', '_');

    std::string subDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + safeDatasetName + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_hdf5_fixlength_originalDataCompression/";

       // === 新增：索引目录，对接 doc16（HDF5 uniform索引程序）生成的路径 ===
    // 只用来读 big_block_minmax 拿真实 globalMin/globalMax，不做任何索引查询筛选
    indexDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + safeDatasetName + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_hdf5_uniform_index/";

    // === 从索引的 big_block_minmax 读取真实 globalMin/globalMax ===
    std::vector<ScidxInterval<float>> allBigBlockIndices = loadBigBlockIndexFile(indexDir + "big_block_minmax");

    float globalMin = std::numeric_limits<float>::max();
    float globalMax = std::numeric_limits<float>::lowest();

    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }

    std::cout << "[Debug] indexDir: " << indexDir << std::endl;
    std::cout << "[Debug] big_block_minmax entries: " << allBigBlockIndices.size() << std::endl;
    std::cout << "Global Min: " << globalMin << std::endl;
    std::cout << "Global Max: " << globalMax << std::endl;

    // === 修复点：用真实 globalMin/globalMax 算 error_bound，
    //     跟压缩程序（doc25）里的算法保持一致，才能正确解压 ===
    float error_bound = relative_error_bound * (globalMax - globalMin);
    std::cout << "Computed error_bound: " << error_bound << std::endl;

    

    std::vector<size_t> allGlobalSmallBlockIds(totalBlocksNumber);  
    for (size_t i = 0; i < totalBlocksNumber; ++i) {
        allGlobalSmallBlockIds[i] = i;
    }
   
    size_t myLocalBlockCount = allGlobalSmallBlockIds.size();


    //时间
    auto afterPreProcessingTime = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<float> preProcessingTime = afterPreProcessingTime - totalStart;
    std::cout << "[Rank " << mpi_rank << "] [Time1]: preProcessingTime: " << preProcessingTime.count() << " seconds" << std::endl;


    // ============================================================
    // 解压逻辑完全不变
    // ============================================================
    std::vector<std::vector<float>> decompressedTargetOriginalBlockdata = batchDecompressBlocksAllRead(
        subDir, allGlobalSmallBlockIds, smallBlockSize, myLocalBlockCount, error_bound);  


    // ============================================================
    // 新增：解压完成之后、跑 Marching Cubes 之前，对数据做 log10 变换
    // 这一步只是数值空间的转换，不涉及重新压缩/重新解压
    // ============================================================
    /*auto logStart = std::chrono::high_resolution_clock::now();
    applyLog10ToDecompressedData(decompressedTargetOriginalBlockdata);
    auto logEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> logTime = logEnd - logStart;
    std::cout << "[Time] log10 transform time: " << logTime.count() << " seconds" << std::endl;

    std::cout << "[Info] 注意: --query_range 传入的 isovalue 现在需要是 log10 空间下的值,"
              << " 例如原始密度1000对应的 isovalue 应该填 log10(1000)=3, 而不是1000本身" << std::endl;*/


    /*auto restoreStart = std::chrono::high_resolution_clock::now();

    //将分块解压的数据恢复成分块前顺序一维数据
    auto restored = restoreVolumeFromBlocks(decompressedTargetOriginalBlockdata, smallBlockShape, stepDataShape);
    
    auto restoreEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> restoreTime = restoreEnd - restoreStart;
    std::cout << "[Time] Restore Volume From Blocks: " << restoreTime.count() << " seconds" << std::endl;


    //写入vtk
    writeVolumeToVTK("decompressed_volume.vtk", restored, stepDataShape);

    //auto loadStart = std::chrono::high_resolution_clock::now();
    
    //util::Image3D<float> image = util::loadImage<float>("decompressed_volume.vtk");
    
    //auto loadEnd = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> loadTime = loadEnd - loadStart;
    //std::cout << "[Time] LoadImage time: " << loadTime.count() << " seconds" << std::endl;*/


// 输出目录（大规模scratch存储，避免home目录配额/IO压力）
    std::string outputDir = "/expanse/lustre/scratch/sdi/temp_project/iso_results/";
    createDirectory(outputDir);  // 代码里已有这个函数，确保目录存在

    // 拼接一个带辨识度的文件名，避免每次运行相互覆盖
    std::string outFileName = outputDir + "isosurface_" + safeDatasetName
        + "_step" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum)
        + "_eb" + std::to_string(static_cast<long long>(error_bound))
        + "_iso" + std::to_string(queryRange[0])
        + ".vtk";

    std::cout << "[Info] Output vtk path: " << outFileName << std::endl;

    //计入iso surface (isovalue 现在是在 log10 空间下解释)
    RunAndSaveIsosurfaceMesh(
        decompressedTargetOriginalBlockdata,         //分块解压数据 (已做log10变换)
        queryRange[0],                                     //  isovalue值 (log10空间)
        smallBlockShape,                                // block 大小
        stepDataShape,                           // 总volume 尺寸
        outFileName                                    // 输出ISO surface文件名（改到scratch目录）
    );
    return 0;
}

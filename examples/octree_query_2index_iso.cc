#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <scidx_octree_interval.h>   // OctreeNode, buildOctree, queryOctree
#include <scidx_octree.h>            // 保留，供对比使用
#include <scidx_octree_hilbert2.h>   // [Hilbert2] decompressOctreeUniformHilbert / decompressOctreeStaggerHilbertNew
#include <SZ3/api/sz.hpp>
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <utility>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include <array>

// ============================================================
// 工具函数：创建目录
// ============================================================
void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
}

// ============================================================
// 读取 big_block_minmax 文件（大块粗筛用，跟索引结构无关，不变）
// ============================================================
std::vector<ScidxInterval<double>> loadBigBlockIndexFile(const std::string& filename) {
    std::vector<ScidxInterval<double>> bigBlockIndices;

    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return bigBlockIndices;
    }

    double minVal, maxVal;
    while (inFile.read(reinterpret_cast<char*>(&minVal), sizeof(double)) &&
           inFile.read(reinterpret_cast<char*>(&maxVal), sizeof(double))) {
        bigBlockIndices.push_back({minVal, maxVal});
    }

    inFile.close();
    return bigBlockIndices;
}

// ============================================================
// 全局小块 ID 计算（跟索引结构无关，不变）
// ============================================================
size_t computeGlobalSmallId(size_t block_id, size_t local_small_block_idx,
    size_t Nx, size_t Ny, size_t Nz,
    size_t sx, size_t sy, size_t sz,
    size_t small_x, size_t small_y, size_t small_z) {

    size_t bx = Nx / sx;
    size_t by = Ny / sy;
    size_t bz = Nz / sz;

    size_t big_blocks_per_step = bx * by * bz;

    size_t step_idx = block_id / big_blocks_per_step;
    size_t block_id_in_step = block_id % big_blocks_per_step;

    size_t big_block_x = block_id_in_step % bx;
    size_t big_block_y = (block_id_in_step / bx) % by;
    size_t big_block_z = block_id_in_step / (bx * by);

    size_t local_small_blocks_x = sx / small_x;
    size_t local_small_blocks_y = sy / small_y;
    size_t local_small_blocks_z = sz / small_z;

    size_t local_small_x = local_small_block_idx % local_small_blocks_x;
    size_t local_small_y = (local_small_block_idx / local_small_blocks_x) % local_small_blocks_y;
    size_t local_small_z = local_small_block_idx / (local_small_blocks_x * local_small_blocks_y);

    size_t global_small_x = big_block_x * local_small_blocks_x + local_small_x;
    size_t global_small_y = big_block_y * local_small_blocks_y + local_small_y;
    size_t global_small_z = big_block_z * local_small_blocks_z + local_small_z;

    size_t global_blocks_x = Nx / small_x;
    size_t global_blocks_y = Ny / small_y;
    size_t global_blocks_z = Nz / small_z;

    size_t global_id_in_step = global_small_x +
           global_small_y * global_blocks_x +
           global_small_z * global_blocks_x * global_blocks_y;

    size_t globalsmallid = global_id_in_step +
       step_idx * global_blocks_x * global_blocks_y * global_blocks_z;

    return globalsmallid;
}

// ============================================================
// [Hilbert2] 查询 UNIFORM 树，同时把 listLow / listHigh 一并返回
// UNIFORM 解压时已经直接产出 listLow/listHigh（按行主序flatId索引），
// 不再需要从树里反过来抠一遍，extractListHighFromUniformOctree 整个删掉
// ============================================================
std::tuple<std::vector<size_t>, std::vector<OctreeNode<double>>, std::vector<double>, std::vector<double>>
queryOctreeUniformRawIdsWithTree(
    size_t actualStepNum,
    const std::string& indexDir,
    const std::vector<double>& queryRange,
    double error_bound,
    const std::vector<size_t>& blockCountOnEachDim)
{
    std::string treeID = indexDir + std::to_string(actualStepNum) + "-0";
    std::cout << "[queryOctreeUniformRawIds] treeID: " << treeID << std::endl;

    auto time_before_decompress = std::chrono::high_resolution_clock::now();

    // [Hilbert2] 换成新的UNIFORM解压函数，同时拿到 listLow / listHigh
    std::vector<double> listLow, listHigh;
    std::vector<OctreeNode<double>> octree =
        decompressOctreeUniformHilbertNew<double>(treeID, error_bound, blockCountOnEachDim,
                                               &listLow, &listHigh);

    auto time_after_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompressTime = time_after_decompress - time_before_decompress;
    std::cerr << "[Time4][Octree] decompress uniform (" << treeID << "): "
              << decompressTime.count() << " seconds" << std::endl;

    auto time_before_query = std::chrono::high_resolution_clock::now();
    std::vector<size_t> ids = queryOctree(
        octree, queryRange[0], queryRange[1], blockCountOnEachDim, error_bound);
    auto time_after_query = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> queryTime = time_after_query - time_before_query;
    std::cerr << "[Time5][Octree] query uniform (" << treeID << "): "
              << queryTime.count() << " seconds" << std::endl;

    std::cout << "[queryOctreeUniformRawIds] Step " << actualStepNum
              << " -> " << ids.size() << " raw block IDs" << std::endl;

    return {ids, octree, listLow, listHigh};
}

// ============================================================
// [Hilbert2] 查询 STAGGER 树，需要同时传入 listLow 和 listHigh
// （STAGGER 的 min/max 都通过查 UNIFORM 邻居的 listLow/listHigh 求平均预测）
// ============================================================
std::vector<size_t> queryOctreeStaggerRawIds(
    size_t actualStepNum,
    const std::string& staggerIndexDir,
    const std::vector<double>& queryRange,
    double error_bound,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<double>& listLow,
    const std::vector<double>& listHigh)
{
    std::string staggerTreeID = staggerIndexDir + std::to_string(actualStepNum) + "-0";
    std::cout << "[queryOctreeStaggerRawIds] treeID: " << staggerTreeID << std::endl;

    auto time_before_decompress = std::chrono::high_resolution_clock::now();

    // [Hilbert2] 换成新的STAGGER解压函数（Hilbert2版，函数名带New后缀，跟压缩端对齐）
    std::vector<OctreeNode<double>> staggerOctree =
        decompressOctreeStaggerHilbertNew<double>(
            staggerTreeID, error_bound,
            staggerBlockCountOnEachDim, uniformBlockCountOnEachDim,
            listLow, listHigh);

    auto time_after_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompressTime = time_after_decompress - time_before_decompress;
    std::cerr << "[Time4][Octree Stagger] decompress (" << staggerTreeID << "): "
              << decompressTime.count() << " seconds" << std::endl;

    auto time_before_query = std::chrono::high_resolution_clock::now();
    std::vector<size_t> ids = queryOctree(
        staggerOctree, queryRange[0], queryRange[1], staggerBlockCountOnEachDim, error_bound);
    auto time_after_query = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> queryTime = time_after_query - time_before_query;
    std::cerr << "[Time5][Octree Stagger] query (" << staggerTreeID << "): "
              << queryTime.count() << " seconds" << std::endl;

    std::cout << "[queryOctreeStaggerRawIds] Step " << actualStepNum
              << " -> " << ids.size() << " raw block IDs" << std::endl;
    return ids;
}

// ============================================================
// [新增] x-fastest id -> z-fastest id 坐标轴转换
// ============================================================
inline size_t convertXFastestToZFastest(size_t xFastestId, const std::vector<size_t>& dim) {
    size_t bx = xFastestId % dim[0];
    size_t by = (xFastestId / dim[0]) % dim[1];
    size_t bz = xFastestId / (dim[0] * dim[1]);
    return bx * dim[1] * dim[2] + by * dim[2] + bz;
}

// ============================================================
// [更新] 错位块 flat ID -> 覆盖的均匀块 flat ID 集合
// ============================================================
std::vector<size_t> convertStaggerToUniformIds(
    size_t staggerFlatId,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& smallBlockShape,
    const std::vector<size_t>& halfBlockShape,
    const std::vector<size_t>& staggerBlockCount,
    const std::vector<size_t>& uniformBlockCount)
{
    size_t sk0 = staggerFlatId % staggerBlockCount[0];
    size_t sk1 = (staggerFlatId / staggerBlockCount[0]) % staggerBlockCount[1];
    size_t sk2 = staggerFlatId / (staggerBlockCount[0] * staggerBlockCount[1]);
    std::array<size_t, 3> sk = {sk0, sk1, sk2};

    std::array<std::pair<size_t, size_t>, 3> uRange;
    for (int d = 0; d < 3; d++) {
        size_t k = sk[d];
        size_t start, end_excl;

        if (k == 0) {
            start    = 0;
            end_excl = halfBlockShape[d] + smallBlockShape[d];
        } else {
            start    = halfBlockShape[d] + k * smallBlockShape[d];
            end_excl = std::min(start + smallBlockShape[d], shape[d]);
        }

        if (start >= shape[d] || end_excl == 0 || start >= end_excl) {
            return {};
        }

        size_t u_start = start / smallBlockShape[d];
        size_t u_end   = (end_excl - 1) / smallBlockShape[d];
        u_end = std::min(u_end, uniformBlockCount[d] - 1);
        uRange[d] = {u_start, u_end};
    }

    std::vector<size_t> result;
    for (size_t ux = uRange[0].first; ux <= uRange[0].second; ux++) {
        for (size_t uy = uRange[1].first; uy <= uRange[1].second; uy++) {
            for (size_t uz = uRange[2].first; uz <= uRange[2].second; uz++) {
                size_t flatId = ux
                              + uy * uniformBlockCount[0]
                              + uz * uniformBlockCount[0] * uniformBlockCount[1];
                result.push_back(flatId);
            }
        }
    }
    return result;
}

// ============================================================
// 以下：原始数据解压函数（跟索引结构无关，原样保留）
// ============================================================

struct BlockConfig {
    size_t blockSize;
    size_t unpredSizeBits;
    size_t dataSizeBytes;
    size_t signBytesPerBlock;

    BlockConfig(size_t bs) : blockSize(bs) {
        if (bs == 64) {
            unpredSizeBits = 6; dataSizeBytes = 1; signBytesPerBlock = 8;
        } else if (bs == 4096) {
            unpredSizeBits = 12; dataSizeBytes = 2; signBytesPerBlock = 512;
        } else if (bs == 262144) {
            unpredSizeBits = 18; dataSizeBytes = 3; signBytesPerBlock = 32768;
        } else if (bs == 16777216) {
            unpredSizeBits = 24; dataSizeBytes = 4; signBytesPerBlock = 2097152;
        } else if (bs == 1073741824) {
            unpredSizeBits = 30; dataSizeBytes = 4; signBytesPerBlock = 134217728;
        } else {
            throw std::runtime_error("Unsupported block size");
        }
    }
};

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

uint64_t readValueInline(const uint8_t* data, size_t numBytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < numBytes; i++) {
        value |= (static_cast<uint64_t>(data[i]) << (i * 8));
    }
    return value;
}

std::vector<std::vector<double>> universalBatchDecompressBlocks(
    const std::string& subDir,
    const std::vector<size_t>& blockIds,
    size_t blockSize,
    size_t totalBlocks,
    double error_bound) {

    int rank = 0;
    std::cout << "universal read: " << std::endl;

    BlockConfig config(blockSize);

    auto beginReadOriginalBlocks = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> decompressedOriginalresult;

    std::cout << "subDir = " << subDir << "\n";

    std::string unpredDataFileName = subDir + "universal_unpredData.bin";
    std::string compressedFileName = subDir + "universal_compressed_data.bin";
    std::string signFileName = subDir + "universal_sign_data.bin";

    std::ifstream unpredStream(unpredDataFileName, std::ios::binary);
    std::ifstream compStream(compressedFileName, std::ios::binary);
    std::ifstream signStream(signFileName, std::ios::binary);

    if (!unpredStream || !compStream || !signStream) {
        throw std::runtime_error("Failed to open one or more input files.");
    }

    unpredStream.seekg(0, std::ios::end);
    size_t unpredFileSize = unpredStream.tellg();
    unpredStream.seekg(0, std::ios::beg);

    compStream.seekg(0, std::ios::end);
    size_t compFileSize = compStream.tellg();
    compStream.seekg(0, std::ios::beg);

    signStream.seekg(0, std::ios::end);
    size_t signFileSize = signStream.tellg();
    signStream.seekg(0, std::ios::beg);

    std::vector<uint8_t> allUnpredDataRead(unpredFileSize);
    std::vector<uint8_t> allCompDataRead(compFileSize);
    std::vector<uint8_t> allSignDataRead(signFileSize);

    unpredStream.read(reinterpret_cast<char*>(allUnpredDataRead.data()), unpredFileSize);
    compStream.read(reinterpret_cast<char*>(allCompDataRead.data()), compFileSize);
    signStream.read(reinterpret_cast<char*>(allSignDataRead.data()), signFileSize);

    size_t unpredSizeBytes = (totalBlocks * config.unpredSizeBits + 7) / 8;
    std::vector<uint8_t> compressedUnpredSizes(allUnpredDataRead.begin(),
                                              allUnpredDataRead.begin() + unpredSizeBytes);
    std::vector<size_t> unpredSizes;
    unpackBitsInline(compressedUnpredSizes, unpredSizes, config.unpredSizeBits, totalBlocks);

    std::vector<uint8_t> bitCounts(totalBlocks);
    std::memcpy(bitCounts.data(), allCompDataRead.data(), totalBlocks);

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

    std::vector<size_t> unpredOffsets(totalBlocks, 0);
    std::vector<size_t> compOffsets(totalBlocks, 0);

    for (size_t i = 1; i < totalBlocks; ++i) {
        unpredOffsets[i] = unpredOffsets[i - 1] + unpredSizes[i - 1];
        compOffsets[i] = compOffsets[i - 1] + compSizes[i - 1];
    }

    std::vector<double> unpredData;
    std::vector<uint8_t> compData;
    std::vector<uint8_t> signBits(config.signBytesPerBlock);

    for (size_t bid : blockIds) {
        size_t unpredSize = unpredSizes[bid];
        size_t unpredOffset = unpredSizeBytes + sizeof(double) * unpredOffsets[bid];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredDataRead.data() + unpredOffset,
                   unpredSize * sizeof(double));

        uint8_t bitCount = bitCounts[bid];
        size_t dataSize = compSizes[bid];
        size_t compOffset = totalBlocks + totalBlocks * config.dataSizeBytes + compOffsets[bid];
        compData.resize(dataSize);
        std::memcpy(compData.data(), allCompDataRead.data() + compOffset, dataSize);

        size_t signOffset = config.signBytesPerBlock * bid;
        std::memcpy(signBits.data(), allSignDataRead.data() + signOffset, config.signBytesPerBlock);

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

// ============================================================
// 以下：Marching Cubes 相关函数（跟索引结构无关，原样保留）
// ============================================================

inline const double* getNeighborBlockPointer(
    size_t blockId,
    int dx, int dy, int dz,
    const std::unordered_map<size_t, const double*>& blockMap,
    const std::vector<size_t>& blockCountPerDim)
{
    size_t bz = blockId % blockCountPerDim[2];
    size_t remaining = blockId / blockCountPerDim[2];
    size_t by = remaining % blockCountPerDim[1];
    size_t bx = remaining / blockCountPerDim[1];

    int nbr_x = static_cast<int>(bx) + dx;
    int nbr_y = static_cast<int>(by) + dy;
    int nbr_z = static_cast<int>(bz) + dz;

    if (nbr_x < 0 || nbr_x >= static_cast<int>(blockCountPerDim[0]) ||
        nbr_y < 0 || nbr_y >= static_cast<int>(blockCountPerDim[1]) ||
        nbr_z < 0 || nbr_z >= static_cast<int>(blockCountPerDim[2])) {
        return nullptr;
    }

    size_t nbrId = static_cast<size_t>(nbr_x) * blockCountPerDim[1] * blockCountPerDim[2] +
                   static_cast<size_t>(nbr_y) * blockCountPerDim[2] +
                   static_cast<size_t>(nbr_z);

    auto it = blockMap.find(nbrId);
    return (it != blockMap.end()) ? it->second : nullptr;
}

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

util::TriangleMesh<double> RunMarchingCubesOnDecompressedBlocks(
    const std::vector<std::vector<double>>& decompressedBlocks,
    const std::vector<size_t>& expandGlobalSmallBlockIds,
    const std::vector<size_t>& localGlobalSmallBlockIds,
    double isovalue,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& dataShape)
{
    std::vector<size_t> blockCountOnEachDim(3);
    for (size_t i = 0; i < 3; i++) {
        blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];
    }

    size_t Bx = blockShape[0];
    size_t By = blockShape[1];
    size_t Bz = blockShape[2];

    std::unordered_map<size_t, const double*> blockMap;
    blockMap.reserve(expandGlobalSmallBlockIds.size());
    for (size_t i = 0; i < expandGlobalSmallBlockIds.size(); i++) {
        blockMap[expandGlobalSmallBlockIds[i]] = decompressedBlocks[i].data();
    }

    std::vector<size_t> sortedBlockIds = expandGlobalSmallBlockIds;
    std::sort(sortedBlockIds.begin(), sortedBlockIds.end());

    std::vector<std::array<double, 3>> globalPoints;
    std::vector<std::array<double, 3>> globalNormals;
    std::vector<std::array<int, 3>> globalTriangles;

    size_t estimatedPoints = decompressedBlocks.size() * (Bx-1) * (By-1) * (Bz-1) * 6;
    globalPoints.reserve(estimatedPoints);
    globalNormals.reserve(estimatedPoints);
    globalTriangles.reserve(estimatedPoints / 3);

    int totalVertexOffset = 0;

    for (size_t globalBlockId : sortedBlockIds) {

        const double* blockData = blockMap[globalBlockId];

        size_t block_z = globalBlockId % blockCountOnEachDim[2];
        size_t remaining = globalBlockId / blockCountOnEachDim[2];
        size_t block_y = remaining % blockCountOnEachDim[1];
        size_t block_x = remaining / blockCountOnEachDim[1];

        size_t global_x_start = block_x * Bx;
        size_t global_y_start = block_y * By;
        size_t global_z_start = block_z * Bz;

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

        if (neighbor_xplus != nullptr && max_x == Bx) {
            for (size_t lz = 0; lz < internal_max_z; ++lz) {
                for (size_t ly = 0; ly < internal_max_y; ++ly) {
                    size_t gx = global_x_start + (Bx - 1);
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + lz;
                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) continue;
                    processBoundaryXPlusCube(blockData, neighbor_xplus, ly, lz, Bx, By,
                                            gx, gy, gz, isovalue, localPoints, localNormals,
                                            localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        if (neighbor_yplus != nullptr && max_y == By) {
            for (size_t lz = 0; lz < internal_max_z; ++lz) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + (By - 1);
                    size_t gz = global_z_start + lz;
                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) continue;
                    processBoundaryYPlusCube(blockData, neighbor_yplus, lx, lz, Bx, By,
                                            gx, gy, gz, isovalue, localPoints, localNormals,
                                            localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        if (neighbor_zplus != nullptr && max_z == Bz) {
            for (size_t ly = 0; ly < internal_max_y; ++ly) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + (Bz - 1);
                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) continue;
                    processBoundaryZPlusCube(blockData, neighbor_zplus, lx, ly, Bx, By,
                                            gx, gy, gz, isovalue, localPoints, localNormals,
                                            localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        if (neighbor_xplus != nullptr && neighbor_yplus != nullptr &&
            max_x == Bx && max_y == By) {
            const double* neighbor_xy = getNeighborBlockPointer(globalBlockId, 1, 1, 0,
                                                               blockMap, blockCountOnEachDim);
            if (neighbor_xy != nullptr) {
                for (size_t lz = 0; lz < internal_max_z; ++lz) {
                    size_t gx = global_x_start + (Bx - 1);
                    size_t gy = global_y_start + (By - 1);
                    size_t gz = global_z_start + lz;
                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) continue;
                    processEdgeXYCube(blockData, neighbor_xplus, neighbor_yplus, neighbor_xy,
                                     lz, Bx, By, gx, gy, gz, isovalue,
                                     localPoints, localNormals, localTriangles,
                                     localPointMap, localPtIdx);
                }
            }
        }

        if (neighbor_xplus != nullptr && neighbor_zplus != nullptr &&
            max_x == Bx && max_z == Bz) {
            const double* neighbor_xz = getNeighborBlockPointer(globalBlockId, 1, 0, 1,
                                                               blockMap, blockCountOnEachDim);
            if (neighbor_xz != nullptr) {
                for (size_t ly = 0; ly < internal_max_y; ++ly) {
                    size_t gx = global_x_start + (Bx - 1);
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + (Bz - 1);
                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) continue;
                    processEdgeXZCube(blockData, neighbor_xplus, neighbor_zplus, neighbor_xz,
                                     ly, Bx, By, gx, gy, gz, isovalue,
                                     localPoints, localNormals, localTriangles,
                                     localPointMap, localPtIdx);
                }
            }
        }

        if (neighbor_yplus != nullptr && neighbor_zplus != nullptr &&
            max_y == By && max_z == Bz) {
            const double* neighbor_yz = getNeighborBlockPointer(globalBlockId, 0, 1, 1,
                                                               blockMap, blockCountOnEachDim);
            if (neighbor_yz != nullptr) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + (By - 1);
                    size_t gz = global_z_start + (Bz - 1);
                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) continue;
                    processEdgeYZCube(blockData, neighbor_yplus, neighbor_zplus, neighbor_yz,
                                     lx, Bx, By, gx, gy, gz, isovalue,
                                     localPoints, localNormals, localTriangles,
                                     localPointMap, localPtIdx);
                }
            }
        }

        if (neighbor_xplus != nullptr && neighbor_yplus != nullptr && neighbor_zplus != nullptr &&
            max_x == Bx && max_y == By && max_z == Bz) {
            const double* neighbor_xy = getNeighborBlockPointer(globalBlockId, 1, 1, 0, blockMap, blockCountOnEachDim);
            const double* neighbor_xz = getNeighborBlockPointer(globalBlockId, 1, 0, 1, blockMap, blockCountOnEachDim);
            const double* neighbor_yz = getNeighborBlockPointer(globalBlockId, 0, 1, 1, blockMap, blockCountOnEachDim);
            const double* neighbor_xyz = getNeighborBlockPointer(globalBlockId, 1, 1, 1, blockMap, blockCountOnEachDim);

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
    const std::vector<size_t>& expandedBlockIds,
    const std::vector<size_t>& localGlobalSmallBlockIds,
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
        expandedBlockIds,
        localGlobalSmallBlockIds,
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

// ============================================================
// main()
// ============================================================
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

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc) {
            inputFileName = argv[++i];
        } else if (arg == "--variable_name" && i + 1 < argc) {
            variableName = argv[++i];
        } else if (arg == "--dimensions" && i + 1 < argc) {
            nDim = std::stoul(argv[++i]);
        } else if (arg == "--begin_step" && i + 1 < argc) {
            beginStepNum = std::stoul(argv[++i]);
        } else if (arg == "--end_step" && i + 1 < argc) {
            endStepNum = std::stoul(argv[++i]);
        } else if (arg == "--stepData_shape" && i + nDim < argc) {
            for (size_t j = 0; j < nDim; j++) stepDataShape.push_back(std::stoul(argv[++i]));
        } else if (arg == "--bigBlock_shape" && i + nDim < argc) {
            for (size_t j = 0; j < nDim; j++) blockShape.push_back(std::stoul(argv[++i]));
        } else if (arg == "--small_block_shape") {
            if (nDim) {
                if ((int)(i + nDim) < argc) {
                    for (size_t j = i + 1; j < i + 1 + nDim; j++) smallBlockShape.push_back(atoi(argv[j]));
                } else {
                    std::cerr << "--small_block_shape option requires [# of dimensions] argument." << std::endl;
                    return 1;
                }
            }
        } else if (arg == "--query_range" && i + 2 < argc) {
            queryRange.push_back(std::stod(argv[++i]));
            queryRange.push_back(std::stod(argv[++i]));
        } else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        }
    }

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

    // [Hilbert2] 索引目录：跟压缩阶段 octree_2index_compress.cc 里的 subDir_uniform/subDir_stagger 保持一致（_new后缀）
    indexDir = "/expanse/lustre/scratch/sdi/temp_project/octree_index_compressed/" + inputFileBaseName + "_" + safeVarName + "_"
             + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
             + std::to_string(extraValue) + "_newour_octree_hilbert_stagger1_index_new/";

    std::string staggerIndexDir = "/expanse/lustre/scratch/sdi/temp_project/octree_index_compressed/" + inputFileBaseName + "_" + safeVarName + "_"
                                + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                                + std::to_string(extraValue) + "_newour_octree_hilbert_stagger2_index_new/";

    // 均匀/错位分块数：跟压缩阶段保持一致
    std::vector<size_t> halfBlockShape(nDim);
    std::vector<size_t> staggerBlockCount(nDim);
    std::vector<size_t> uniformBlockCount(nDim);
    for (size_t d = 0; d < nDim; d++) {
        halfBlockShape[d]    = smallBlockShape[d] / 2;
        uniformBlockCount[d] = stepDataShape[d] / smallBlockShape[d];
        staggerBlockCount[d] = uniformBlockCount[d];
    }

    std::string mySubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + safeVarName + "_"
                          + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                          + std::to_string(extraValue) + "_blockZFP_originalDataCompression/";

    auto afterProcessTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> preProcessTime = afterProcessTime - totalStart;
    std::cerr << "[Rank " << mpi_rank << "[Time1] : preProcessTime: " << preProcessTime.count() << " seconds" << std::endl;

    // ============================================================
    // 阶段一：大块粗筛（跟索引结构无关，不变）
    // ============================================================
    std::vector<ScidxInterval<double>> allBigBlockIndices = loadBigBlockIndexFile(indexDir + "big_block_minmax");
    std::vector<ScidxInterval<double>> allStaggerBigBlockIndices =
        loadBigBlockIndexFile(staggerIndexDir + "big_block_minmax");

    std::cout << "[Debug] uniform big_block_minmax entries: " << allBigBlockIndices.size() << "\n";
    std::cout << "[Debug] stagger big_block_minmax entries: " << allStaggerBigBlockIndices.size() << "\n";

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

    std::unordered_set<size_t> selectedStepsSet;
    for (size_t i = 0; i < allBigBlockIndices.size(); i++) {
        if (isovalue <= allBigBlockIndices[i].high && isovalue >= allBigBlockIndices[i].low) {
            selectedStepsSet.insert(i);
        }
    }
    for (size_t i = 0; i < allStaggerBigBlockIndices.size(); i++) {
        if (isovalue <= allStaggerBigBlockIndices[i].high && isovalue >= allStaggerBigBlockIndices[i].low) {
            selectedStepsSet.insert(i);
        }
    }
    std::vector<size_t> selectedSteps(selectedStepsSet.begin(), selectedStepsSet.end());
    std::sort(selectedSteps.begin(), selectedSteps.end());
    std::cout << "Selected Steps (uniform+stagger union): " << selectedSteps.size() << std::endl;

    auto afterQueryBigBlock = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readAndQueryBigBlock = afterQueryBigBlock - afterProcessTime;
    std::cerr << "[Rank " << mpi_rank << "[Time2] : read and query bigBlocks: " << readAndQueryBigBlock.count() << " seconds" << std::endl;

    // ============================================================
    // 阶段二+三+四：Octree 查询 + 坐标转换 + 合并去重
    // ============================================================
    std::unordered_set<size_t> hitUniformBlocksSet;
    std::chrono::duration<double> mergeTime;
    for (size_t stepRelIdx : selectedSteps) {
        size_t actualStepNum = beginStepNum + stepRelIdx;
        size_t stepOffset    = stepRelIdx * smallBlocksPerStep;

        // [Hilbert2] UNIFORM树解压，同时拿到 listLow / listHigh
        auto uniformResult = queryOctreeUniformRawIdsWithTree(
            actualStepNum, indexDir, queryRange, error_bound, uniformBlockCount);
        std::vector<size_t> uniformRawIds = std::get<0>(uniformResult);
        std::vector<OctreeNode<double>> uniformOctree = std::get<1>(uniformResult);
        std::vector<double> listLow  = std::get<2>(uniformResult);
        std::vector<double> listHigh = std::get<3>(uniformResult);

        // [Hilbert2] STAGGER 查询，用 UNIFORM 现成的 listLow/listHigh
        std::vector<size_t> staggerRawIds = queryOctreeStaggerRawIds(
            actualStepNum, staggerIndexDir, queryRange, error_bound,
            staggerBlockCount, uniformBlockCount, listLow, listHigh);

        auto time_before_merge = std::chrono::high_resolution_clock::now();

        for (size_t rawId : uniformRawIds)
            if (rawId < smallBlocksPerStep) {
                //size_t zid = convertXFastestToZFastest(rawId, uniformBlockCount);
                hitUniformBlocksSet.insert(stepOffset + rawId);
        }

        for (size_t staggerId : staggerRawIds) {
            auto uniformIds = convertStaggerToUniformIds(
                staggerId, stepDataShape, smallBlockShape,
                halfBlockShape, staggerBlockCount, uniformBlockCount);
            for (size_t uid : uniformIds)
                if (uid < smallBlocksPerStep) {
                    //size_t zid = convertXFastestToZFastest(uid, uniformBlockCount);
                    hitUniformBlocksSet.insert(stepOffset + uid);
                }
        }

        auto time_after_merge = std::chrono::high_resolution_clock::now();
        mergeTime = time_after_merge - time_before_merge;
        std::cerr << "[Time_merge] ID mapping + insert + dedup: "
                << mergeTime.count() << " seconds" << std::endl;
    }

    auto time_before_sort = std::chrono::high_resolution_clock::now();
    std::vector<size_t> localGlobalSmallBlockIds(
        hitUniformBlocksSet.begin(), hitUniformBlocksSet.end());
    std::sort(localGlobalSmallBlockIds.begin(), localGlobalSmallBlockIds.end());
    auto time_after_sort = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sortTime = time_after_sort - time_before_sort;
    std::cerr << "[Time_sort] set->vector + sort: " << sortTime.count() << " seconds" << std::endl;

    std::cerr << "[CHECK] Combined hit uniform blocks (deduped): "
            << localGlobalSmallBlockIds.size() << std::endl;

    for (size_t bid : localGlobalSmallBlockIds) {
        if (bid >= totalBlocksNumber) {
            std::cerr << "[BUG] bid " << bid << " >= totalBlocksNumber = " << totalBlocksNumber << std::endl;
            exit(1);
        }
    }

    // ============================================================
    // 阶段五：邻居扩展（跟索引结构无关，不变）
    // ============================================================
    auto startExpansion = std::chrono::high_resolution_clock::now();

    std::vector<size_t> blockCountOnEachDim(3);
    for (size_t i = 0; i < 3; i++) {
        blockCountOnEachDim[i] = (stepDataShape[i] + smallBlockShape[i] - 1) / smallBlockShape[i];
    }

    std::cout << "[Info] Block grid layout: "
              << blockCountOnEachDim[0] << " x "
              << blockCountOnEachDim[1] << " x "
              << blockCountOnEachDim[2] << " = "
              << (blockCountOnEachDim[0] * blockCountOnEachDim[1] * blockCountOnEachDim[2])
              << " total blocks" << std::endl;

    std::unordered_set<size_t> allBlocksSet;
    for (size_t coreId : localGlobalSmallBlockIds) {
        allBlocksSet.insert(coreId);
    }

    size_t originalCoreCount = localGlobalSmallBlockIds.size();
    std::cout << "[Info] Expanding 26 neighbors for " << originalCoreCount << " core blocks..." << std::endl;

    struct NeighborDir { int dx, dy, dz; };
    std::vector<NeighborDir> neighborDirections;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                neighborDirections.push_back({dx, dy, dz});
            }

    for (size_t coreBlockId : localGlobalSmallBlockIds) {
        size_t block_z = coreBlockId % blockCountOnEachDim[2];
        size_t remaining = coreBlockId / blockCountOnEachDim[2];
        size_t block_y = remaining % blockCountOnEachDim[1];
        size_t block_x = remaining / blockCountOnEachDim[1];

        for (const auto& dir : neighborDirections) {
            int neighbor_x = static_cast<int>(block_x) + dir.dx;
            int neighbor_y = static_cast<int>(block_y) + dir.dy;
            int neighbor_z = static_cast<int>(block_z) + dir.dz;

            if (neighbor_x < 0 || neighbor_x >= static_cast<int>(blockCountOnEachDim[0]) ||
                neighbor_y < 0 || neighbor_y >= static_cast<int>(blockCountOnEachDim[1]) ||
                neighbor_z < 0 || neighbor_z >= static_cast<int>(blockCountOnEachDim[2])) {
                continue;
            }

            size_t neighborId = static_cast<size_t>(neighbor_x) * blockCountOnEachDim[1] * blockCountOnEachDim[2] +
                               static_cast<size_t>(neighbor_y) * blockCountOnEachDim[2] +
                               static_cast<size_t>(neighbor_z);

            allBlocksSet.insert(neighborId);
        }
    }

    std::vector<size_t> expandedBlockIds(allBlocksSet.begin(), allBlocksSet.end());

    auto endExpansion = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> expansionTime = endExpansion - startExpansion;
    size_t addedNeighbors = expandedBlockIds.size() - originalCoreCount;

    std::cout << "========== Neighbor Expansion Statistics ==========" << std::endl;
    std::cout << "Core blocks (from index):  " << originalCoreCount << std::endl;
    std::cout << "Neighbor blocks added:     " << addedNeighbors << std::endl;
    std::cout << "Total blocks to decompress: " << expandedBlockIds.size() << std::endl;
    std::cout << "Expansion time:            " << expansionTime.count() << " seconds" << std::endl;
    std::cout << "====================================================" << std::endl;

    // ============================================================
    // 阶段六：解压原始数据 + Marching Cubes（跟索引结构无关，不变）
    // ============================================================
    std::vector<std::vector<double>> decompressedTargetOriginalBlockdata =
        universalBatchDecompressBlocks(
            mySubDir,
            expandedBlockIds,
            smallBlockSize,
            totalBlocksNumber,
            error_bound
        );

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

        std::vector<size_t> stepExpandedBlockIds;
        std::vector<size_t> stepLocalGlobalSmallBlockIds;

        for (size_t id : expandedBlockIds) {
            if (id >= stepOffset && id < stepOffset + smallBlocksPerStep) {
                stepExpandedBlockIds.push_back(id);
            }
        }
        for (size_t id : localGlobalSmallBlockIds) {
            if (id >= stepOffset && id < stepOffset + smallBlocksPerStep) {
                stepLocalGlobalSmallBlockIds.push_back(id);
            }
        }

        std::cout << "[Info] Step " << currentStepNum << " expanded blocks: "
                << stepExpandedBlockIds.size() << std::endl;
        std::cout << "[Info] Step " << currentStepNum << " core blocks: "
                << stepLocalGlobalSmallBlockIds.size() << std::endl;

        std::vector<std::vector<double>> stepBlocks;
        stepBlocks.reserve(stepExpandedBlockIds.size());

        std::unordered_map<size_t, size_t> blockIdToIndex;
        for (size_t i = 0; i < expandedBlockIds.size(); ++i) {
            blockIdToIndex[expandedBlockIds[i]] = i;
        }

        for (size_t blockId : stepExpandedBlockIds) {
            auto it = blockIdToIndex.find(blockId);
            if (it != blockIdToIndex.end()) {
                stepBlocks.push_back(decompressedTargetOriginalBlockdata[it->second]);
            } else {
                std::cerr << "Error: Block " << blockId << " not found in decompressed data!" << std::endl;
            }
        }

        std::cout << "[Info] Extracted " << stepBlocks.size() << " blocks for Step "
                << currentStepNum << std::endl;

        std::string outFile = "/expanse/lustre/scratch/sdi/temp_project/iso_results/new_octree_isosurface_step_"
            + std::to_string(currentStepNum)
            + "_iso_" + std::to_string(queryRange[0]) + ".vtk";

        double iso_value = queryRange[0];

        RunAndSaveIsosurfaceMesh(
            stepBlocks,
            stepExpandedBlockIds,
            stepLocalGlobalSmallBlockIds,
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
// ============================================================================
// HDF5 Octree/Hilbert2 双索引（uniform + stagger）查询 + 26 邻居扩展
// + gapless Marching Cubes，float 版本，单 step 结构
//
// 对接关系：
//   - 索引来源：doc14（HDF5 float32 Octree/Hilbert2 索引构建器），目录
//     *_hdf5_octree_hilbert_uniform_index/ 和 *_hdf5_octree_hilbert_stagger_index/，
//     错位分块用"新规则"（半块与第一个整块合并成新block0），与 doc12/doc14 一致。
//   - Octree/Hilbert2 查询逻辑：float 化后照抄 doc13
//     （decompressOctreeUniformHilbertNew / decompressOctreeStaggerHilbertNew /
//      queryOctree），STAGGER 查询需要 UNIFORM 解压时顺带产出的 listLow/listHigh
//      做8邻居平均预测。
//   - stagger->uniform 坐标转换：doc13 里的"新规则"版 convertStaggerToUniformIds，
//     staggerBlockCount[d] = uniformBlockCount[d]（不再 +1）。
//   - 原始数据压缩格式：doc9 对接的 fixed_*.bin（固定块大小），
//     float 版 batchDecompressBlocksSelective。
//   - 26 邻居扩展 + Marching Cubes 主体：float 化后照抄 doc9/doc6 的做法，
//     遍历"核心块+26邻居"的扩展集合，消除块边界缝隙。
//
// 命令行参数与 doc9 / HDF5双索引AVL查询程序一致（单个逻辑 step）：
//   --input_file <hdf5路径，仅用于目录命名>
//   --variable_name <group/dataset>
//   --dimensions 3
//   --stepData_shape X Y Z
//   --small_block_shape bx by bz
//   --begin_step N --end_step N
//   --query_range low high
//   --relative_error 1e-3 extraValue
// ============================================================================

#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <array>
#include <scidx_octree_interval.h>   // OctreeNode, buildOctree, queryOctree
#include <scidx_octree.h>            // 保留，供对比使用
#include <scidx_octree_hilbert2.h>   // decompressOctreeUniformHilbertNew / decompressOctreeStaggerHilbertNew
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


void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
}

// ============================================================
// 加载 big_block_minmax（float 版，doc14 输出格式：单 step 一对 min/max）
// ============================================================
std::vector<ScidxInterval<float>> loadBigBlockIndexFile(const std::string& filename) {
    std::vector<ScidxInterval<float>> bigBlockIndices;

    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return bigBlockIndices;
    }

    float minVal, maxVal;
    while (inFile.read(reinterpret_cast<char*>(&minVal), sizeof(float)) &&
           inFile.read(reinterpret_cast<char*>(&maxVal), sizeof(float))) {
        bigBlockIndices.push_back({minVal, maxVal});
    }

    inFile.close();
    return bigBlockIndices;
}

// ============================================================
// [Hilbert2] 查询 UNIFORM 树（float 版），同时把 listLow / listHigh 一并返回
// 供 STAGGER 查询用（UNIFORM 解压时已直接产出，不需要从树里反抠）
// ============================================================
std::tuple<std::vector<size_t>, std::vector<OctreeNode<float>>, std::vector<float>, std::vector<float>>
queryOctreeUniformRawIdsWithTree(
    size_t actualStepNum,
    const std::string& indexDir,
    const std::vector<float>& queryRange,
    float error_bound,
    const std::vector<size_t>& blockCountOnEachDim)
{
    std::string treeID = indexDir + std::to_string(actualStepNum) + "-0";
    std::cout << "[queryOctreeUniformRawIds] treeID: " << treeID << std::endl;

    auto time_before_decompress = std::chrono::high_resolution_clock::now();

    std::vector<float> listLow, listHigh;
    std::vector<OctreeNode<float>> octree =
        decompressOctreeUniformHilbertNew<float>(treeID, error_bound, blockCountOnEachDim,
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
// [Hilbert2] 查询 STAGGER 树（float 版），需要传入 UNIFORM 的 listLow/listHigh
// ============================================================
std::vector<size_t> queryOctreeStaggerRawIds(
    size_t actualStepNum,
    const std::string& staggerIndexDir,
    const std::vector<float>& queryRange,
    float error_bound,
    const std::vector<size_t>& staggerBlockCountOnEachDim,
    const std::vector<size_t>& uniformBlockCountOnEachDim,
    const std::vector<float>& listLow,
    const std::vector<float>& listHigh)
{
    std::string staggerTreeID = staggerIndexDir + std::to_string(actualStepNum) + "-0";
    std::cout << "[queryOctreeStaggerRawIds] treeID: " << staggerTreeID << std::endl;

    auto time_before_decompress = std::chrono::high_resolution_clock::now();

    std::vector<OctreeNode<float>> staggerOctree =
        decompressOctreeStaggerHilbertNew<float>(
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
// 错位块 flat ID -> 覆盖的均匀块 flat ID 集合（"新规则"，与 doc14/doc12 一致）
//
//   k=0  : [0, half+block)                       // 半块与第一个整块合并
//   k>=1 : [half + k*block, half + (k+1)*block) ∩ [0, shape)
//
// staggerBlockCount[d] = uniformBlockCount[d]（不再 +1）
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
// 选择性解压：对接 fixed_*.bin 格式（doc9 一致，float 版）
// ============================================================
std::vector<std::vector<float>> batchDecompressBlocksSelective(
    const std::string& subDir,
    const std::vector<size_t>& blockIds,
    size_t blockSize,
    size_t totalBlocks,
    float error_bound) {

    auto start_time_read = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> decompressedOriginalresult;

    std::string unpredDataFileName = subDir + "fixed_unpredData.bin";
    std::string compressedFileName = subDir + "fixed_compressed_data.bin";
    std::string signFileName = subDir + "fixed_sign_data.bin";

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

    std::vector<unsigned char> allUnpredData(unpredFileSize);
    std::vector<unsigned char> allCompData(compFileSize);
    std::vector<unsigned char> allSignData(signFileSize);

    unpredStream.read(reinterpret_cast<char*>(allUnpredData.data()), unpredFileSize);
    compStream.read(reinterpret_cast<char*>(allCompData.data()), compFileSize);
    signStream.read(reinterpret_cast<char*>(allSignData.data()), signFileSize);

    std::vector<unsigned char> unpredSizes(totalBlocks);
    std::memcpy(unpredSizes.data(), allUnpredData.data(), totalBlocks);

    std::vector<unsigned char> bitCounts(totalBlocks);
    std::vector<unsigned char> compSizes(totalBlocks);
    std::memcpy(bitCounts.data(), allCompData.data(), totalBlocks);
    std::memcpy(compSizes.data(), allCompData.data() + totalBlocks, totalBlocks);

    const size_t signBytesPerBlock = (blockSize + 7) / 8;

    auto begin_decompress = std::chrono::high_resolution_clock::now();

    std::vector<size_t> unpredOffsets(totalBlocks, 0);
    std::vector<size_t> compOffsets(totalBlocks, 0);
    for (size_t i = 1; i < totalBlocks; ++i) {
        unpredOffsets[i] = unpredOffsets[i - 1] + unpredSizes[i - 1];
        compOffsets[i] = compOffsets[i - 1] + compSizes[i - 1];
    }

    std::vector<float> unpredData;
    std::vector<unsigned char> compData;
    std::vector<unsigned char> signBits(signBytesPerBlock);
    std::vector<int> quant_inds(blockSize);
    std::vector<float> decompressedBlock(blockSize);

    int radius = 512;

    for (size_t bid : blockIds) {
        size_t local_id = bid;

        unsigned char unpredSize = unpredSizes[local_id];
        size_t unpredOffset = totalBlocks + sizeof(float) * unpredOffsets[local_id];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredData.data() + unpredOffset, unpredSize * sizeof(float));

        unsigned char bitCount = bitCounts[local_id];
        unsigned char dataSize = compSizes[local_id];
        size_t compOffset = 2 * totalBlocks + compOffsets[local_id];
        compData.resize(dataSize);
        std::memcpy(compData.data(), allCompData.data() + compOffset, dataSize);

        size_t signOffset = signBytesPerBlock * local_id;
        std::memcpy(signBits.data(), allSignData.data() + signOffset, signBytesPerBlock);

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

        double error_bound_double = static_cast<double>(error_bound);

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

    std::cerr << "[Time6]: read all compressed small blocks time: " << readTime.count() << " seconds" << std::endl;
    std::cerr << "[Time7]: decompress small blocks time: " << compressTime.count() << " seconds" << std::endl;

    return decompressedOriginalresult;
}


// ============================================================================
// 邻居块指针查找（float 版，与 doc9 一致）
// ============================================================================
inline const float* getNeighborBlockPointer(
    size_t blockId,
    int dx, int dy, int dz,
    const std::unordered_map<size_t, const float*>& blockMap,
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

// ============================================================================
// Marching Cubes 辅助函数（float 版，逐字对应 doc9）
// ============================================================================
inline void processInternalCube(
    const float* blockData,
    size_t local_x, size_t local_y, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t base = local_x + local_y * Bx + local_z * Bx * By;

    std::array<float, 8> cubeValues = {{
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

    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};

    std::array<std::array<float, 3>, 8> cubeGradients;
    for (int i = 0; i < 8; ++i) {
        size_t px = local_x + (i & 1);
        size_t py = local_y + ((i >> 1) & 1);
        size_t pz = local_z + ((i >> 2) & 1);

        size_t px_safe = std::min(px, Bx - 1);
        size_t py_safe = std::min(py, By - 1);
        size_t pz_safe = std::min(pz, Bx - 1);

        std::array<float, 3> grad = {0.0f, 0.0f, 0.0f};

        if (px_safe == 0) {
            grad[0] = blockData[px_safe + py_safe*Bx + pz_safe*Bx*By] -
                      blockData[(px_safe+1) + py_safe*Bx + pz_safe*Bx*By];
        } else if (px_safe == Bx - 1) {
            grad[0] = blockData[(px_safe-1) + py_safe*Bx + pz_safe*Bx*By] -
                      blockData[px_safe + py_safe*Bx + pz_safe*Bx*By];
        } else {
            grad[0] = (blockData[(px_safe-1) + py_safe*Bx + pz_safe*Bx*By] -
                       blockData[(px_safe+1) + py_safe*Bx + pz_safe*Bx*By]) / 2.0f;
        }

        if (py_safe == 0) {
            grad[1] = blockData[px_safe + py_safe*Bx + pz_safe*Bx*By] -
                      blockData[px_safe + (py_safe+1)*Bx + pz_safe*Bx*By];
        } else if (py_safe == By - 1) {
            grad[1] = blockData[px_safe + (py_safe-1)*Bx + pz_safe*Bx*By] -
                      blockData[px_safe + py_safe*Bx + pz_safe*Bx*By];
        } else {
            grad[1] = (blockData[px_safe + (py_safe-1)*Bx + pz_safe*Bx*By] -
                       blockData[px_safe + (py_safe+1)*Bx + pz_safe*Bx*By]) / 2.0f;
        }

        if (pz_safe == 0) {
            grad[2] = blockData[px_safe + py_safe*Bx + pz_safe*Bx*By] -
                      blockData[px_safe + py_safe*Bx + (pz_safe+1)*Bx*By];
        } else if (pz_safe == Bx - 1) {
            grad[2] = blockData[px_safe + py_safe*Bx + (pz_safe-1)*Bx*By] -
                      blockData[px_safe + py_safe*Bx + pz_safe*Bx*By];
        } else {
            grad[2] = (blockData[px_safe + py_safe*Bx + (pz_safe-1)*Bx*By] -
                       blockData[px_safe + py_safe*Bx + (pz_safe+1)*Bx*By]) / 2.0f;
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

                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);

                std::array<float, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
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

inline void processBoundaryXPlusCube(
    const float* blockData,
    const float* neighbor_xplus,
    size_t local_y, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;

    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr = 0 + local_y * Bx + local_z * Bx * By;

    std::array<float, 8> cubeValues = {{
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

    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};

    std::array<std::array<float, 3>, 8> cubeGradients;

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
        const float* dataPtr = vi.in_neighbor ? neighbor_xplus : blockData;

        std::array<float, 3> grad;

        if (vi.in_neighbor) {
            if (vi.px == 0) {
                float val_left  = blockData[local_x + vi.py * Bx + vi.pz * Bx * By];
                float val_right = (vi.px + 1 < Bx) ?
                    neighbor_xplus[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By] : val_left;
                grad[0] = (val_left - val_right) / 2.0f;
            } else {
                grad[0] = (neighbor_xplus[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                           neighbor_xplus[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
            }
        } else {
            if (vi.px == Bx - 1) {
                float val_left  = (vi.px > 0) ?
                    blockData[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] :
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                float val_right = neighbor_xplus[0 + vi.py * Bx + vi.pz * Bx * By];
                grad[0] = (val_left - val_right) / 2.0f;
            } else {
                grad[0] = (blockData[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                           blockData[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
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
                       dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.pz == 0) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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

                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);

                std::array<float, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
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

inline void processBoundaryYPlusCube(
    const float* blockData,
    const float* neighbor_yplus,
    size_t local_x, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_y = By - 1;

    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr  = local_x + 0       * Bx + local_z * Bx * By;

    std::array<float, 8> cubeValues = {{
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

    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};

    std::array<std::array<float, 3>, 8> cubeGradients;

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
        const float* dataPtr = vi.in_neighbor ? neighbor_yplus : blockData;

        std::array<float, 3> grad;

        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.in_neighbor) {
            if (vi.py == 0) {
                float val_left  = blockData[vi.px + local_y * Bx + vi.pz * Bx * By];
                float val_right = (vi.py + 1 < By) ?
                    neighbor_yplus[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By] : val_left;
                grad[1] = (val_left - val_right) / 2.0f;
            } else {
                grad[1] = (neighbor_yplus[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                           neighbor_yplus[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
            }
        } else {
            if (vi.py == By - 1) {
                float val_left  = (vi.py > 0) ?
                    blockData[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] :
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                float val_right = neighbor_yplus[vi.px + 0 * Bx + vi.pz * Bx * By];
                grad[1] = (val_left - val_right) / 2.0f;
            } else {
                grad[1] = (blockData[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                           blockData[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
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
                       dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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

                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);

                std::array<float, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
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

inline void processBoundaryZPlusCube(
    const float* blockData,
    const float* neighbor_zplus,
    size_t local_x, size_t local_y,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_z = Bx - 1;

    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr  = local_x + local_y * Bx + 0 * Bx * By;

    std::array<float, 8> cubeValues = {{
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

    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};

    std::array<std::array<float, 3>, 8> cubeGradients;

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
        const float* dataPtr = vi.in_neighbor ? neighbor_zplus : blockData;

        std::array<float, 3> grad;

        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.py == 0) {
            grad[1] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.in_neighbor) {
            if (vi.pz == 0) {
                float val_left  = blockData[vi.px + vi.py * Bx + local_z * Bx * By];
                float val_right = (vi.pz + 1 < Bx) ?
                    neighbor_zplus[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By] : val_left;
                grad[2] = (val_left - val_right) / 2.0f;
            } else {
                grad[2] = (neighbor_zplus[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                           neighbor_zplus[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
            }
        } else {
            if (vi.pz == Bx - 1) {
                float val_left  = (vi.pz > 0) ?
                    blockData[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] :
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                float val_right = neighbor_zplus[vi.px + vi.py * Bx + 0 * Bx * By];
                grad[2] = (val_left - val_right) / 2.0f;
            } else {
                grad[2] = (blockData[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                           blockData[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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

                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);

                std::array<float, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
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

inline void processEdgeXYCube(
    const float* blockData,
    const float* neighbor_xplus,
    const float* neighbor_yplus,
    const float* neighbor_xy_diagonal,
    size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
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

    std::array<float, 8> cubeValues = {{
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

    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};

    std::array<std::array<float, 3>, 8> cubeGradients;

    struct VertexInfo {
        size_t px, py, pz;
        const float* dataPtr;
        const float* neighbor_x;
        const float* neighbor_y;
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
        std::array<float, 3> grad;

        if (vi.px == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            float val_right = (vi.px + 1 < Bx)
                ? vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[0] = (val_left - val_right) / 2.0f;
        } else if (vi.px == Bx - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = (vi.px > 0)
                ? vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            float val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0f;
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.py == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            float val_up   = (vi.py + 1 < By)
                ? vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[1] = (val_down - val_up) / 2.0f;
        } else if (vi.py == By - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = (vi.py > 0)
                ? vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By]
                : val_curr;
            float val_up   = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0f;
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.pz == 0) {
            grad[2] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                float w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);

                std::array<float, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
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

inline void processEdgeXZCube(
    const float* blockData,
    const float* neighbor_xplus,
    const float* neighbor_zplus,
    const float* neighbor_xz_diagonal,
    size_t local_y,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
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

    std::array<float, 8> cubeValues = {{
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

    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};

    std::array<std::array<float, 3>, 8> cubeGradients;

    struct VertexInfo {
        size_t px, py, pz;
        const float* dataPtr;
        const float* neighbor_x;
        const float* neighbor_z;
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
        std::array<float, 3> grad;

        if (vi.px == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            float val_right = (vi.px + 1 < Bx)
                ? vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[0] = (val_left - val_right) / 2.0f;
        } else if (vi.px == Bx - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = (vi.px > 0)
                ? vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            float val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0f;
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.py == 0) {
            grad[1] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.pz == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            float val_front = (vi.pz + 1 < Bx)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]
                : val_curr;
            grad[2] = (val_back - val_front) / 2.0f;
        } else if (vi.pz == Bx - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = (vi.pz > 0)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By]
                : val_curr;
            float val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0f;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                float w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);

                std::array<float, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
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

inline void processEdgeYZCube(
    const float* blockData,
    const float* neighbor_yplus,
    const float* neighbor_zplus,
    const float* neighbor_yz_diagonal,
    size_t local_x,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
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

    std::array<float, 8> cubeValues = {{
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

    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};

    std::array<std::array<float, 3>, 8> cubeGradients;

    struct VertexInfo {
        size_t px, py, pz;
        const float* dataPtr;
        const float* neighbor_y;
        const float* neighbor_z;
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
        std::array<float, 3> grad;

        if (vi.px == 0) {
            grad[0] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.py == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            float val_up   = (vi.py + 1 < By)
                ? vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[1] = (val_down - val_up) / 2.0f;
        } else if (vi.py == By - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = (vi.py > 0)
                ? vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By]
                : val_curr;
            float val_up   = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0f;
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.pz == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            float val_front = (vi.pz + 1 < Bx)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]
                : val_curr;
            grad[2] = (val_back - val_front) / 2.0f;
        } else if (vi.pz == Bx - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = (vi.pz > 0)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By]
                : val_curr;
            float val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0f;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                float w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);

                std::array<float, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
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

inline void processCornerXYZCube(
    const float* blockData,
    const float* neighbor_xplus,
    const float* neighbor_yplus,
    const float* neighbor_zplus,
    const float* neighbor_xy_diagonal,
    const float* neighbor_xz_diagonal,
    const float* neighbor_yz_diagonal,
    const float* neighbor_xyz_diagonal,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
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

    std::array<float, 8> cubeValues = {{
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

    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};

    std::array<std::array<float, 3>, 8> cubeGradients;

    struct VertexInfo {
        size_t px, py, pz;
        const float* dataPtr;
        const float* neighbor_x;
        const float* neighbor_y;
        const float* neighbor_z;
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
        std::array<float, 3> grad;

        if (vi.px == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            float val_right = (vi.px + 1 < Bx)
                ? vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[0] = (val_left - val_right) / 2.0f;
        } else if (vi.px == Bx - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = (vi.px > 0)
                ? vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            float val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0f;
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.py == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            float val_up   = (vi.py + 1 < By)
                ? vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[1] = (val_down - val_up) / 2.0f;
        } else if (vi.py == By - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = (vi.py > 0)
                ? vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By]
                : val_curr;
            float val_up   = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0f;
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }

        if (vi.pz == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            float val_front = (vi.pz + 1 < Bx)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]
                : val_curr;
            grad[2] = (val_back - val_front) / 2.0f;
        } else if (vi.pz == Bx - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = (vi.pz > 0)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By]
                : val_curr;
            float val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0f;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                float w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);

                std::array<float, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
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

// ============================================================================
// 主函数：Marching Cubes（float 版），遍历"扩展块集合"（核心块 + 26 邻居）
// ============================================================================
util::TriangleMesh<float> RunMarchingCubesOnDecompressedBlocks(
    const std::vector<std::vector<float>>& decompressedBlocks,
    const std::vector<size_t>& expandGlobalSmallBlockIds,
    float isovalue,
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

    std::unordered_map<size_t, const float*> blockMap;
    blockMap.reserve(expandGlobalSmallBlockIds.size());
    for (size_t i = 0; i < expandGlobalSmallBlockIds.size(); i++) {
        blockMap[expandGlobalSmallBlockIds[i]] = decompressedBlocks[i].data();
    }

    std::vector<size_t> sortedBlockIds = expandGlobalSmallBlockIds;
    std::sort(sortedBlockIds.begin(), sortedBlockIds.end());

    std::vector<std::array<float, 3>> globalPoints;
    std::vector<std::array<float, 3>> globalNormals;
    std::vector<std::array<int, 3>> globalTriangles;

    size_t estimatedPoints = decompressedBlocks.size() * (Bx-1) * (By-1) * (Bz-1) * 6;
    globalPoints.reserve(estimatedPoints);
    globalNormals.reserve(estimatedPoints);
    globalTriangles.reserve(estimatedPoints / 3);

    int totalVertexOffset = 0;

    for (size_t globalBlockId : sortedBlockIds) {

        const float* blockData = blockMap[globalBlockId];

        size_t block_z = globalBlockId % blockCountOnEachDim[2];
        size_t remaining = globalBlockId / blockCountOnEachDim[2];
        size_t block_y = remaining % blockCountOnEachDim[1];
        size_t block_x = remaining / blockCountOnEachDim[1];

        size_t global_x_start = block_x * Bx;
        size_t global_y_start = block_y * By;
        size_t global_z_start = block_z * Bz;

        const float* neighbor_xplus = getNeighborBlockPointer(globalBlockId, 1, 0, 0, blockMap, blockCountOnEachDim);
        const float* neighbor_yplus = getNeighborBlockPointer(globalBlockId, 0, 1, 0, blockMap, blockCountOnEachDim);
        const float* neighbor_zplus = getNeighborBlockPointer(globalBlockId, 0, 0, 1, blockMap, blockCountOnEachDim);

        size_t max_x = std::min(Bx, dataShape[0] - global_x_start);
        size_t max_y = std::min(By, dataShape[1] - global_y_start);
        size_t max_z = std::min(Bz, dataShape[2] - global_z_start);

        std::vector<std::array<float, 3>> localPoints;
        std::vector<std::array<float, 3>> localNormals;
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

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

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

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

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

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processBoundaryZPlusCube(blockData, neighbor_zplus, lx, ly, Bx, By,
                                            gx, gy, gz, isovalue, localPoints, localNormals,
                                            localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        if (neighbor_xplus != nullptr && neighbor_yplus != nullptr &&
            max_x == Bx && max_y == By) {

            const float* neighbor_xy = getNeighborBlockPointer(globalBlockId, 1, 1, 0,
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

        if (neighbor_xplus != nullptr && neighbor_zplus != nullptr &&
            max_x == Bx && max_z == Bz) {

            const float* neighbor_xz = getNeighborBlockPointer(globalBlockId, 1, 0, 1,
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

        if (neighbor_yplus != nullptr && neighbor_zplus != nullptr &&
            max_y == By && max_z == Bz) {

            const float* neighbor_yz = getNeighborBlockPointer(globalBlockId, 0, 1, 1,
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

        if (neighbor_xplus != nullptr && neighbor_yplus != nullptr && neighbor_zplus != nullptr &&
            max_x == Bx && max_y == By && max_z == Bz) {

            const float* neighbor_xy = getNeighborBlockPointer(globalBlockId, 1, 1, 0,
                                                               blockMap, blockCountOnEachDim);
            const float* neighbor_xz = getNeighborBlockPointer(globalBlockId, 1, 0, 1,
                                                               blockMap, blockCountOnEachDim);
            const float* neighbor_yz = getNeighborBlockPointer(globalBlockId, 0, 1, 1,
                                                               blockMap, blockCountOnEachDim);
            const float* neighbor_xyz = getNeighborBlockPointer(globalBlockId, 1, 1, 1,
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

    return util::TriangleMesh<float>(globalPoints, globalNormals, globalTriangles);
}


void RunAndSaveIsosurfaceMesh(
    const std::vector<std::vector<float>>& decompressedBlocks,
    const std::vector<size_t>& expandGlobalSmallBlockIds,
    float isovalue,
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

    util::TriangleMesh<float> mesh = RunMarchingCubesOnDecompressedBlocks(
        decompressedBlocks,
        expandGlobalSmallBlockIds,
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

    std::string inputFileName, variableName;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0, smallBlockSize = 1, totalBlocksNumber = 1;
    std::vector<float> queryRange;
    float relative_error_bound = 1E-3;
    size_t extraValue = 0;

    // ---- 解析命令行参数（与 doc9 一致，单个逻辑 step）----
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
            for (size_t j = 0; j < nDim; j++) {
                stepDataShape.push_back(std::stoul(argv[++i]));
            }
        } else if (arg == "--small_block_shape") {
            if (nDim) {
                if ((int)(i + nDim) < argc) {
                    for (size_t j = i + 1; j < i + 1 + nDim; j++) {
                        smallBlockShape.push_back(atoi(argv[j]));
                    }
                } else {
                    std::cerr << "--small_block_shape option requires [# of dimensions] argument." << std::endl;
                    MPI_Finalize();
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

    if (nDim == 0 || stepDataShape.size() != nDim || smallBlockShape.size() != nDim) {
        std::cerr << "Error: missing or invalid arguments (--dimensions/--stepData_shape/--small_block_shape)." << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (queryRange.size() != 2) {
        std::cerr << "Error: Invalid query range. Use --query_range <low> <high>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    // ---- 单个逻辑 step（与 doc9 一致的循环结构，通常 nSteps==1）----
    size_t smallBlocksPerStep = 1;
    for (size_t d = 0; d < nDim; d++) {
        smallBlocksPerStep *= stepDataShape[d] / smallBlockShape[d];
        smallBlockSize     *= smallBlockShape[d];
    }
    size_t nSteps = endStepNum - beginStepNum + 1;
    totalBlocksNumber = nSteps * smallBlocksPerStep;

    // ---- doc14/doc12 "新规则" 的错位分块参数 ----
    // staggerBlockCount[d] = uniformBlockCount[d]（不再 +1），
    // 因为开头半块与第一个整块合并进新 block0。
    std::vector<size_t> halfBlockShape(nDim);
    std::vector<size_t> uniformBlockCount(nDim);
    std::vector<size_t> staggerBlockCount(nDim);
    for (size_t d = 0; d < nDim; d++) {
        halfBlockShape[d]    = smallBlockShape[d] / 2;
        uniformBlockCount[d] = stepDataShape[d] / smallBlockShape[d];
        staggerBlockCount[d] = uniformBlockCount[d];
    }

    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    // ---- 索引目录：对接 doc14 的两套输出 ----
    std::string uniformIndexDir = "/expanse/lustre/scratch/sdi/temp_project/octree_index_compressed/"
        + inputFileBaseName + "_" + safeVarName + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_hdf5_octree_hilbert_uniform_index/";

    std::string staggerIndexDir = "/expanse/lustre/scratch/sdi/temp_project/octree_index_compressed/"
        + inputFileBaseName + "_" + safeVarName + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_hdf5_octree_hilbert_stagger_index/";

    // ---- 原始压缩数据目录：对接 doc9 的 fixed_*.bin 格式 ----
    /*std::string mySubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + safeVarName + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_hdf5_fixlength_originalDataCompression/";*/

    std::string mySubDir = "/expanse/lustre/scratch/sdi/temp_project/nyx_original_compress/" 
    + inputFileBaseName + "_" + safeVarName + "_"
    + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
    + std::to_string(extraValue) + "_hdf5_fixlength_originalDataCompression/";

    auto afterProcessTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> preProcessTime = afterProcessTime - totalStart;
    std::cerr << "[Rank " << mpi_rank << "][Time1] : preProcessTime: " << preProcessTime.count() << " seconds" << std::endl;

    // ---- 加载两套索引的 big_block_minmax（step 级粗筛）----
    std::vector<ScidxInterval<float>> uniformBigBlockIndices = loadBigBlockIndexFile(uniformIndexDir + "big_block_minmax");
    std::vector<ScidxInterval<float>> staggerBigBlockIndices = loadBigBlockIndexFile(staggerIndexDir + "big_block_minmax");

    std::cout << "[Debug] uniform big_block_minmax entries: " << uniformBigBlockIndices.size() << "\n";
    std::cout << "[Debug] stagger big_block_minmax entries: " << staggerBigBlockIndices.size() << "\n";

    float globalMin = std::numeric_limits<float>::max();
    float globalMax = std::numeric_limits<float>::lowest();
    for (const auto& interval : uniformBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }

    std::cout << "Global Min: " << globalMin << std::endl;
    std::cout << "Global Max: " << globalMax << std::endl;

    float error_bound = relative_error_bound * (globalMax - globalMin);
    std::cout << "Computed error_bound: " << error_bound << std::endl;

    float isovalue = queryRange[0];

    // ---- step 级粗筛：uniform 和 stagger 命中的 step 取并集 ----
    std::unordered_set<size_t> selectedStepsSet;
    for (size_t i = 0; i < uniformBigBlockIndices.size(); i++) {
        if (isovalue <= uniformBigBlockIndices[i].high && isovalue >= uniformBigBlockIndices[i].low) {
            selectedStepsSet.insert(i);
        }
    }
    for (size_t i = 0; i < staggerBigBlockIndices.size(); i++) {
        if (isovalue <= staggerBigBlockIndices[i].high && isovalue >= staggerBigBlockIndices[i].low) {
            selectedStepsSet.insert(i);
        }
    }
    std::vector<size_t> selectedSteps(selectedStepsSet.begin(), selectedStepsSet.end());
    std::sort(selectedSteps.begin(), selectedSteps.end());
    std::cout << "Selected Steps (uniform+stagger union): " << selectedSteps.size() << std::endl;

    auto afterQueryBigBlock = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readAndQueryBigBlock = afterQueryBigBlock - afterProcessTime;
    std::cerr << "[Rank " << mpi_rank << "][Time2] : read and query bigBlocks: " << readAndQueryBigBlock.count() << " seconds" << std::endl;

    // ---- 精细查询两套 Octree/Hilbert2 树 + stagger->uniform 转换 + 合并去重 ----
    std::unordered_set<size_t> hitUniformBlocksSet;
    for (size_t stepRelIdx : selectedSteps) {
        size_t actualStepNum = beginStepNum + stepRelIdx;
        size_t stepOffset    = stepRelIdx * smallBlocksPerStep;

        // UNIFORM 树解压查询，同时拿到 listLow / listHigh 供 STAGGER 用
        auto uniformResult = queryOctreeUniformRawIdsWithTree(
            actualStepNum, uniformIndexDir, queryRange, error_bound, uniformBlockCount);
        std::vector<size_t> uniformRawIds = std::get<0>(uniformResult);
        std::vector<float> listLow  = std::get<2>(uniformResult);
        std::vector<float> listHigh = std::get<3>(uniformResult);

        // STAGGER 树解压查询，复用 UNIFORM 的 listLow/listHigh
        std::vector<size_t> staggerRawIds = queryOctreeStaggerRawIds(
            actualStepNum, staggerIndexDir, queryRange, error_bound,
            staggerBlockCount, uniformBlockCount, listLow, listHigh);

        auto time_before_merge = std::chrono::high_resolution_clock::now();

        for (size_t rawId : uniformRawIds)
            if (rawId < smallBlocksPerStep)
                hitUniformBlocksSet.insert(stepOffset + rawId);

        for (size_t staggerId : staggerRawIds) {
            auto uniformIds = convertStaggerToUniformIds(
                staggerId, stepDataShape, smallBlockShape,
                halfBlockShape, staggerBlockCount, uniformBlockCount);
            for (size_t uid : uniformIds)
                if (uid < smallBlocksPerStep)
                    hitUniformBlocksSet.insert(stepOffset + uid);
        }

        auto time_after_merge = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> mergeTime = time_after_merge - time_before_merge;
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

    std::cerr << "[CHECK] Combined hit uniform blocks (deduped, core): "
            << localGlobalSmallBlockIds.size() << std::endl;

    for (size_t bid : localGlobalSmallBlockIds) {
        if (bid >= totalBlocksNumber) {
            std::cerr << "[BUG] bid " << bid << " >= totalBlocksNumber = " << totalBlocksNumber << std::endl;
            MPI_Finalize();
            exit(1);
        }
    }

    // ============================================================
    // 26 邻居扩展：核心块 + 周围26个方向的邻居块一起解压，
    // Marching Cubes 时才能在核心块边界找到邻居数据，消除缝隙。
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
    std::sort(expandedBlockIds.begin(), expandedBlockIds.end());

    auto endExpansion = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> expansionTime = endExpansion - startExpansion;

    size_t addedNeighbors = expandedBlockIds.size() - originalCoreCount;
    std::cout << "========== Neighbor Expansion Statistics ==========" << std::endl;
    std::cout << "Core blocks (from index):   " << originalCoreCount << std::endl;
    std::cout << "Neighbor blocks added:      " << addedNeighbors << std::endl;
    std::cout << "Total blocks to decompress: " << expandedBlockIds.size() << std::endl;
    std::cout << "Expansion time:             " << expansionTime.count() << " seconds" << std::endl;
    std::cout << "====================================================" << std::endl;

    // ---- 解压扩展块（对接 doc9 的 fixed_*.bin 格式）----
    std::vector<std::vector<float>> decompressedTargetOriginalBlockdata =
        batchDecompressBlocksSelective(
            mySubDir,
            expandedBlockIds,
            smallBlockSize,
            totalBlocksNumber,
            error_bound
        );

    auto finishDecompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompressTime = finishDecompress - endExpansion;
    std::cerr << "[Time_decompress] decompress expanded blocks: " << decompressTime.count() << " seconds" << std::endl;

    double ratio_block = (double)decompressedTargetOriginalBlockdata.size() / smallBlocksPerStep * 100.0;
    std::cout << "[Info] Blocks per step: " << smallBlocksPerStep << std::endl;
    std::cout << "[Info] Total decompressed (expanded) blocks: " << decompressedTargetOriginalBlockdata.size() << std::endl;
    std::cout << "[Info] Decompression ratio: " << ratio_block << "%" << std::endl;

    // ---- 按 step 切分并跑 Marching Cubes（结构与 doc9 一致，nSteps 通常为1）----
    std::cout << "\n========================================" << std::endl;
    std::cout << "Extracting isosurfaces for " << nSteps << " steps..." << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::unordered_map<size_t, size_t> blockIdToIndex;
    for (size_t i = 0; i < expandedBlockIds.size(); ++i) {
        blockIdToIndex[expandedBlockIds[i]] = i;
    }

    for (size_t step = 0; step < nSteps; ++step) {
        auto stepStart = std::chrono::high_resolution_clock::now();

        size_t currentStepNum = beginStepNum + step;
        size_t stepOffset = step * smallBlocksPerStep;

        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Processing Step " << currentStepNum << std::endl;

        std::vector<size_t> stepExpandedBlockIds;
        for (size_t id : expandedBlockIds) {
            if (id >= stepOffset && id < stepOffset + smallBlocksPerStep) {
                stepExpandedBlockIds.push_back(id);
            }
        }

        std::cout << "[Info] Step " << currentStepNum << " expanded blocks: "
                << stepExpandedBlockIds.size() << std::endl;

        std::vector<std::vector<float>> stepBlocks;
        stepBlocks.reserve(stepExpandedBlockIds.size());
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

        std::string outFile = "/expanse/lustre/scratch/sdi/temp_project/iso_results/isosurface_"
            + safeVarName + "_step_" + std::to_string(currentStepNum)
            + "_iso_" + std::to_string(queryRange[0]) + "_hdf5_octree_dualindex_gapless.vtk";

        float iso_value = queryRange[0];

        RunAndSaveIsosurfaceMesh(
            stepBlocks,
            stepExpandedBlockIds,
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
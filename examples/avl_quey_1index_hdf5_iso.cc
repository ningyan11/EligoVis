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

// ============================================================
// 加载 big_block_minmax（float 版，对接 doc16 索引程序生成的格式）
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

std::vector<size_t> safeQueryOverlapIds(ScidxAVLNode<float>* root, const ScidxInterval<float>& query) {
    if (root && root->id == static_cast<size_t>(-1)) {
        return {};  // dummy 树，返回空
    }
    return queryOverlapIds(root, query);
}

// ============================================================
// 查询单棵AVL树（float版，无大块层，文件名格式为 {stepNum}-0）
// 对接 doc16 索引程序生成的树（uniform only）
// ============================================================
std::vector<size_t> queryIndexRawIds(
    size_t actualStepNum,
    const std::string& indexDir,
    const std::vector<float>& queryRange,
    float error_bound)
{
    std::string treeID = indexDir + std::to_string(actualStepNum) + "-0";

    std::cout << "[queryIndexRawIds] treeID: " << treeID << std::endl;

    auto [decompressedTree, decodedSkippedMin, decodedSkippedMax, skippedIdOut] =
        decompressOptimizedAVL(treeID, error_bound);

    auto time_before_query = std::chrono::high_resolution_clock::now();

    adjustAndUpdateMaxHigh(decompressedTree, error_bound);

    ScidxInterval<float> query;
    query.low  = queryRange[0];
    query.high = queryRange[1];

    auto ids = safeQueryOverlapIds(decompressedTree, query);

    for (size_t i = 0; i < decodedSkippedMin.size(); ++i) {
        if (skippedIdOut[i] == static_cast<size_t>(-1)) continue;
        float minWithError = decodedSkippedMin[i] - error_bound;
        float maxWithError = decodedSkippedMax[i] + error_bound;
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


// ============================================================
// 选择性解压：对接 doc15/doc11 压缩程序生成的 fixed_*.bin 格式
// （固定 blockSize，unpredSizes/bitCounts/compSizes 每块各占1字节，
// 与 doc1 里的 batchDecompressBlocksAllRead 完全一致的格式，float版）
// blockIds 可以是全部块也可以是子集，只有 blockIds 里列出的块会被真正解压
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

    // 全量读取到内存（跟doc1的batchDecompressBlocksAllRead一致）
    std::vector<unsigned char> allUnpredData(unpredFileSize);
    std::vector<unsigned char> allCompData(compFileSize);
    std::vector<unsigned char> allSignData(signFileSize);

    unpredStream.read(reinterpret_cast<char*>(allUnpredData.data()), unpredFileSize);
    compStream.read(reinterpret_cast<char*>(allCompData.data()), compFileSize);
    signStream.read(reinterpret_cast<char*>(allSignData.data()), signFileSize);

    // 读取 unpredSizes (每块1字节)
    std::vector<unsigned char> unpredSizes(totalBlocks);
    std::memcpy(unpredSizes.data(), allUnpredData.data(), totalBlocks);

    // 读取 bitCounts / compSizes (每块各1字节)
    std::vector<unsigned char> bitCounts(totalBlocks);
    std::vector<unsigned char> compSizes(totalBlocks);
    std::memcpy(bitCounts.data(), allCompData.data(), totalBlocks);
    std::memcpy(compSizes.data(), allCompData.data() + totalBlocks, totalBlocks);

    const size_t signBytesPerBlock = (blockSize + 7) / 8;

    auto begin_decompress = std::chrono::high_resolution_clock::now();

    // 提前计算所有偏移
    std::vector<size_t> unpredOffsets(totalBlocks, 0);
    std::vector<size_t> compOffsets(totalBlocks, 0);
    for (size_t i = 1; i < totalBlocks; ++i) {
        unpredOffsets[i] = unpredOffsets[i - 1] + unpredSizes[i - 1];
        compOffsets[i] = compOffsets[i - 1] + compSizes[i - 1];
    }

    // 循环外预分配缓冲区（复用）
    std::vector<float> unpredData;
    std::vector<unsigned char> compData;
    std::vector<unsigned char> signBits(signBytesPerBlock);
    std::vector<int> quant_inds(blockSize);
    std::vector<float> decompressedBlock(blockSize);

    int radius = 512;

    for (size_t bid : blockIds) {
        size_t local_id = bid;

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

        // 解码 bit-packed 量化索引
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

        // 重建 metadata（error_bound 存的是 double，跟 SZ3 内部格式一致）
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

        // SZ3 解压配置
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
// 辅助函数1（float版）：获取邻居块指针
// 说明：不做26邻居预扩展解压，blockMap里只有索引命中的核心块。
// 邻居块没被解压时返回nullptr，对应边界区域会被跳过（块边界可能出现缝隙）。
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
// 辅助函数2（float版）：处理内部cube
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

// ============================================================================
// 辅助函数3（float版）：处理X+边界cube
// ============================================================================
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

// ============================================================================
// 辅助函数4（float版）：处理Y+边界cube
// ============================================================================
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

// ============================================================================
// 辅助函数5（float版）：处理Z+边界cube
// ============================================================================
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


// ============================================================================
// 辅助函数6（float版）：处理XY边棱cube
// ============================================================================
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

// ============================================================================
// 辅助函数7（float版）：处理XZ边棱cube
// ============================================================================
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

// ============================================================================
// 辅助函数8（float版）：处理YZ边棱cube
// ============================================================================
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

// ============================================================================
// 辅助函数9（float版）：处理XYZ角点cube
// ============================================================================
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
// 主函数：RunMarchingCubesOnDecompressedBlocks （float 版本）
// 分区域处理：内部cube（快速路径）+ 面/棱/角边界（需要邻居块指针）
// 无26邻居预扩展解压，blockMap里只有索引命中的核心块。
// ============================================================================
util::TriangleMesh<float> RunMarchingCubesOnDecompressedBlocks(
    const std::vector<std::vector<float>>& decompressedBlocks,
    const std::vector<size_t>& coreGlobalSmallBlockIds,
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
    blockMap.reserve(coreGlobalSmallBlockIds.size());
    for (size_t i = 0; i < coreGlobalSmallBlockIds.size(); i++) {
        blockMap[coreGlobalSmallBlockIds[i]] = decompressedBlocks[i].data();
    }

    std::vector<size_t> sortedBlockIds = coreGlobalSmallBlockIds;
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
    const std::vector<size_t>& coreGlobalSmallBlockIds,
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
    std::vector<size_t> stepDataShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0, smallBlockSize = 1, totalBlocksNumber = 1;
    std::vector<float> queryRange;
    float relative_error_bound = 1E-3;
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
                    MPI_Finalize();
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

    // === 单个"逻辑step"，无大块层，直接用smallBlockShape计算分块数量 ===
    size_t smallBlocksPerStep = 1;
    for (size_t d = 0; d < nDim; d++) {
        smallBlocksPerStep *= stepDataShape[d] / smallBlockShape[d];
        smallBlockSize     *= smallBlockShape[d];
    }
    size_t nSteps = endStepNum - beginStepNum + 1;
    totalBlocksNumber = nSteps * smallBlocksPerStep;

    if (queryRange.size() != 2) {
        std::cerr << "Error: Invalid query range. Use --query_range <low> <high>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    // === 索引目录：对接 doc16（HDF5 uniform索引程序）生成的路径 ===
    indexDir = "/expanse/lustre/scratch/sdi/temp_project/nyx_index_compress/" + inputFileBaseName + "_" + safeVarName + "_"
             + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
             + std::to_string(extraValue) + "_hdf5_uniform_index/";

    // === 原始压缩数据目录：对接 doc15/doc11（HDF5 fixlength压缩程序）生成的路径 ===
    std::string mySubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + safeVarName + "_"
             + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
             + std::to_string(extraValue) + "_hdf5_fixlength_originalDataCompression/";

    auto afterProcessTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> preProcessTime = afterProcessTime - totalStart;
    std::cerr << "[Rank " << mpi_rank << "][Time1] : preProcessTime: " << preProcessTime.count() << " seconds" << std::endl;

    // === 加载 big_block_minmax（step级粗筛）===
    std::vector<ScidxInterval<float>> allBigBlockIndices = loadBigBlockIndexFile(indexDir + "big_block_minmax");

    std::cout << "[Debug] uniform big_block_minmax entries: " << allBigBlockIndices.size() << "\n";

    float globalMin = std::numeric_limits<float>::max();
    float globalMax = std::numeric_limits<float>::lowest();

    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }

    std::cout << "Global Min: " << globalMin << std::endl;
    std::cout << "Global Max: " << globalMax << std::endl;

    float error_bound = relative_error_bound * (globalMax - globalMin);
    std::cout << "Computed error_bound: " << error_bound << std::endl;

    float isovalue = queryRange[0];

    // === 筛出可能命中的step ===
    std::vector<size_t> selectedSteps;
    for (size_t i = 0; i < allBigBlockIndices.size(); i++) {
        if (isovalue <= allBigBlockIndices[i].high && isovalue >= allBigBlockIndices[i].low) {
            selectedSteps.push_back(i);
        }
    }
    std::cout << "Selected Steps: " << selectedSteps.size() << std::endl;

    auto afterQueryBigBlock = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readAndQueryBigBlock = afterQueryBigBlock - afterProcessTime;
    std::cerr << "[Rank " << mpi_rank << "][Time2] : read and query bigBlocks: " << readAndQueryBigBlock.count() << " seconds" << std::endl;

    // === 精细查询 AVL 树，得到核心命中块 ===
    std::unordered_set<size_t> hitBlocksSet;
    for (size_t stepRelIdx : selectedSteps) {
        size_t actualStepNum = beginStepNum + stepRelIdx;
        size_t stepOffset    = stepRelIdx * smallBlocksPerStep;

        std::vector<size_t> rawIds =
            queryIndexRawIds(actualStepNum, indexDir, queryRange, error_bound);

        auto time_before_merge = std::chrono::high_resolution_clock::now();

        for (size_t rawId : rawIds)
            if (rawId < smallBlocksPerStep)
                hitBlocksSet.insert(stepOffset + rawId);

        auto time_after_merge = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> mergeTime = time_after_merge - time_before_merge;
        std::cerr << "[Time_merge] insert + dedup: "
                << mergeTime.count() << " seconds" << std::endl;
    }

    auto time_before_sort = std::chrono::high_resolution_clock::now();

    std::vector<size_t> coreGlobalSmallBlockIds(
        hitBlocksSet.begin(), hitBlocksSet.end());
    std::sort(coreGlobalSmallBlockIds.begin(), coreGlobalSmallBlockIds.end());

    auto time_after_sort = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sortTime = time_after_sort - time_before_sort;
    std::cerr << "[Time_sort] set->vector + sort: "
            << sortTime.count() << " seconds" << std::endl;

    std::cerr << "[CHECK] hit blocks (deduped): "
            << coreGlobalSmallBlockIds.size() << std::endl;

    for (size_t bid : coreGlobalSmallBlockIds) {
        if (bid >= totalBlocksNumber) {
            std::cerr << "[BUG] bid " << bid << " >= totalBlocksNumber = " << totalBlocksNumber << std::endl;
            MPI_Finalize();
            exit(1);
        }
    }

    // ============================================================
    // 说明：不做26邻居扩展。直接用 coreGlobalSmallBlockIds（索引命中的核心块）
    // 去解压 fixed_*.bin 格式的原始压缩数据。
    // 代价：块与块交界处若两侧不是同时命中，会缺少邻居数据而被跳过，
    // mesh可能在块边界处出现缝隙。
    // ============================================================
    std::vector<std::vector<float>> decompressedTargetOriginalBlockdata =
        batchDecompressBlocksSelective(
            mySubDir,
            coreGlobalSmallBlockIds,
            smallBlockSize,
            totalBlocksNumber,
            error_bound
        );

    auto finishDecompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompressTime = finishDecompress - time_after_sort;
    std::cerr << "[Time_decompress] decompress core blocks: " << decompressTime.count() << " seconds" << std::endl;

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

        std::vector<size_t> stepCoreBlockIds;
        for (size_t id : coreGlobalSmallBlockIds) {
            if (id >= stepOffset && id < stepOffset + smallBlocksPerStep) {
                stepCoreBlockIds.push_back(id);
            }
        }

        std::cout << "[Info] Step " << currentStepNum << " core blocks: "
                << stepCoreBlockIds.size() << std::endl;

        std::vector<std::vector<float>> stepBlocks;
        stepBlocks.reserve(stepCoreBlockIds.size());

        std::unordered_map<size_t, size_t> blockIdToIndex;
        for (size_t i = 0; i < coreGlobalSmallBlockIds.size(); ++i) {
            blockIdToIndex[coreGlobalSmallBlockIds[i]] = i;
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

        std::string outFile = "/expanse/lustre/scratch/sdi/temp_project/iso_results/isosurface_"
            + safeVarName + "_step_" + std::to_string(currentStepNum)
            + "_iso_" + std::to_string(queryRange[0]) + "_hdf5_uniformOnly.vtk";

        float iso_value = queryRange[0];

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

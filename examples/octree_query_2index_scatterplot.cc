#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <array>
#include <stdexcept>

#include <scidx_octree_interval.h>   // OctreeNode, queryOctree
#include <scidx_octree.h>
#include <scidx_octree_hilbert2.h>   // decompressOctreeUniformHilbertNew / decompressOctreeStaggerHilbertNew
#include <SZ3/api/sz.hpp>
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>

// ============================================================
// Load big_block_minmax file
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
// Octree UNIFORM 树：解压 + 查询
// 同时输出 listLow / listHigh，供同一变量的 STAGGER 树解压使用
//
// 计时说明：
//   解压函数内部已分别打印真实的读取/解压时间。
//   这里只在外层记录解压总耗时（读取+解压）和查询耗时，不做比例拆分。
// ============================================================
std::vector<size_t> queryOctreeUniform(
    size_t actualStepNum,
    const std::string& indexDir,
    const std::vector<double>& queryRange,
    double error_bound,
    const std::vector<size_t>& uniformBlockCount,
    std::vector<double>& listLow,
    std::vector<double>& listHigh,
    double& total_index_load,
    double& total_index_query)
{
    std::string treeID = indexDir + std::to_string(actualStepNum) + "-0";

    auto t_load_start = std::chrono::high_resolution_clock::now();
    std::vector<OctreeNode<double>> octree =
        decompressOctreeUniformHilbertNew<double>(treeID, error_bound, uniformBlockCount,
                                                  &listLow, &listHigh);
    auto t_load_end = std::chrono::high_resolution_clock::now();

    auto t_q_start = std::chrono::high_resolution_clock::now();
    std::vector<size_t> ids = queryOctree(
        octree, queryRange[0], queryRange[1], uniformBlockCount, error_bound);
    auto t_q_end = std::chrono::high_resolution_clock::now();

    total_index_load  += std::chrono::duration<double>(t_load_end - t_load_start).count();
    total_index_query += std::chrono::duration<double>(t_q_end    - t_q_start).count();

    return ids;
}

// ============================================================
// Octree STAGGER 树：解压 + 查询
// 必须传入同一变量 UNIFORM 树产出的 listLow / listHigh
// ============================================================
std::vector<size_t> queryOctreeStagger(
    size_t actualStepNum,
    const std::string& staggerIndexDir,
    const std::vector<double>& queryRange,
    double error_bound,
    const std::vector<size_t>& staggerBlockCount,
    const std::vector<size_t>& uniformBlockCount,
    const std::vector<double>& listLow,
    const std::vector<double>& listHigh,
    double& total_index_load,
    double& total_index_query)
{
    std::string treeID = staggerIndexDir + std::to_string(actualStepNum) + "-0";

    auto t_load_start = std::chrono::high_resolution_clock::now();
    std::vector<OctreeNode<double>> octree =
        decompressOctreeStaggerHilbertNew<double>(
            treeID, error_bound, staggerBlockCount, uniformBlockCount, listLow, listHigh);
    auto t_load_end = std::chrono::high_resolution_clock::now();

    auto t_q_start = std::chrono::high_resolution_clock::now();
    std::vector<size_t> ids = queryOctree(
        octree, queryRange[0], queryRange[1], staggerBlockCount, error_bound);
    auto t_q_end = std::chrono::high_resolution_clock::now();

    total_index_load  += std::chrono::duration<double>(t_load_end - t_load_start).count();
    total_index_query += std::chrono::duration<double>(t_q_end    - t_q_start).count();

    return ids;
}

// ============================================================
// 错位块 flat ID -> 覆盖的均匀块 flat ID 集合（Octree 版）
// 注意：与 AVL 版不同
//   k=0  : [0, half + small)
//   k>=1 : [half + k*small, half + (k+1)*small) ∩ [0, shape)
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
        if (start >= shape[d] || end_excl == 0 || start >= end_excl) return {};
        size_t u_start = start / smallBlockShape[d];
        size_t u_end   = (end_excl - 1) / smallBlockShape[d];
        u_end = std::min(u_end, uniformBlockCount[d] - 1);
        uRange[d] = {u_start, u_end};
    }

    std::vector<size_t> result;
    for (size_t ux = uRange[0].first; ux <= uRange[0].second; ux++)
        for (size_t uy = uRange[1].first; uy <= uRange[1].second; uy++)
            for (size_t uz = uRange[2].first; uz <= uRange[2].second; uz++) {
                size_t flatId = ux
                              + uy * uniformBlockCount[0]
                              + uz * uniformBlockCount[0] * uniformBlockCount[1];
                result.push_back(flatId);
            }
    return result;
}

// ============================================================
// Block config
// ============================================================
struct BlockConfig {
    size_t blockSize;
    size_t unpredSizeBits;
    size_t dataSizeBytes;
    size_t signBytesPerBlock;

    BlockConfig(size_t bs) : blockSize(bs) {
        if      (bs == 64)         { unpredSizeBits = 6;  dataSizeBytes = 1; signBytesPerBlock = 8; }
        else if (bs == 4096)       { unpredSizeBits = 12; dataSizeBytes = 2; signBytesPerBlock = 512; }
        else if (bs == 262144)     { unpredSizeBits = 18; dataSizeBytes = 3; signBytesPerBlock = 32768; }
        else if (bs == 16777216)   { unpredSizeBits = 24; dataSizeBytes = 4; signBytesPerBlock = 2097152; }
        else if (bs == 1073741824) { unpredSizeBits = 30; dataSizeBytes = 4; signBytesPerBlock = 134217728; }
        else throw std::runtime_error("Unsupported block size");
    }
};

void unpackBitsInline(const std::vector<uint8_t>& input, std::vector<size_t>& output,
                      size_t bitsPerValue, size_t numValues) {
    output.clear(); output.reserve(numValues);
    if (input.empty() || numValues == 0) return;
    size_t byteIndex = 0, bitIndex = 0;
    uint64_t mask = (1ULL << bitsPerValue) - 1;
    for (size_t i = 0; i < numValues; i++) {
        uint64_t value = 0; size_t bitsRead = 0;
        while (bitsRead < bitsPerValue) {
            if (byteIndex >= input.size()) throw std::runtime_error("Insufficient data");
            size_t bitsToRead = std::min(bitsPerValue - bitsRead, 8 - bitIndex);
            uint8_t byteMask = ((1 << bitsToRead) - 1) << bitIndex;
            uint8_t bits = (input[byteIndex] & byteMask) >> bitIndex;
            value |= (static_cast<uint64_t>(bits) << bitsRead);
            bitsRead += bitsToRead; bitIndex += bitsToRead;
            if (bitIndex >= 8) { bitIndex = 0; byteIndex++; }
        }
        output.push_back(static_cast<size_t>(value & mask));
    }
}

uint64_t readValueInline(const uint8_t* data, size_t numBytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < numBytes; i++)
        value |= (static_cast<uint64_t>(data[i]) << (i * 8));
    return value;
}

// ============================================================
// Universal batch decompress blocks（选择性解压）
// ============================================================
std::vector<std::vector<double>> universalBatchDecompressBlocks(
    const std::string& subDir,
    const std::vector<size_t>& blockIds,
    size_t blockSize,
    size_t totalBlocks,
    double error_bound,
    double& total_data_read,
    double& total_data_decomp)
{
    BlockConfig config(blockSize);
    std::vector<std::vector<double>> result;

    std::string unpredDataFileName = subDir + "universal_unpredData.bin";
    std::string compressedFileName = subDir + "universal_compressed_data.bin";
    std::string signFileName       = subDir + "universal_sign_data.bin";

    auto t_read_start = std::chrono::high_resolution_clock::now();

    std::ifstream unpredStream(unpredDataFileName, std::ios::binary);
    std::ifstream compStream(compressedFileName,   std::ios::binary);
    std::ifstream signStream(signFileName,         std::ios::binary);
    if (!unpredStream || !compStream || !signStream)
        throw std::runtime_error("Failed to open compressed data files: " + subDir);

    unpredStream.seekg(0, std::ios::end); size_t unpredFileSize = unpredStream.tellg(); unpredStream.seekg(0);
    compStream.seekg(0, std::ios::end);   size_t compFileSize   = compStream.tellg();   compStream.seekg(0);
    signStream.seekg(0, std::ios::end);   size_t signFileSize   = signStream.tellg();   signStream.seekg(0);

    std::vector<uint8_t> allUnpredData(unpredFileSize);
    std::vector<uint8_t> allCompData(compFileSize);
    std::vector<uint8_t> allSignData(signFileSize);

    unpredStream.read(reinterpret_cast<char*>(allUnpredData.data()), unpredFileSize);
    compStream.read(reinterpret_cast<char*>(allCompData.data()),     compFileSize);
    signStream.read(reinterpret_cast<char*>(allSignData.data()),     signFileSize);

    auto t_read_end = std::chrono::high_resolution_clock::now();
    total_data_read += std::chrono::duration<double>(t_read_end - t_read_start).count();

    size_t unpredSizeBytes = (totalBlocks * config.unpredSizeBits + 7) / 8;
    std::vector<uint8_t> compressedUnpredSizes(allUnpredData.begin(),
                                               allUnpredData.begin() + unpredSizeBytes);
    std::vector<size_t> unpredSizes;
    unpackBitsInline(compressedUnpredSizes, unpredSizes, config.unpredSizeBits, totalBlocks);

    std::vector<uint8_t> bitCounts(totalBlocks);
    std::memcpy(bitCounts.data(), allCompData.data(), totalBlocks);

    std::vector<size_t> compSizes(totalBlocks);
    for (size_t i = 0; i < totalBlocks; i++) {
        size_t offset = totalBlocks + i * config.dataSizeBytes;
        compSizes[i] = static_cast<size_t>(readValueInline(allCompData.data() + offset,
                                                           config.dataSizeBytes));
    }

    std::vector<size_t> unpredOffsets(totalBlocks, 0);
    std::vector<size_t> compOffsets(totalBlocks, 0);
    for (size_t i = 1; i < totalBlocks; ++i) {
        unpredOffsets[i] = unpredOffsets[i-1] + unpredSizes[i-1];
        compOffsets[i]   = compOffsets[i-1]   + compSizes[i-1];
    }

    auto t_decomp_start = std::chrono::high_resolution_clock::now();

    std::vector<double>  unpredData;
    std::vector<uint8_t> compData;
    std::vector<uint8_t> signBits(config.signBytesPerBlock);

    for (size_t bid : blockIds) {
        size_t unpredSize   = unpredSizes[bid];
        size_t unpredOffset = unpredSizeBytes + sizeof(double) * unpredOffsets[bid];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredData.data() + unpredOffset,
                    unpredSize * sizeof(double));

        uint8_t bitCount   = bitCounts[bid];
        size_t  dataSize   = compSizes[bid];
        size_t  compOffset = totalBlocks + totalBlocks * config.dataSizeBytes + compOffsets[bid];
        compData.resize(dataSize);
        std::memcpy(compData.data(), allCompData.data() + compOffset, dataSize);

        size_t signOffset = config.signBytesPerBlock * bid;
        std::memcpy(signBits.data(), allSignData.data() + signOffset, config.signBytesPerBlock);

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
                bitsRead += available; bitPos += available;
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
        size_t correctUnpredSize = unpredSize;
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize) + sizeof(size_t));
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data()),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data() + unpredSize));

        SZ3::Config conf(blockSize);
        conf.cmprAlgo       = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo        = true;
        conf.regression     = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound  = error_bound;
        conf.quantbinCnt    = 1024;

        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose  = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);

        std::vector<double> decompressedBlock(blockSize);
        const SZ3::uchar* metaPtr = metadataBuffer.data();
        size_t metaSize = metadataBuffer.size();
        decompose.load(metaPtr, metaSize);
        decompose.decompress(conf, quant_inds, decompressedBlock.data());
        result.push_back(std::move(decompressedBlock));
    }

    auto t_decomp_end = std::chrono::high_resolution_clock::now();
    total_data_decomp += std::chrono::duration<double>(t_decomp_end - t_decomp_start).count();

    return result;
}

// ============================================================
// CSP computation（与 AVL 索引版完全相同）
// For each cell: map center (f_avg, g_avg) to bin,
// accumulate 1/|∇f × ∇g| as density weight.
// ============================================================
std::vector<std::vector<double>> computeCSP(
    const std::vector<std::vector<double>>& blocksF,
    const std::vector<std::vector<double>>& blocksG,
    const std::vector<size_t>& expandedBlockIds,
    const std::vector<size_t>& coreBlockIds,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& dataShape,
    size_t totalBlocksNumber,
    double f_min, double f_max,
    double g_min, double g_max,
    size_t csp_res_f,
    size_t csp_res_g)
{
    size_t Bx = blockShape[0], By = blockShape[1], Bz = blockShape[2];

    std::vector<size_t> blockCountOnEachDim(3);
    for (size_t i = 0; i < 3; i++)
        blockCountOnEachDim[i] = (dataShape[i]+blockShape[i]-1)/blockShape[i];

    std::vector<const std::vector<double>*> blockPtrF(totalBlocksNumber, nullptr);
    std::vector<const std::vector<double>*> blockPtrG(totalBlocksNumber, nullptr);
    for (size_t i = 0; i < expandedBlockIds.size(); i++) {
        blockPtrF[expandedBlockIds[i]] = &blocksF[i];
        blockPtrG[expandedBlockIds[i]] = &blocksG[i];
    }

    std::vector<std::vector<double>> cspImage(csp_res_f,
                                              std::vector<double>(csp_res_g, 0.0));
    double df = (f_max - f_min) / csp_res_f;
    double dg = (g_max - g_min) / csp_res_g;

    auto getVal = [&](const std::vector<const std::vector<double>*>& ptrs,
                      size_t blockId, size_t lx, size_t ly, size_t lz) -> double {
        if (blockId == static_cast<size_t>(-1) || blockId >= totalBlocksNumber) return 0.0;
        const auto* ptr = ptrs[blockId];
        if (ptr == nullptr) return 0.0;
        return (*ptr)[lx + ly*Bx + lz*Bx*By];
    };

    auto getNeighborId = [&](size_t blockId, int dx, int dy, int dz) -> size_t {
        size_t bz  = blockId % blockCountOnEachDim[2];
        size_t rem = blockId / blockCountOnEachDim[2];
        size_t by  = rem % blockCountOnEachDim[1];
        size_t bx  = rem / blockCountOnEachDim[1];
        int nx=(int)bx+dx, ny=(int)by+dy, nz=(int)bz+dz;
        if (nx<0||nx>=(int)blockCountOnEachDim[0]||
            ny<0||ny>=(int)blockCountOnEachDim[1]||
            nz<0||nz>=(int)blockCountOnEachDim[2])
            return static_cast<size_t>(-1);
        return (size_t)nx*blockCountOnEachDim[1]*blockCountOnEachDim[2]
             + (size_t)ny*blockCountOnEachDim[2]
             + (size_t)nz;
    };

    for (size_t blockId : coreBlockIds) {
        if (blockId >= totalBlocksNumber || blockPtrF[blockId] == nullptr) continue;

        size_t block_z  = blockId % blockCountOnEachDim[2];
        size_t rem_     = blockId / blockCountOnEachDim[2];
        size_t block_y  = rem_ % blockCountOnEachDim[1];
        size_t block_x  = rem_ / blockCountOnEachDim[1];
        size_t gx_start = block_x * Bx;
        size_t gy_start = block_y * By;
        size_t gz_start = block_z * Bz;

        for (size_t lz = 0; lz < Bz; ++lz)
        for (size_t ly = 0; ly < By; ++ly)
        for (size_t lx = 0; lx < Bx; ++lx) {
            size_t gx = gx_start+lx, gy = gy_start+ly, gz = gz_start+lz;
            if (gx+1>=dataShape[0]||gy+1>=dataShape[1]||gz+1>=dataShape[2]) continue;

            size_t bxp   = (lx+1<Bx)?blockId:getNeighborId(blockId,1,0,0);
            size_t byp   = (ly+1<By)?blockId:getNeighborId(blockId,0,1,0);
            size_t bzp   = (lz+1<Bz)?blockId:getNeighborId(blockId,0,0,1);
            size_t bxyp  = getNeighborId(blockId,(lx+1<Bx)?0:1,(ly+1<By)?0:1,0);
            size_t bxzp  = getNeighborId(blockId,(lx+1<Bx)?0:1,0,(lz+1<Bz)?0:1);
            size_t byzp  = getNeighborId(blockId,0,(ly+1<By)?0:1,(lz+1<Bz)?0:1);
            size_t bxyzp = getNeighborId(blockId,(lx+1<Bx)?0:1,(ly+1<By)?0:1,(lz+1<Bz)?0:1);
            size_t lxp=(lx+1<Bx)?lx+1:0;
            size_t lyp=(ly+1<By)?ly+1:0;
            size_t lzp=(lz+1<Bz)?lz+1:0;

            double fv[8] = {
                getVal(blockPtrF,blockId,lx, ly, lz),
                getVal(blockPtrF,bxp,   lxp,ly, lz),
                getVal(blockPtrF,bxyp,  lxp,lyp,lz),
                getVal(blockPtrF,byp,   lx, lyp,lz),
                getVal(blockPtrF,bzp,   lx, ly, lzp),
                getVal(blockPtrF,bxzp,  lxp,ly, lzp),
                getVal(blockPtrF,bxyzp, lxp,lyp,lzp),
                getVal(blockPtrF,byzp,  lx, lyp,lzp)
            };
            double gv[8] = {
                getVal(blockPtrG,blockId,lx, ly, lz),
                getVal(blockPtrG,bxp,   lxp,ly, lz),
                getVal(blockPtrG,bxyp,  lxp,lyp,lz),
                getVal(blockPtrG,byp,   lx, lyp,lz),
                getVal(blockPtrG,bzp,   lx, ly, lzp),
                getVal(blockPtrG,bxzp,  lxp,ly, lzp),
                getVal(blockPtrG,bxyzp, lxp,lyp,lzp),
                getVal(blockPtrG,byzp,  lx, lyp,lzp)
            };

            double cell_fmin=*std::min_element(fv,fv+8);
            double cell_fmax=*std::max_element(fv,fv+8);
            double cell_gmin=*std::min_element(gv,gv+8);
            double cell_gmax=*std::max_element(gv,gv+8);
            if (cell_fmax<f_min||cell_fmin>f_max) continue;
            if (cell_gmax<g_min||cell_gmin>g_max) continue;

            double grad_fx=0.25*((fv[1]-fv[0])+(fv[2]-fv[3])+(fv[5]-fv[4])+(fv[6]-fv[7]));
            double grad_fy=0.25*((fv[3]-fv[0])+(fv[2]-fv[1])+(fv[7]-fv[4])+(fv[6]-fv[5]));
            double grad_fz=0.25*((fv[4]-fv[0])+(fv[5]-fv[1])+(fv[6]-fv[2])+(fv[7]-fv[3]));
            double grad_gx=0.25*((gv[1]-gv[0])+(gv[2]-gv[3])+(gv[5]-gv[4])+(gv[6]-gv[7]));
            double grad_gy=0.25*((gv[3]-gv[0])+(gv[2]-gv[1])+(gv[7]-gv[4])+(gv[6]-gv[5]));
            double grad_gz=0.25*((gv[4]-gv[0])+(gv[5]-gv[1])+(gv[6]-gv[2])+(gv[7]-gv[3]));

            double cross_x=grad_fy*grad_gz-grad_fz*grad_gy;
            double cross_y=grad_fz*grad_gx-grad_fx*grad_gz;
            double cross_z=grad_fx*grad_gy-grad_fy*grad_gx;
            double cross_mag=std::sqrt(cross_x*cross_x+cross_y*cross_y+cross_z*cross_z);
            if (cross_mag<1e-12) continue;

            double fc=0.0, gc=0.0;
            for (int i=0;i<8;i++){fc+=fv[i];gc+=gv[i];}
            fc/=8.0; gc/=8.0;

            double fc_c=std::max(f_min,std::min(f_max,fc));
            double gc_c=std::max(g_min,std::min(g_max,gc));
            size_t bin_f=static_cast<size_t>((fc_c-f_min)/df);
            size_t bin_g=static_cast<size_t>((gc_c-g_min)/dg);
            if (bin_f>=csp_res_f) bin_f=csp_res_f-1;
            if (bin_g>=csp_res_g) bin_g=csp_res_g-1;

            cspImage[bin_f][bin_g] += 1.0/cross_mag;
        }
    }
    return cspImage;
}

// ============================================================
// Save CSP image
// Format: [res_f: size_t][res_g: size_t][density: double*]
// ============================================================
void saveCSPImage(const std::vector<std::vector<double>>& cspImage,
                  const std::string& outFile)
{
    std::ofstream ofs(outFile, std::ios::binary);
    if (!ofs) { std::cerr << "Error: cannot open " << outFile << std::endl; return; }
    size_t res_f = cspImage.size();
    size_t res_g = res_f > 0 ? cspImage[0].size() : 0;
    ofs.write(reinterpret_cast<const char*>(&res_f), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&res_g), sizeof(size_t));
    for (const auto& row : cspImage)
        ofs.write(reinterpret_cast<const char*>(row.data()), row.size()*sizeof(double));
    ofs.close();
    std::cout << "[CSP] Saved CSP image (" << res_f << " x " << res_g
              << ") to " << outFile << std::endl;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[])
{
    auto totalStart = std::chrono::high_resolution_clock::now();
    MPI_Init(&argc, &argv);

    std::string inputFileName, varNameF, varNameG;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0;
    double relative_error_bound = 1e-3;
    size_t extraValue = 0;
    std::vector<double> queryRangeF, queryRangeG;
    size_t csp_res = 256;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg=="--input_file"        &&i+1<argc) inputFileName=argv[++i];
        else if (arg=="--variable_name_f"   &&i+1<argc) varNameF=argv[++i];
        else if (arg=="--variable_name_g"   &&i+1<argc) varNameG=argv[++i];
        else if (arg=="--dimensions"        &&i+1<argc) nDim=std::stoul(argv[++i]);
        else if (arg=="--begin_step"        &&i+1<argc) beginStepNum=std::stoul(argv[++i]);
        else if (arg=="--end_step"          &&i+1<argc) endStepNum=std::stoul(argv[++i]);
        else if (arg=="--stepData_shape"    &&i+nDim<argc)
            for (size_t j=0;j<nDim;j++) stepDataShape.push_back(std::stoul(argv[++i]));
        else if (arg=="--small_block_shape" &&i+nDim<argc)
            for (size_t j=0;j<nDim;j++) smallBlockShape.push_back(std::stoul(argv[++i]));
        else if (arg=="--query_range_f"     &&i+2<argc) {
            queryRangeF.push_back(std::stod(argv[++i]));
            queryRangeF.push_back(std::stod(argv[++i]));
        }
        else if (arg=="--query_range_g"     &&i+2<argc) {
            queryRangeG.push_back(std::stod(argv[++i]));
            queryRangeG.push_back(std::stod(argv[++i]));
        }
        else if (arg=="--relative_error"    &&i+2<argc) {
            relative_error_bound=std::stod(argv[++i]);
            extraValue=std::stoi(argv[++i]);
        }
        else if (arg=="--csp_res"           &&i+1<argc) csp_res=std::stoul(argv[++i]);
    }

    if (queryRangeF.size()!=2||queryRangeG.size()!=2) {
        std::cerr<<"Error: need --query_range_f <lo> <hi> and --query_range_g <lo> <hi>"<<std::endl;
        MPI_Finalize(); return 1;
    }

    size_t smallBlocksPerStep=1, smallBlockSize=1;
    for (size_t d=0;d<nDim;d++) {
        smallBlocksPerStep *= stepDataShape[d]/smallBlockShape[d];
        smallBlockSize     *= smallBlockShape[d];
    }
    size_t nSteps=endStepNum-beginStepNum+1;
    size_t totalBlocksNumber=nSteps*smallBlocksPerStep;

    // Octree 版：错位分块数 = 均匀分块数（与 AVL 版的 +1 不同）
    std::vector<size_t> halfBlockShape(nDim), staggerBlockCount(nDim), uniformBlockCount(nDim);
    for (size_t d=0;d<nDim;d++) {
        halfBlockShape[d]    = smallBlockShape[d]/2;
        uniformBlockCount[d] = stepDataShape[d]/smallBlockShape[d];
        staggerBlockCount[d] = uniformBlockCount[d];
    }

    std::string safeF=varNameF; std::replace(safeF.begin(),safeF.end(),'/','_');
    std::string safeG=varNameG; std::replace(safeG.begin(),safeG.end(),'/','_');
    std::string baseName=std::filesystem::path(inputFileName).filename().string();

    // 索引目录：与 Octree Hilbert2 压缩端保持一致（_new 后缀）
    auto mkIndexDir = [&](const std::string& safeVar, const std::string& suffix) {
        return "/expanse/lustre/scratch/sdi/temp_project/octree_index_compressed/"
              +baseName+"_"+safeVar+"_"
              +std::to_string(beginStepNum)+"_"+std::to_string(endStepNum)+"_"
              +std::to_string(extraValue)+"_newour_octree_hilbert_"+suffix+"_index_new/";
    };
    auto mkDataDir = [&](const std::string& safeVar) {
        return "/home/nyan/scidx/scidx/"+baseName+"_"+safeVar+"_"
              +std::to_string(beginStepNum)+"_"+std::to_string(endStepNum)+"_"
              +std::to_string(extraValue)+"_blockZFP_originalDataCompression/";
    };

    std::string indexDirF1=mkIndexDir(safeF,"stagger1");
    std::string indexDirF2=mkIndexDir(safeF,"stagger2");
    std::string indexDirG1=mkIndexDir(safeG,"stagger1");
    std::string indexDirG2=mkIndexDir(safeG,"stagger2");
    std::string dataDirF=mkDataDir(safeF);
    std::string dataDirG=mkDataDir(safeG);

    auto afterParseTime=std::chrono::high_resolution_clock::now();
    std::cerr<<"[Time1] Preprocessing: "
             <<std::chrono::duration<double>(afterParseTime-totalStart).count()<<" seconds"<<std::endl;

    // ---- Stage 2: Big block prefilter ----
    auto t_prefilter_start=std::chrono::high_resolution_clock::now();

    auto bbF1=loadBigBlockIndexFile(indexDirF1+"big_block_minmax");
    auto bbF2=loadBigBlockIndexFile(indexDirF2+"big_block_minmax");
    auto bbG1=loadBigBlockIndexFile(indexDirG1+"big_block_minmax");
    auto bbG2=loadBigBlockIndexFile(indexDirG2+"big_block_minmax");

    // 与 Octree 等值面版一致：全局值域取自 uniform 的 big_block_minmax
    double F_globalMin=std::numeric_limits<double>::max();
    double F_globalMax=std::numeric_limits<double>::lowest();
    for (const auto& iv:bbF1) { F_globalMin=std::min(F_globalMin,iv.low); F_globalMax=std::max(F_globalMax,iv.high); }
    double error_bound_F=relative_error_bound*(F_globalMax-F_globalMin);
    std::cout<<"error_bound_F = "<<error_bound_F<<std::endl;

    double G_globalMin=std::numeric_limits<double>::max();
    double G_globalMax=std::numeric_limits<double>::lowest();
    for (const auto& iv:bbG1) { G_globalMin=std::min(G_globalMin,iv.low); G_globalMax=std::max(G_globalMax,iv.high); }
    double error_bound_G=relative_error_bound*(G_globalMax-G_globalMin);
    std::cout<<"error_bound_G = "<<error_bound_G<<std::endl;

    std::unordered_set<size_t> candidateStepsF,candidateStepsG;
    for (size_t i=0;i<bbF1.size();i++) if (bbF1[i].high>=queryRangeF[0]&&bbF1[i].low<=queryRangeF[1]) candidateStepsF.insert(i);
    for (size_t i=0;i<bbF2.size();i++) if (bbF2[i].high>=queryRangeF[0]&&bbF2[i].low<=queryRangeF[1]) candidateStepsF.insert(i);
    for (size_t i=0;i<bbG1.size();i++) if (bbG1[i].high>=queryRangeG[0]&&bbG1[i].low<=queryRangeG[1]) candidateStepsG.insert(i);
    for (size_t i=0;i<bbG2.size();i++) if (bbG2[i].high>=queryRangeG[0]&&bbG2[i].low<=queryRangeG[1]) candidateStepsG.insert(i);

    std::vector<size_t> candidateSteps;
    for (size_t s:candidateStepsF) if (candidateStepsG.count(s)) candidateSteps.push_back(s);
    std::sort(candidateSteps.begin(),candidateSteps.end());

    auto t_prefilter_end=std::chrono::high_resolution_clock::now();
    std::cerr<<"[Time2] Big block prefilter: "
             <<std::chrono::duration<double>(t_prefilter_end-t_prefilter_start).count()
             <<" seconds, candidate steps: "<<candidateSteps.size()<<std::endl;

    // ---- Stage 3: Dual Octree index query ----
    // total_index_load : 4 棵树解压函数的外层实测总耗时（读取+解压）
    //                    读取/解压各自的真实值见日志中库函数内部的打印
    double total_index_load=0, total_index_query=0;
    double total_merge_sort=0;
    double total_expansion=0;
    std::unordered_set<size_t> hitUniformBlocksSet;

    std::vector<size_t> blockCountOnEachDim(3);
    for (size_t i=0;i<3;i++)
        blockCountOnEachDim[i]=(stepDataShape[i]+smallBlockShape[i]-1)/smallBlockShape[i];

    for (size_t stepRelIdx:candidateSteps) {
        size_t actualStepNum=beginStepNum+stepRelIdx;
        size_t stepOffset=stepRelIdx*smallBlocksPerStep;

        // F：先 uniform（产出 listLow/High），再 stagger（使用 F 自己的 list）
        std::vector<double> listLowF, listHighF;
        auto uniformF=queryOctreeUniform(actualStepNum,indexDirF1,queryRangeF,error_bound_F,
                                         uniformBlockCount,listLowF,listHighF,
                                         total_index_load,total_index_query);
        auto staggerF=queryOctreeStagger(actualStepNum,indexDirF2,queryRangeF,error_bound_F,
                                         staggerBlockCount,uniformBlockCount,listLowF,listHighF,
                                         total_index_load,total_index_query);

        // G：同上，使用 G 自己的 list
        std::vector<double> listLowG, listHighG;
        auto uniformG=queryOctreeUniform(actualStepNum,indexDirG1,queryRangeG,error_bound_G,
                                         uniformBlockCount,listLowG,listHighG,
                                         total_index_load,total_index_query);
        auto staggerG=queryOctreeStagger(actualStepNum,indexDirG2,queryRangeG,error_bound_G,
                                         staggerBlockCount,uniformBlockCount,listLowG,listHighG,
                                         total_index_load,total_index_query);

        // ---- ID Mapping + Union/Intersect ----
        auto t_merge_start=std::chrono::high_resolution_clock::now();

        std::unordered_set<size_t> setF,setG;

        for (size_t id:uniformF) if (id<smallBlocksPerStep) setF.insert(stepOffset+id);
        for (size_t sid:staggerF) {
            auto uids=convertStaggerToUniformIds(sid,stepDataShape,smallBlockShape,
                                                 halfBlockShape,staggerBlockCount,uniformBlockCount);
            for (size_t uid:uids) if (uid<smallBlocksPerStep) setF.insert(stepOffset+uid);
        }
        for (size_t id:uniformG) if (id<smallBlocksPerStep) setG.insert(stepOffset+id);
        for (size_t sid:staggerG) {
            auto uids=convertStaggerToUniformIds(sid,stepDataShape,smallBlockShape,
                                                 halfBlockShape,staggerBlockCount,uniformBlockCount);
            for (size_t uid:uids) if (uid<smallBlocksPerStep) setG.insert(stepOffset+uid);
        }

        // C+ = C+_f ∩ C+_g
        for (size_t id:setF) if (setG.count(id)) hitUniformBlocksSet.insert(id);

        auto t_merge_end=std::chrono::high_resolution_clock::now();
        total_merge_sort+=std::chrono::duration<double>(t_merge_end-t_merge_start).count();
    }

    auto t_sort_start=std::chrono::high_resolution_clock::now();

    std::vector<size_t> coreBlockIds(hitUniformBlocksSet.begin(),hitUniformBlocksSet.end());
    std::sort(coreBlockIds.begin(),coreBlockIds.end());

    auto t_sort_end=std::chrono::high_resolution_clock::now();
    total_merge_sort+=std::chrono::duration<double>(t_sort_end-t_sort_start).count();

    std::cerr<<"[Time_total] merge + sort: "<<total_merge_sort<<" seconds"<<std::endl;

    // ---- Stage 4: 26-neighbor expansion ----
    auto t_expand_start=std::chrono::high_resolution_clock::now();

    std::unordered_set<size_t> expandedSet;
    for (size_t cid:coreBlockIds) expandedSet.insert(cid);

    struct Dir{int dx,dy,dz;};
    std::vector<Dir> dirs;
    for (int dz=-1;dz<=1;dz++) for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++)
        if (dx||dy||dz) dirs.push_back({dx,dy,dz});

    for (size_t cid:coreBlockIds) {
        size_t bz=cid%blockCountOnEachDim[2];
        size_t rem=cid/blockCountOnEachDim[2];
        size_t by=rem%blockCountOnEachDim[1];
        size_t bx=rem/blockCountOnEachDim[1];
        for (const auto& d:dirs) {
            int nx=(int)bx+d.dx, ny=(int)by+d.dy, nz=(int)bz+d.dz;
            if (nx<0||nx>=(int)blockCountOnEachDim[0]||
                ny<0||ny>=(int)blockCountOnEachDim[1]||
                nz<0||nz>=(int)blockCountOnEachDim[2]) continue;
            expandedSet.insert((size_t)nx*blockCountOnEachDim[1]*blockCountOnEachDim[2]
                              +(size_t)ny*blockCountOnEachDim[2]+(size_t)nz);
        }
    }

    std::vector<size_t> expandedBlockIds(expandedSet.begin(),expandedSet.end());
    std::sort(expandedBlockIds.begin(),expandedBlockIds.end());

    auto t_expand_end=std::chrono::high_resolution_clock::now();
    total_expansion=std::chrono::duration<double>(t_expand_end-t_expand_start).count();

    std::cerr<<"[Time3] Index Load (read+decomp, 4 trees): "<<total_index_load<<" seconds"<<std::endl;
    std::cerr<<"[Time3] Index Query (4 trees):             "<<total_index_query<<" seconds"<<std::endl;
    std::cerr<<"Expansion time:   "<<total_expansion<<" seconds"<<std::endl;
    std::cerr<<"[CHECK] Core blocks: "<<coreBlockIds.size()<<", Expanded blocks: "<<expandedBlockIds.size()<<std::endl;

    // ---- Stage 5: Selective decompression ----
    double total_data_read=0,total_data_decomp=0;

    auto decompressedF=universalBatchDecompressBlocks(
        dataDirF,expandedBlockIds,smallBlockSize,totalBlocksNumber,error_bound_F,
        total_data_read,total_data_decomp);
    auto decompressedG=universalBatchDecompressBlocks(
        dataDirG,expandedBlockIds,smallBlockSize,totalBlocksNumber,error_bound_G,
        total_data_read,total_data_decomp);

    std::cerr<<"[Time5] Data Read (F+G):          "<<total_data_read<<" seconds"<<std::endl;
    std::cerr<<"[Time5] Data Decompression (F+G): "<<total_data_decomp<<" seconds"<<std::endl;

    // ---- Stage 6: CSP computation ----
    auto t_csp_start=std::chrono::high_resolution_clock::now();
    auto cspImage=computeCSP(
        decompressedF,decompressedG,
        expandedBlockIds,coreBlockIds,
        smallBlockShape,stepDataShape,
        totalBlocksNumber,
        queryRangeF[0],queryRangeF[1],
        queryRangeG[0],queryRangeG[1],
        csp_res,csp_res);
    auto t_csp_end=std::chrono::high_resolution_clock::now();
    std::cerr<<"[Time6] CSP Computation: "
             <<std::chrono::duration<double>(t_csp_end-t_csp_start).count()<<" seconds"<<std::endl;

    std::string outFile="csp_octree_"+safeF+"_"+safeG+".bin";
    saveCSPImage(cspImage,outFile);

    auto totalEnd=std::chrono::high_resolution_clock::now();
    double totalTime=std::chrono::duration<double>(totalEnd-totalStart).count();

    std::cerr<<"\n========== TIME SUMMARY (OCTREE) =========="<<std::endl;
    std::cerr<<"[Stage 1] Preprocessing:                            "
             <<std::chrono::duration<double>(afterParseTime-totalStart).count()<<" s"<<std::endl;
    std::cerr<<"[Stage 2] Big block prefilter:                      "
             <<std::chrono::duration<double>(t_prefilter_end-t_prefilter_start).count()<<" s"<<std::endl;
    std::cerr<<"[Stage 3] Index Load (read+decomp, 4 trees):        "<<total_index_load<<" s"<<std::endl;
    std::cerr<<"          (read / decomp split: sum the library's own printed lines above)"<<std::endl;
    std::cerr<<"[Stage 3] Index Query (4 trees):                    "<<total_index_query<<" s"<<std::endl;
    std::cerr<<"[Stage 4] merge + sort:                             "<<total_merge_sort<<" s"<<std::endl;
    std::cerr<<"[Stage 4] Expansion time:                           "<<total_expansion<<" s"<<std::endl;
    std::cerr<<"[Stage 5] Data Read (F+G):                          "<<total_data_read<<" s"<<std::endl;
    std::cerr<<"[Stage 5] Data Decompression (F+G):                 "<<total_data_decomp<<" s"<<std::endl;
    std::cerr<<"[Stage 6] CSP Computation:                          "
             <<std::chrono::duration<double>(t_csp_end-t_csp_start).count()<<" s"<<std::endl;
    std::cerr<<"[TOTAL]                                             "<<totalTime<<" s"<<std::endl;
    std::cerr<<"==========================================="<<std::endl;

    MPI_Finalize();
    return 0;
}
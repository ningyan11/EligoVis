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

#include <scidx_avl.h>
#include <SZ3/api/sz.hpp>
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>

// ============================================================
// Load big_block_minmax file (only for computing error_bound)
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
// Universal batch decompress ALL blocks
// Matches isosurface baseline: universalBatchDecompressBlocksAllRead
// Prints [Time6] read and [Time7] decompress
// ============================================================
std::vector<std::vector<double>> universalBatchDecompressBlocksAllRead(
    const std::string& subDir,
    const std::vector<size_t>& blockIds,
    size_t blockSize,
    size_t totalBlocks,
    double error_bound,
    double& total_data_read,
    double& total_data_decomp)
{
    std::cout << "universal all read: " << std::endl;

    BlockConfig config(blockSize);
    std::vector<std::vector<double>> result;

    std::string unpredDataFileName = subDir + "universal_unpredData.bin";
    std::string compressedFileName = subDir + "universal_compressed_data.bin";
    std::string signFileName       = subDir + "universal_sign_data.bin";

    // --- Data Read ---
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
    double readTime = std::chrono::duration<double>(t_read_end - t_read_start).count();
    total_data_read += readTime;
    std::cout << "[Time6]: read all compressed small blocks time: " << readTime << " seconds" << std::endl;

    // Parse header
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

    // --- Data Decompression ---
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
    double decompTime = std::chrono::duration<double>(t_decomp_end - t_decomp_start).count();
    total_data_decomp += decompTime;
    std::cout << "[Time7]: decompress small blocks time: " << decompTime << " seconds" << std::endl;

    return result;
}

// ============================================================
// Gaussian smoothing on all blocks
// ============================================================
void gaussianSmoothBlocks(
    std::vector<std::vector<double>>& blockData,
    const std::vector<size_t>& blockIds,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& blockCountOnEachDim,
    double sigma)
{
    size_t Bx = blockShape[0], By = blockShape[1], Bz = blockShape[2];
    std::unordered_map<size_t, size_t> blockIdToIdx;
    for (size_t i = 0; i < blockIds.size(); ++i)
        blockIdToIdx[blockIds[i]] = i;

    double k0 = std::exp(-0.5 / (sigma * sigma));
    double k1 = 1.0;
    double ksum = 2.0 * k0 + k1;
    k0 /= ksum; k1 /= ksum;
    double kern[3] = {k0, k1, k0};

    std::vector<std::vector<double>> smoothedData(blockData.size());

    for (size_t bidx = 0; bidx < blockIds.size(); ++bidx) {
        size_t blockId = blockIds[bidx];
        const std::vector<double>& src = blockData[bidx];
        size_t block_z = blockId % blockCountOnEachDim[2];
        size_t rem     = blockId / blockCountOnEachDim[2];
        size_t block_y = rem % blockCountOnEachDim[1];
        size_t block_x = rem / blockCountOnEachDim[1];
        std::vector<double> smoothed(Bx * By * Bz, 0.0);

        for (size_t lz = 0; lz < Bz; ++lz)
        for (size_t ly = 0; ly < By; ++ly)
        for (size_t lx = 0; lx < Bx; ++lx) {
            double val = 0.0, wsum = 0.0;
            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                double w = kern[dx+1]*kern[dy+1]*kern[dz+1];
                int nx = (int)lx+dx, ny = (int)ly+dy, nz = (int)lz+dz;
                int bdx=0, bdy=0, bdz=0;
                if      (nx < 0)        { nx += (int)Bx; bdx = -1; }
                else if (nx >= (int)Bx) { nx -= (int)Bx; bdx = +1; }
                if      (ny < 0)        { ny += (int)By; bdy = -1; }
                else if (ny >= (int)By) { ny -= (int)By; bdy = +1; }
                if      (nz < 0)        { nz += (int)Bz; bdz = -1; }
                else if (nz >= (int)Bz) { nz -= (int)Bz; bdz = +1; }
                const std::vector<double>* srcPtr = &src;
                if (bdx || bdy || bdz) {
                    int nbx=(int)block_x+bdx, nby=(int)block_y+bdy, nbz=(int)block_z+bdz;
                    if (nbx<0||nbx>=(int)blockCountOnEachDim[0]||
                        nby<0||nby>=(int)blockCountOnEachDim[1]||
                        nbz<0||nbz>=(int)blockCountOnEachDim[2]) continue;
                    size_t nbrId=(size_t)nbx*blockCountOnEachDim[1]*blockCountOnEachDim[2]
                                +(size_t)nby*blockCountOnEachDim[2]+(size_t)nbz;
                    auto it=blockIdToIdx.find(nbrId);
                    if (it==blockIdToIdx.end()) continue;
                    srcPtr=&blockData[it->second];
                }
                size_t nidx=(size_t)nx+(size_t)ny*Bx+(size_t)nz*Bx*By;
                val+=w*(*srcPtr)[nidx]; wsum+=w;
            }
            size_t cidx=lx+ly*Bx+lz*Bx*By;
            smoothed[cidx]=(wsum>0.0)?(val/wsum):src[cidx];
        }
        smoothedData[bidx]=std::move(smoothed);
    }
    blockData=std::move(smoothedData);
}

// ============================================================
// CSP computation - baseline version
// Same traversal as Isosurface: global space z→y→x,
// then getBlockValue-style direct lookup from global coords
// ============================================================
std::vector<std::vector<double>> computeCSP(
    const std::vector<std::vector<double>>& blocksF,
    const std::vector<std::vector<double>>& blocksG,
    const std::vector<size_t>& allBlockIds,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& dataShape,
    double f_min, double f_max,
    double g_min, double g_max,
    size_t csp_res_f,
    size_t csp_res_g)
{
    size_t Bx = blockShape[0], By = blockShape[1], Bz = blockShape[2];

    size_t blocks_x = (dataShape[0] + Bx - 1) / Bx;
    size_t blocks_y = (dataShape[1] + By - 1) / By;
    size_t blocks_z = (dataShape[2] + Bz - 1) / Bz;

    std::vector<std::vector<double>> cspImage(csp_res_f,
                                              std::vector<double>(csp_res_g, 0.0));
    double df = (f_max - f_min) / csp_res_f;
    double dg = (g_max - g_min) / csp_res_g;

    // 和 Isosurface 的 getBlockValue 完全一样：从全局坐标直接算 blockIndex
    auto getBlockVal = [&](const std::vector<std::vector<double>>& blocks,
                           size_t gx, size_t gy, size_t gz) -> double {
        size_t block_x = gx / Bx;
        size_t block_y = gy / By;
        size_t block_z = gz / Bz;
        size_t blockIndex = block_z
                          + block_y * blocks_z
                          + block_x * blocks_z * blocks_y;
        size_t lx = gx % Bx, ly = gy % By, lz = gz % Bz;
        size_t indexInBlock = lx + ly * Bx + lz * Bx * By;
        return blocks[blockIndex][indexInBlock];
    };

    // 和 Isosurface 完全一样的全局空间坐标遍历：z→y→x
    for (size_t gz = 0; gz < dataShape[2] - 1; ++gz)
    for (size_t gy = 0; gy < dataShape[1] - 1; ++gy)
    for (size_t gx = 0; gx < dataShape[0] - 1; ++gx) {

        // cell 8 个顶点，和 Isosurface 取 cubeValues 完全一样
        double fv[8] = {
            getBlockVal(blocksF, gx,   gy,   gz),
            getBlockVal(blocksF, gx+1, gy,   gz),
            getBlockVal(blocksF, gx+1, gy+1, gz),
            getBlockVal(blocksF, gx,   gy+1, gz),
            getBlockVal(blocksF, gx,   gy,   gz+1),
            getBlockVal(blocksF, gx+1, gy,   gz+1),
            getBlockVal(blocksF, gx+1, gy+1, gz+1),
            getBlockVal(blocksF, gx,   gy+1, gz+1)
        };
        double gv[8] = {
            getBlockVal(blocksG, gx,   gy,   gz),
            getBlockVal(blocksG, gx+1, gy,   gz),
            getBlockVal(blocksG, gx+1, gy+1, gz),
            getBlockVal(blocksG, gx,   gy+1, gz),
            getBlockVal(blocksG, gx,   gy,   gz+1),
            getBlockVal(blocksG, gx+1, gy,   gz+1),
            getBlockVal(blocksG, gx+1, gy+1, gz+1),
            getBlockVal(blocksG, gx,   gy+1, gz+1)
        };

        // Cell-level early rejection
        double cell_fmin=*std::min_element(fv,fv+8);
        double cell_fmax=*std::max_element(fv,fv+8);
        double cell_gmin=*std::min_element(gv,gv+8);
        double cell_gmax=*std::max_element(gv,gv+8);
        if (cell_fmax<f_min||cell_fmin>f_max) continue;
        if (cell_gmax<g_min||cell_gmin>g_max) continue;

        // Gradient
        double grad_fx=0.25*((fv[1]-fv[0])+(fv[2]-fv[3])+(fv[5]-fv[4])+(fv[6]-fv[7]));
        double grad_fy=0.25*((fv[3]-fv[0])+(fv[2]-fv[1])+(fv[7]-fv[4])+(fv[6]-fv[5]));
        double grad_fz=0.25*((fv[4]-fv[0])+(fv[5]-fv[1])+(fv[6]-fv[2])+(fv[7]-fv[3]));
        double grad_gx=0.25*((gv[1]-gv[0])+(gv[2]-gv[3])+(gv[5]-gv[4])+(gv[6]-gv[7]));
        double grad_gy=0.25*((gv[3]-gv[0])+(gv[2]-gv[1])+(gv[7]-gv[4])+(gv[6]-gv[5]));
        double grad_gz=0.25*((gv[4]-gv[0])+(gv[5]-gv[1])+(gv[6]-gv[2])+(gv[7]-gv[3]));

        // |∇f × ∇g|
        double cross_x=grad_fy*grad_gz-grad_fz*grad_gy;
        double cross_y=grad_fz*grad_gx-grad_fx*grad_gz;
        double cross_z=grad_fx*grad_gy-grad_fy*grad_gx;
        double cross_mag=std::sqrt(cross_x*cross_x+cross_y*cross_y+cross_z*cross_z);
        if (cross_mag<1e-12) continue;

        // Cell center
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
    return cspImage;
}

// ============================================================
// Save CSP image
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
    int mpi_rank = 0;

    // --------------------------------------------------------
    // Parameter parsing (same as index version)
    // --------------------------------------------------------
    std::string inputFileName, varNameF, varNameG;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0;
    double relative_error_bound = 1e-3;
    size_t extraValue = 0;
    std::vector<double> queryRangeF, queryRangeG;
    double smoothSigma = 1.2;
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
        else if (arg=="--smooth_sigma"      &&i+1<argc) smoothSigma=std::stod(argv[++i]);
        else if (arg=="--csp_res"           &&i+1<argc) csp_res=std::stoul(argv[++i]);
    }

    if (queryRangeF.size()!=2||queryRangeG.size()!=2) {
        std::cerr<<"Error: need --query_range_f <lo> <hi> and --query_range_g <lo> <hi>"<<std::endl;
        MPI_Finalize(); return 1;
    }

    // Derived parameters
    size_t smallBlocksPerStep=1, smallBlockSize=1;
    for (size_t d=0;d<nDim;d++) {
        smallBlocksPerStep *= stepDataShape[d]/smallBlockShape[d];
        smallBlockSize     *= smallBlockShape[d];
    }
    size_t nSteps=endStepNum-beginStepNum+1;
    size_t totalBlocksNumber=nSteps*smallBlocksPerStep;

    std::vector<size_t> blockCountOnEachDim(3);
    for (size_t i=0;i<3;i++)
        blockCountOnEachDim[i]=(stepDataShape[i]+smallBlockShape[i]-1)/smallBlockShape[i];

    std::string safeF=varNameF; std::replace(safeF.begin(),safeF.end(),'/','_');
    std::string safeG=varNameG; std::replace(safeG.begin(),safeG.end(),'/','_');
    std::string baseName=std::filesystem::path(inputFileName).filename().string();

    // Data directories (same as index version)
    std::string dataDirF = "/home/nyan/scidx/scidx/" + baseName + "_" + safeF + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_blockZFP_originalDataCompression/";
    std::string dataDirG = "/home/nyan/scidx/scidx/" + baseName + "_" + safeG + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_blockZFP_originalDataCompression/";

    // Index directories (only used to read big_block_minmax for error_bound computation)
    std::string indexDirF1 = "/home/nyan/scidx/scidx/" + baseName + "_" + safeF + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_newour_stagger1_index/";
    std::string indexDirG1 = "/home/nyan/scidx/scidx/" + baseName + "_" + safeG + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_newour_stagger1_index/";

    // ---- F 的 error_bound（从 F 的 big_block_minmax 读取全局值域）----
    auto bbF1 = loadBigBlockIndexFile(indexDirF1 + "big_block_minmax");
    double F_globalMin = std::numeric_limits<double>::max();
    double F_globalMax = std::numeric_limits<double>::lowest();
    for (const auto& iv : bbF1) {
        F_globalMin = std::min(F_globalMin, iv.low);
        F_globalMax = std::max(F_globalMax, iv.high);
    }
    double error_bound_F = relative_error_bound * (F_globalMax - F_globalMin);
    std::cout << "error_bound_F = " << error_bound_F << std::endl;

    // ---- G 的 error_bound（从 G 的 big_block_minmax 读取全局值域）----
    auto bbG1 = loadBigBlockIndexFile(indexDirG1 + "big_block_minmax");
    double G_globalMin = std::numeric_limits<double>::max();
    double G_globalMax = std::numeric_limits<double>::lowest();
    for (const auto& iv : bbG1) {
        G_globalMin = std::min(G_globalMin, iv.low);
        G_globalMax = std::max(G_globalMax, iv.high);
    }
    double error_bound_G = relative_error_bound * (G_globalMax - G_globalMin);
    std::cout << "error_bound_G = " << error_bound_G << std::endl;

    auto afterParseTime = std::chrono::high_resolution_clock::now();
    std::cerr << "[Time1] Preprocessing: "
              << std::chrono::duration<double>(afterParseTime-totalStart).count()
              << " seconds" << std::endl;

    // --------------------------------------------------------
    // Build full block list: ALL blocks, no index filtering
    // Same as isosurface baseline: allGlobalSmallBlockIds = [0..N-1]
    // --------------------------------------------------------
    std::vector<size_t> allGlobalSmallBlockIds(totalBlocksNumber);
    for (size_t i = 0; i < totalBlocksNumber; ++i)
        allGlobalSmallBlockIds[i] = i;

    std::cerr << "[CHECK] Total blocks (no filtering): " << totalBlocksNumber << std::endl;

    // --------------------------------------------------------
    // Decompress ALL blocks for F and G
    // --------------------------------------------------------
    double total_data_read=0, total_data_decomp=0;

    std::cout << "\n[Baseline] Decompressing ALL blocks for F..." << std::endl;
    auto decompressedF = universalBatchDecompressBlocksAllRead(
        dataDirF, allGlobalSmallBlockIds, smallBlockSize, totalBlocksNumber,
        error_bound_F, total_data_read, total_data_decomp);

    std::cout << "\n[Baseline] Decompressing ALL blocks for G..." << std::endl;
    auto decompressedG = universalBatchDecompressBlocksAllRead(
        dataDirG, allGlobalSmallBlockIds, smallBlockSize, totalBlocksNumber,
        error_bound_G, total_data_read, total_data_decomp);

    std::cerr << "[Time] Data Read   (F+G total): " << total_data_read   << " seconds" << std::endl;
    std::cerr << "[Time] Data Decomp (F+G total): " << total_data_decomp << " seconds" << std::endl;

    // --------------------------------------------------------
    // Gaussian smoothing on ALL blocks
    // --------------------------------------------------------
    auto t_smooth_start=std::chrono::high_resolution_clock::now();
    /*gaussianSmoothBlocks(decompressedF, allGlobalSmallBlockIds, smallBlockShape,
                         blockCountOnEachDim, smoothSigma);
    gaussianSmoothBlocks(decompressedG, allGlobalSmallBlockIds, smallBlockShape,
                         blockCountOnEachDim, smoothSigma);*/
    auto t_smooth_end=std::chrono::high_resolution_clock::now();
    std::cerr << "[Time] Gaussian Smoothing (F+G): "
              << std::chrono::duration<double>(t_smooth_end-t_smooth_start).count()
              << " seconds" << std::endl;

    // --------------------------------------------------------
    // CSP computation on ALL blocks
    // (cell-level early rejection with query range still applied)
    // --------------------------------------------------------
    auto t_csp_start=std::chrono::high_resolution_clock::now();

    auto cspImage = computeCSP(
        decompressedF, decompressedG,
        allGlobalSmallBlockIds,
        smallBlockShape, stepDataShape,
        queryRangeF[0], queryRangeF[1],
        queryRangeG[0], queryRangeG[1],
        csp_res, csp_res);

    auto t_csp_end=std::chrono::high_resolution_clock::now();
    std::cerr << "[Time] CSP Computation: "
              << std::chrono::duration<double>(t_csp_end-t_csp_start).count()
              << " seconds" << std::endl;

    // Save CSP image
    std::string outFile = "csp_" + safeF + "_" + safeG + "_decompressed_baseline.bin";
    saveCSPImage(cspImage, outFile);

    // --------------------------------------------------------
    // Time summary
    // --------------------------------------------------------
    auto totalEnd=std::chrono::high_resolution_clock::now();
    double totalTime=std::chrono::duration<double>(totalEnd-totalStart).count();

    std::cerr << "\n========== TIME SUMMARY (BASELINE) ==========" << std::endl;
    std::cerr << "[Stage 1] Preprocessing:               "
              << std::chrono::duration<double>(afterParseTime-totalStart).count() << " s" << std::endl;
    std::cerr << "[Stage 2] Data Read   (F+G):           " << total_data_read   << " s" << std::endl;
    std::cerr << "[Stage 2] Data Decomp (F+G):           " << total_data_decomp << " s" << std::endl;
    std::cerr << "[Stage 3] Gaussian Smoothing (F+G):    "
              << std::chrono::duration<double>(t_smooth_end-t_smooth_start).count() << " s" << std::endl;
    std::cerr << "[Stage 4] CSP Computation:             "
              << std::chrono::duration<double>(t_csp_end-t_csp_start).count() << " s" << std::endl;
    std::cerr << "[TOTAL]                                " << totalTime << " s" << std::endl;
    std::cerr << "=============================================" << std::endl;

    MPI_Finalize();
    return 0;
}

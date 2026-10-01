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

// 自包含的通用解压函数，无需额外依赖


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




/*struct BlockConfig {
    size_t blockSize;
    size_t unpredSizeBits;
    size_t dataSizeBytes;
    size_t signBytesPerBlock;
    
    BlockConfig(size_t bs) : blockSize(bs) {
        if (bs == 64) {
            unpredSizeBits = 7;   // 6 → 7
            dataSizeBytes = 1;
            signBytesPerBlock = 8;
        } else if (bs == 4096) {
            unpredSizeBits = 13;  // 12 → 13
            dataSizeBytes = 2;
            signBytesPerBlock = 512;
        } else if (bs == 262144) {
            unpredSizeBits = 19;  // 18 → 19
            dataSizeBytes = 3;
            signBytesPerBlock = 32768;
        } else if (bs == 16777216) {
            unpredSizeBits = 25;  // 24 → 25
            dataSizeBytes = 4;
            signBytesPerBlock = 2097152;
        } else if (bs == 1073741824) {
            unpredSizeBits = 31;  // 30 → 31
            dataSizeBytes = 4;
            signBytesPerBlock = 134217728;
        } else {
            throw std::runtime_error("Unsupported block size");
        }
    }
};*/

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


// 通用解压函数 - 全量读取版本
std::vector<std::vector<double>> universalBatchDecompressBlocksAllRead(
    const std::string& subDir,
    const std::vector<size_t>& blockIds,
    size_t blockSize,
    size_t totalBlocks,
    double error_bound) {
    
    std::cout << "universal all read: " << std::endl;
    auto start_time_read = std::chrono::high_resolution_clock::now();
    
    // 获取块配置
    BlockConfig config(blockSize);
    
    std::vector<std::vector<double>> decompressedOriginalresult;
    
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
    
    // 读取所有文件到内存
    std::vector<uint8_t> allUnpredData(unpredFileSize);
    std::vector<uint8_t> allCompData(compFileSize);
    std::vector<uint8_t> allSignData(signFileSize);
    
    unpredStream.read(reinterpret_cast<char*>(allUnpredData.data()), unpredFileSize);
    compStream.read(reinterpret_cast<char*>(allCompData.data()), compFileSize);
    signStream.read(reinterpret_cast<char*>(allSignData.data()), signFileSize);
    
    // 解析unpredSizes
    size_t unpredSizeBytes = (totalBlocks * config.unpredSizeBits + 7) / 8;
    std::vector<uint8_t> compressedUnpredSizes(allUnpredData.begin(),
                                              allUnpredData.begin() + unpredSizeBytes);
    std::vector<size_t> unpredSizes;
    unpackBitsInline(compressedUnpredSizes, unpredSizes, config.unpredSizeBits, totalBlocks);
    
    // 读取bitCounts
    std::vector<uint8_t> bitCounts(totalBlocks);
    std::memcpy(bitCounts.data(), allCompData.data(), totalBlocks);
    
    // 读取compSizes
    std::vector<size_t> compSizes(totalBlocks);
    for (size_t i = 0; i < totalBlocks; i++) {
        size_t offset = totalBlocks + i * config.dataSizeBytes;
        compSizes[i] = static_cast<size_t>(readValueInline(
            allCompData.data() + offset, config.dataSizeBytes));
    }
    
    auto begin_decompress = std::chrono::high_resolution_clock::now();
    
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
    std::vector<int> quant_inds(blockSize);
    std::vector<double> decompressedBlock(blockSize);
    
    for (size_t bid : blockIds) {
        // 读取unpredData
        size_t unpredSize = unpredSizes[bid];
        size_t unpredOffset = unpredSizeBytes + sizeof(double) * unpredOffsets[bid];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredData.data() + unpredOffset,
                   unpredSize * sizeof(double));
        
        // 读取压缩数据
        uint8_t bitCount = bitCounts[bid];
        size_t dataSize = compSizes[bid];
        size_t compOffset = totalBlocks + totalBlocks * config.dataSizeBytes + compOffsets[bid];
        compData.resize(dataSize);
        std::memcpy(compData.data(), allCompData.data() + compOffset, dataSize);
        
        // 读取符号数据
        size_t signOffset = config.signBytesPerBlock * bid;
        std::memcpy(signBits.data(), allSignData.data() + signOffset, config.signBytesPerBlock);
        
        // 解码数据
        int radius = 512;
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
        
        const SZ3::uchar* metaPtr = metadataBuffer.data();
        size_t metaSize = metadataBuffer.size();
        
        decompose.load(metaPtr, metaSize);
        decompose.decompress(conf, quant_inds, decompressedBlock.data());
        decompressedOriginalresult.push_back(decompressedBlock);
    }
    
    auto end_decompress = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> readTime = begin_decompress - start_time_read;
    std::chrono::duration<double> compressTime = end_decompress - begin_decompress;
    
    std::cout << "[Time6]: read all compressed small blocks time: " << readTime.count() << " seconds" << std::endl;
    std::cout << "[Time7]: decompress small blocks time: " << compressTime.count() << " seconds" << std::endl;
    
    return decompressedOriginalresult;
}

std::vector<std::vector<double>> batchDecompressBlocksAllRead(const std::string& subDir,
        const std::vector<size_t>& blockIds,
        size_t blockSize,
        size_t totalBlocks,
        double error_bound,
        const std::unordered_map<size_t, size_t>& globalToLocal) {
    auto start_time_read = std::chrono::high_resolution_clock::now();      
    std::vector<std::vector<double>> decompressedOriginalresult;

    std::cout << "[Debug] Number of blocks to decompress: " << blockIds.size() << std::endl;

    std::cout << "mySubDir = " << subDir << "\n";



    std::string unpredDataFileName = subDir + "universal_unpredData.bin";
    std::string compressedFileName = subDir + "universal_compressed_data.bin";
    std::string signFileName = subDir + "universal_sign_data.bin";

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
    std::vector<double> unpredData;
    std::vector<unsigned char> compData;
    std::vector<unsigned char> signBits(signBytesPerBlock);
    std::vector<int> quant_inds(blockSize);
    std::vector<double> decompressedBlock(blockSize);


    //double memory_access_time = 0.0;
    //double decompress_only_time = 0.0;

    for (size_t bid : blockIds) {
        size_t local_id = globalToLocal.at(bid); 
        //auto t1 = std::chrono::high_resolution_clock::now();
        

        // Unpred
        unsigned char unpredSize = unpredSizes[local_id];
        size_t unpredOffset = totalBlocks + sizeof(double) * unpredOffsets[local_id];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredData.data() + unpredOffset, unpredSize * sizeof(double));

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
        //memory_access_time += std::chrono::duration<double>(t2 - t1).count();

        
       // 解码数据
       int radius = 512;
       std::vector<int> quant_inds(blockSize);
       /*size_t bitPos = 0, dataIdx = 0;

       for (size_t i = 0; i < blockSize; ++i) {
           unsigned int val = 0, bitsRead = 0;
           while (bitsRead < bitCount && dataIdx < compData.size()) {
               unsigned int available = std::min<unsigned int>(bitCount - bitsRead, 8u - static_cast<unsigned int>(bitPos % 8));
               val |= ((compData[dataIdx] >> (bitPos % 8)) & ((1 << available) - 1)) << bitsRead;
               bitsRead += available;
               bitPos += available;
               if (bitPos % 8 == 0) ++dataIdx;
           }

           int sign = (signBits[i / 8] >> (i % 8)) & 1;
           quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);
           quant_inds[i] += radius;
       }*/

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



        
        // === Metadata + decompression ===
        std::vector<SZ3::uchar> metadataBuffer;
        metadataBuffer.push_back(0b00000010);
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&error_bound),
            reinterpret_cast<SZ3::uchar*>(&error_bound) + sizeof(double));
       
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

        auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);

        const SZ3::uchar* metaPtr = metadataBuffer.data();
        size_t metaSize = metadataBuffer.size();

        decompose.load(metaPtr, metaSize);
        decompose.decompress(conf, quant_inds, decompressedBlock.data());
        decompressedOriginalresult.push_back(decompressedBlock);  
    }

    
    

    // 使用已读取的数据进行解压
    /*for (size_t bid : blockIds) {
        auto start_unpack1 = std::chrono::high_resolution_clock::now();


        size_t unpredOffset = totalBlocks + sizeof(double) * std::accumulate(unpredSizes.begin(), unpredSizes.begin() + bid, 0);
        unsigned char unpredSize = unpredSizes[bid];
        std::vector<double> unpredData(unpredSize);

        std::memcpy(unpredData.data(), allUnpredData.data()+ unpredOffset, unpredSize * sizeof(double));

        // === 获取压缩数据 ===
        unsigned char bitCount = bitCounts[bid];

        //对应的fixlength encodeing 后的bytes数量（不满一个字节，高位填0）
        unsigned char dataSize = compSizes[bid];

        size_t compOffset = 2 * totalBlocks + std::accumulate(compSizes.begin(), compSizes.begin() + bid, size_t(0));
        
        std::vector<unsigned char> compData(dataSize);

        std::memcpy(compData.data(), allCompData.data()+ compOffset, dataSize);



        size_t signOffset = signBytesPerBlock * bid;
        std::vector<unsigned char> signBits(signBytesPerBlock);
        std::memcpy(signBits.data(), allSignData.data()+ signOffset, signBytesPerBlock);

        // 解码数据
        int radius = 512;
        std::vector<int> quant_inds(blockSize);
        size_t bitPos = 0, dataIdx = 0;

        /*for (size_t i = 0; i < blockSize; ++i) {
            unsigned int val = 0, bitsRead = 0;
            while (bitsRead < bitCount && dataIdx < compData.size()) {
                unsigned int available = std::min<unsigned int>(bitCount - bitsRead, 8u - static_cast<unsigned int>(bitPos % 8));
                val |= ((compData[dataIdx] >> (bitPos % 8)) & ((1 << available) - 1)) << bitsRead;
                bitsRead += available;
                bitPos += available;
                if (bitPos % 8 == 0) ++dataIdx;
            }

            int sign = (signBits[i / 8] >> (i % 8)) & 1;
            quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);
            quant_inds[i] += radius;
        }


        auto end_unpack = std::chrono::high_resolution_clock::now();
        total_unpack_time1 += std::chrono::duration<double>(end_unpack - start_unpack1).count();


        auto start_unpack2 = std::chrono::high_resolution_clock::now();



        for (size_t i = 0; i < blockSize; ++i) {
            size_t byteIndex = bitPos / 8;
            size_t bitOffset = bitPos % 8;
        
            uint32_t buffer = 0;
            memcpy(&buffer, compData.data() + byteIndex, sizeof(uint32_t));
        
            uint32_t val = (buffer >> bitOffset) & ((1u << bitCount) - 1);
        
            int sign = (signBits[i / 8] >> (i % 8)) & 1;
            quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);
            quant_inds[i] += radius;
        
            bitPos += bitCount;
        }

        auto end_decompress2 = std::chrono::high_resolution_clock::now();
        total_decompress_time2 += std::chrono::duration<double>(end_decompress2 - start_unpack2).count();



        
        auto start_unpack3 = std::chrono::high_resolution_clock::now();
        // 恢复 metadataBuffer
        std::vector<SZ3::uchar> metadataBuffer;
        metadataBuffer.push_back(0b00000010);  // 标志位
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

        // 解压缩
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

        auto start_unpack4 = std::chrono::high_resolution_clock::now();

        
        total_decompress_time3 += std::chrono::duration<double>(start_unpack4 - start_unpack3).count();


        decompressedOriginalresult.push_back(std::move(decompressedBlock));
    }*/


    //std::cout << "[Time_mem_access]: Memory access time: " << memory_access_time << " seconds" << std::endl;


    auto end_decompress = std::chrono::high_resolution_clock::now(); 
    

    std::chrono::duration<double> readTime = begin_decompress - start_time_read;
    
    std::chrono::duration<double> compressTime = end_decompress - begin_decompress;

    /*int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);*/
    
    std::cout <<"[Time2]:Total read time: " << readTime.count() << " seconds" << std::endl;
    
    std::cout << "Time3]:Total decompress time: " << compressTime.count() << " seconds" << std::endl;

    return decompressedOriginalresult;
}


 
    //baseline
    std::vector<std::vector<double>> rangeFilterData(
        const std::vector<std::vector<double>>& blocks,
        double queryLow,
        double queryHigh) 
    {
        std::vector<std::vector<double>> filteredBlocks;
    
        for (const auto& block : blocks) {
            std::vector<double> filtered;
            for (double val : block) {
                if (val >= queryLow && val <= queryHigh) {
                    filtered.push_back(val);
                }
            }
            if (!filtered.empty()) {
                filteredBlocks.push_back(std::move(filtered));
            }
            

        }


    
        return filteredBlocks;
    }

// 在分块存储的数据里，根据全局坐标 (global_x, global_y, global_z) 找到对应的数值
// 压缩 / 解压后的数据是按照块存储的
// 全局坐标是全局空间坐标
inline double getBlockValue(
    const std::vector<std::vector<double>>& blockData,
    size_t global_x, size_t global_y, size_t global_z,
    const std::vector<size_t>& blockShape,
    size_t blocks_x, size_t blocks_y, size_t blocks_z)
{
    // ------------------------------------------------------------
    // 1. 计算 block 坐标
    // ------------------------------------------------------------
    size_t block_x = global_x / blockShape[0];
    size_t block_y = global_y / blockShape[1];
    size_t block_z = global_z / blockShape[2];

    // ------------------------------------------------------------
    // 2. block 线性索引
    // blockIndex = z + y * Z + x * Z * Y
    // ------------------------------------------------------------
    size_t blockIndex =
        block_z +
        block_y * blocks_z +
        block_x * blocks_z * blocks_y;

    // ------------------------------------------------------------
    // 3. block 内局部坐标
    // ------------------------------------------------------------
    size_t local_x = global_x % blockShape[0];
    size_t local_y = global_y % blockShape[1];
    size_t local_z = global_z % blockShape[2];

    // ------------------------------------------------------------
    // 4. block 内线性索引
    // index = x + y * X + z * X * Y
    // ------------------------------------------------------------
    size_t indexInBlock =
        local_x +
        local_y * blockShape[0] +
        local_z * blockShape[0] * blockShape[1];

    // ------------------------------------------------------------
    // 5. 返回 double 值
    // ------------------------------------------------------------
    return blockData[blockIndex][indexInBlock];
}


// ============================================================
// 将分块数据重组为线性3D数组
// ============================================================
std::vector<double> ReassembleBlocksToVolume(
    const std::vector<std::vector<double>>& decompressedBlocks,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& dataShape)
{
    size_t totalElements = dataShape[0] * dataShape[1] * dataShape[2];
    std::vector<double> volume(totalElements, 0.0);

    size_t blocks_x = (dataShape[0] + blockShape[0] - 1) / blockShape[0];
    size_t blocks_y = (dataShape[1] + blockShape[1] - 1) / blockShape[1];
    size_t blocks_z = (dataShape[2] + blockShape[2] - 1) / blockShape[2];

    for (size_t bx = 0; bx < blocks_x; ++bx) {
        for (size_t by = 0; by < blocks_y; ++by) {
            for (size_t bz = 0; bz < blocks_z; ++bz) {

                size_t blockIndex = bz + by * blocks_z + bx * blocks_z * blocks_y;

                for (size_t lx = 0; lx < blockShape[0]; ++lx) {
                    for (size_t ly = 0; ly < blockShape[1]; ++ly) {
                        for (size_t lz = 0; lz < blockShape[2]; ++lz) {

                            size_t gx = bx * blockShape[0] + lx;
                            size_t gy = by * blockShape[1] + ly;
                            size_t gz = bz * blockShape[2] + lz;

                            // 跳过越界（末尾块可能不满）
                            if (gx >= dataShape[0] || gy >= dataShape[1] || gz >= dataShape[2])
                                continue;

                            // 块内线性索引（与压缩时保持一致）
                            size_t indexInBlock = lx + ly * blockShape[0] + lz * blockShape[0] * blockShape[1];

                            // 全局线性索引（z-major，与 getBlockValue 的坐标系一致）
                            size_t globalIndex = gx + gy * dataShape[0] + gz * dataShape[0] * dataShape[1];

                            volume[globalIndex] = decompressedBlocks[blockIndex][indexInBlock];
                        }
                    }
                }
            }
        }
    }

    return volume;
}


// ============================================================
// 将3D volume数据保存为VTK ImageData (.vti) 文件
// 可直接在 ParaView / VisIt 中打开做体绘制、等值面等可视化
// ============================================================
void SaveVolumeAsVTI(
    const std::vector<double>& volume,
    const std::vector<size_t>& dataShape,   // {nx, ny, nz}
    const std::string& outFile,
    const std::string& fieldName = "Pressure",
    double spacingX = 1.0,
    double spacingY = 1.0,
    double spacingZ = 1.0)
{
    size_t nx = dataShape[0];
    size_t ny = dataShape[1];
    size_t nz = dataShape[2];

    std::ofstream ofs(outFile);
    if (!ofs) {
        std::cerr << "[Error] Cannot open output file: " << outFile << std::endl;
        return;
    }

    // ---- VTK XML ImageData header ----
    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    ofs << "  <ImageData WholeExtent=\"0 " << (nx-1) << " 0 " << (ny-1) << " 0 " << (nz-1) << "\"\n";
    ofs << "             Origin=\"0 0 0\"\n";
    ofs << "             Spacing=\"" << spacingX << " " << spacingY << " " << spacingZ << "\">\n";
    ofs << "    <Piece Extent=\"0 " << (nx-1) << " 0 " << (ny-1) << " 0 " << (nz-1) << "\">\n";
    ofs << "      <PointData Scalars=\"" << fieldName << "\">\n";
    ofs << "        <DataArray type=\"Float64\" Name=\"" << fieldName
        << "\" format=\"ascii\" NumberOfComponents=\"1\">\n";

    // ---- 写数据（VTK ImageData 的点顺序: x 变化最快, z 最慢）----
    // 注意：如果你的 volume 索引是 gx + gy*nx + gz*nx*ny，这里顺序刚好对应
    ofs << std::scientific;
    ofs.precision(8);
    for (size_t gz = 0; gz < nz; ++gz) {
        for (size_t gy = 0; gy < ny; ++gy) {
            for (size_t gx = 0; gx < nx; ++gx) {
                size_t idx = gx + gy * nx + gz * nx * ny;
                ofs << volume[idx];
                if (gx + 1 < nx) ofs << " ";
            }
            ofs << "\n";
        }
    }

    ofs << "        </DataArray>\n";
    ofs << "      </PointData>\n";
    ofs << "    </Piece>\n";
    ofs << "  </ImageData>\n";
    ofs << "</VTKFile>\n";

    std::cout << "[Saved] VTI file: " << outFile
              << " (" << nx << "x" << ny << "x" << nz << ")\n";
}



util::TriangleMesh<double> RunMarchingCubesOnDecompressedBlocksOptimized(
    const std::vector<std::vector<double>>& decompressedBlocks,
    double isovalue,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& dataShape
)
{
    // ------------------------------------------------------------
    // 预计算块维度
    // ------------------------------------------------------------
    size_t blocks_x = (dataShape[0] + blockShape[0] - 1) / blockShape[0];
    size_t blocks_y = (dataShape[1] + blockShape[1] - 1) / blockShape[1];
    size_t blocks_z = (dataShape[2] + blockShape[2] - 1) / blockShape[2];

    // ------------------------------------------------------------
    // 全局结果容器（double）
    // ------------------------------------------------------------
    std::vector<std::array<double, 3>> globalPoints;
    std::vector<std::array<double, 3>> globalNormals;
    std::vector<std::array<int, 3>>    globalTriangles;

    // global edge -> point index
    std::unordered_map<size_t, int> globalPointMap;
    int ptIdx = 0;

    size_t processedCubes = 0;

    // ------------------------------------------------------------
    // 全局 cube 遍历（严格按 MC 原始顺序）
    // ------------------------------------------------------------
    size_t skippedDenom = 0;
    for (size_t z = 0; z < dataShape[2] - 1; ++z) {
        for (size_t y = 0; y < dataShape[1] - 1; ++y) {
            for (size_t x = 0; x < dataShape[0] - 1; ++x) {

                // ------------------------------------------------
                // 1. cube 8 个顶点值（double）
                // ------------------------------------------------
                std::array<double, 8> cubeValues = {{
                    getBlockValue(decompressedBlocks, x,   y,   z,   blockShape, blocks_x, blocks_y, blocks_z),
                    getBlockValue(decompressedBlocks, x+1, y,   z,   blockShape, blocks_x, blocks_y, blocks_z),
                    getBlockValue(decompressedBlocks, x+1, y+1, z,   blockShape, blocks_x, blocks_y, blocks_z),
                    getBlockValue(decompressedBlocks, x,   y+1, z,   blockShape, blocks_x, blocks_y, blocks_z),
                    getBlockValue(decompressedBlocks, x,   y,   z+1, blockShape, blocks_x, blocks_y, blocks_z),
                    getBlockValue(decompressedBlocks, x+1, y,   z+1, blockShape, blocks_x, blocks_y, blocks_z),
                    getBlockValue(decompressedBlocks, x+1, y+1, z+1, blockShape, blocks_x, blocks_y, blocks_z),
                    getBlockValue(decompressedBlocks, x,   y+1, z+1, blockShape, blocks_x, blocks_y, blocks_z)
                }};

                int cellCaseId = util::findCaseId(cubeValues, isovalue);
                if (cellCaseId == 0 || cellCaseId == 255) {
                    continue;
                }
                processedCubes++;

                // ------------------------------------------------
                // 2. cube 8 个顶点坐标（double）
                // ------------------------------------------------
                std::array<std::array<double, 3>, 8> cubePositions = {{
                    {double(x),   double(y),   double(z)},
                    {double(x+1), double(y),   double(z)},
                    {double(x+1), double(y+1), double(z)},
                    {double(x),   double(y+1), double(z)},
                    {double(x),   double(y),   double(z+1)},
                    {double(x+1), double(y),   double(z+1)},
                    {double(x+1), double(y+1), double(z+1)},
                    {double(x),   double(y+1), double(z+1)}
                }};

                // ------------------------------------------------
                // 3. cube 梯度（double）
                // ------------------------------------------------
                std::array<std::array<double, 3>, 8> cubeGradients;

                for (int i = 0; i < 8; ++i) {
                    size_t px = x + (i & 1);
                    size_t py = y + ((i >> 1) & 1);
                    size_t pz = z + ((i >> 2) & 1);

                    std::array<double, 3> grad = {0.0, 0.0, 0.0};

                    // X
                    if (px == 0) {
                        grad[0] =
                            getBlockValue(decompressedBlocks, px,   py, pz, blockShape, blocks_x, blocks_y, blocks_z) -
                            getBlockValue(decompressedBlocks, px+1, py, pz, blockShape, blocks_x, blocks_y, blocks_z);
                    } else if (px == dataShape[0] - 1) {
                        grad[0] =
                            getBlockValue(decompressedBlocks, px-1, py, pz, blockShape, blocks_x, blocks_y, blocks_z) -
                            getBlockValue(decompressedBlocks, px,   py, pz, blockShape, blocks_x, blocks_y, blocks_z);
                    } else {
                        grad[0] =
                            (getBlockValue(decompressedBlocks, px-1, py, pz, blockShape, blocks_x, blocks_y, blocks_z) -
                             getBlockValue(decompressedBlocks, px+1, py, pz, blockShape, blocks_x, blocks_y, blocks_z)) / 2.0;
                    }

                    // Y
                    if (py == 0) {
                        grad[1] =
                            getBlockValue(decompressedBlocks, px, py,   pz, blockShape, blocks_x, blocks_y, blocks_z) -
                            getBlockValue(decompressedBlocks, px, py+1, pz, blockShape, blocks_x, blocks_y, blocks_z);
                    } else if (py == dataShape[1] - 1) {
                        grad[1] =
                            getBlockValue(decompressedBlocks, px, py-1, pz, blockShape, blocks_x, blocks_y, blocks_z) -
                            getBlockValue(decompressedBlocks, px, py,   pz, blockShape, blocks_x, blocks_y, blocks_z);
                    } else {
                        grad[1] =
                            (getBlockValue(decompressedBlocks, px, py-1, pz, blockShape, blocks_x, blocks_y, blocks_z) -
                             getBlockValue(decompressedBlocks, px, py+1, pz, blockShape, blocks_x, blocks_y, blocks_z)) / 2.0;
                    }

                    // Z
                    if (pz == 0) {
                        grad[2] =
                            getBlockValue(decompressedBlocks, px, py, pz,   blockShape, blocks_x, blocks_y, blocks_z) -
                            getBlockValue(decompressedBlocks, px, py, pz+1, blockShape, blocks_x, blocks_y, blocks_z);
                    } else if (pz == dataShape[2] - 1) {
                        grad[2] =
                            getBlockValue(decompressedBlocks, px, py, pz-1, blockShape, blocks_x, blocks_y, blocks_z) -
                            getBlockValue(decompressedBlocks, px, py, pz,   blockShape, blocks_x, blocks_y, blocks_z);
                    } else {
                        grad[2] =
                            (getBlockValue(decompressedBlocks, px, py, pz-1, blockShape, blocks_x, blocks_y, blocks_z) -
                             getBlockValue(decompressedBlocks, px, py, pz+1, blockShape, blocks_x, blocks_y, blocks_z)) / 2.0;
                    }

                    cubeGradients[i] = grad;
                }

                // ------------------------------------------------
                // 4. 生成三角形
                // ------------------------------------------------
                const int* triEdges = util::caseTrianglesEdges[cellCaseId];
                

                for (; *triEdges != -1; triEdges += 3) {
                    std::array<int, 3> tri;

                    for (int i = 0; i < 3; ++i) {
                        int edgeIdx = triEdges[i];

                        size_t globalEdgeIdx =
                            ((z * (dataShape[1] - 1) + y) * (dataShape[0] - 1) + x) * 12 + edgeIdx;

                        auto it = globalPointMap.find(globalEdgeIdx);
                        if (it != globalPointMap.end()) {
                            tri[i] = it->second;
                        } else {
                            const int* vs = util::edgeVertices[edgeIdx];
                            int v1 = vs[0];
                            int v2 = vs[1];

                            
                            double denom = cubeValues[v2] - cubeValues[v1];
                            if (std::abs(denom) < 1e-12) {
                                skippedDenom++;
                                continue;
                            }

                            double w = (isovalue - cubeValues[v1]) / denom;

                            std::array<double, 3> newPt =
                                util::interpolate(cubePositions[v1], cubePositions[v2], w);
                            std::array<double, 3> newNorm =
                                util::interpolate(cubeGradients[v1], cubeGradients[v2], w);

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

    std::cout << "Total skippedDenom = "
          << skippedDenom << std::endl;

    return util::TriangleMesh<double>(
        globalPoints,
        globalNormals,
        globalTriangles
    );
}

void gaussianSmoothBlocks(
    std::vector<std::vector<double>>& blockData,
    const std::vector<size_t>& blockIds,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& blockCountOnEachDim,
    double sigma)
{
    size_t Bx = blockShape[0];
    size_t By = blockShape[1];
    size_t Bz = blockShape[2];

    // 建立 blockId -> index 的映射，方便跨块访问
    std::unordered_map<size_t, size_t> blockIdToIdx;
    for (size_t i = 0; i < blockIds.size(); ++i) {
        blockIdToIdx[blockIds[i]] = i;
    }

    // 计算1D高斯核 (size=3, 即-1,0,+1)
    double k0 = std::exp(-0.5 * 1.0 / (sigma * sigma));  // offset=1
    double k1 = 1.0;                                       // offset=0
    double ksum = 2.0 * k0 + k1;
    k0 /= ksum;
    k1 /= ksum;
    // kern[0]=k0 (offset=-1), kern[1]=k1 (offset=0), kern[2]=k0 (offset=+1)
    double kern[3] = {k0, k1, k0};

    // 对每个块做平滑，结果写入新的vector
    std::vector<std::vector<double>> smoothedData(blockData.size());

    for (size_t bidx = 0; bidx < blockIds.size(); ++bidx) {
        size_t blockId = blockIds[bidx];
        const std::vector<double>& src = blockData[bidx];

        // 解码块3D坐标
        size_t block_z = blockId % blockCountOnEachDim[2];
        size_t rem     = blockId / blockCountOnEachDim[2];
        size_t block_y = rem % blockCountOnEachDim[1];
        size_t block_x = rem / blockCountOnEachDim[1];

        std::vector<double> smoothed(Bx * By * Bz, 0.0);

        for (size_t lz = 0; lz < Bz; ++lz) {
        for (size_t ly = 0; ly < By; ++ly) {
        for (size_t lx = 0; lx < Bx; ++lx) {

            double val  = 0.0;
            double wsum = 0.0;

            for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {

                double w = kern[dx+1] * kern[dy+1] * kern[dz+1];

                int nx = (int)lx + dx;
                int ny = (int)ly + dy;
                int nz = (int)lz + dz;

                // 判断是否需要跨块
                int bdx = 0, bdy = 0, bdz = 0;
                if      (nx < 0)        { nx += (int)Bx; bdx = -1; }
                else if (nx >= (int)Bx) { nx -= (int)Bx; bdx = +1; }
                if      (ny < 0)        { ny += (int)By; bdy = -1; }
                else if (ny >= (int)By) { ny -= (int)By; bdy = +1; }
                if      (nz < 0)        { nz += (int)Bz; bdz = -1; }
                else if (nz >= (int)Bz) { nz -= (int)Bz; bdz = +1; }

                const std::vector<double>* srcPtr = &src;

                if (bdx != 0 || bdy != 0 || bdz != 0) {
                    // 需要访问邻居块
                    int nbx = (int)block_x + bdx;
                    int nby = (int)block_y + bdy;
                    int nbz = (int)block_z + bdz;

                    if (nbx < 0 || nbx >= (int)blockCountOnEachDim[0] ||
                        nby < 0 || nby >= (int)blockCountOnEachDim[1] ||
                        nbz < 0 || nbz >= (int)blockCountOnEachDim[2]) {
                        // 超出数据边界，跳过这个邻居（权重归一化会自动处理）
                        continue;
                    }

                    size_t nbrId = (size_t)nbx * blockCountOnEachDim[1] * blockCountOnEachDim[2]
                                 + (size_t)nby * blockCountOnEachDim[2]
                                 + (size_t)nbz;

                    auto it = blockIdToIdx.find(nbrId);
                    if (it == blockIdToIdx.end()) {
                        // 邻居块没有被加载（在expanded范围之外），跳过
                        continue;
                    }
                    srcPtr = &blockData[it->second];
                }

                size_t nidx = (size_t)nx + (size_t)ny * Bx + (size_t)nz * Bx * By;
                val  += w * (*srcPtr)[nidx];
                wsum += w;

            }}}

            size_t cidx = lx + ly * Bx + lz * Bx * By;
            smoothed[cidx] = (wsum > 0.0) ? (val / wsum) : src[cidx];

        }}}

        smoothedData[bidx] = std::move(smoothed);
    }

    // 写回
    blockData = std::move(smoothedData);
}


void RunAndSaveIsosurfaceMesh(
    const std::vector<std::vector<double>>& decompressedBlocks,
    double isovalue,
    const std::vector<size_t>& smallBlockShape,
    const std::vector<size_t>& stepDataShape,
    const std::string& outFile
) {
    // 打印基本信息
    std::cout << "[Info] Running Marching Cubes on decompressed blocks\n";
    std::cout << "       Volume shape: ("
              << stepDataShape[0] << ", "
              << stepDataShape[1] << ", "
              << stepDataShape[2] << ")\n";
    std::cout << "       Block shape:  ("
              << smallBlockShape[0] << ", "
              << smallBlockShape[1] << ", "
              << smallBlockShape[2] << ")\n";
    std::cout << "       Isovalue:     " << isovalue << "\n";
    std::cout << "       Output file:  " << outFile << "\n";

    // ------------------------------------------------------------
    // 启动计时器
    // ------------------------------------------------------------
    util::Timer timer;
    timer.start();

    // ------------------------------------------------------------
    // 执行 Marching Cubes（double 版本）
    // ------------------------------------------------------------
    util::TriangleMesh<double> mesh =
        RunMarchingCubesOnDecompressedBlocksOptimized(
            decompressedBlocks,
            isovalue,
            smallBlockShape,
            stepDataShape
        );

    timer.stop();

    // ------------------------------------------------------------
    // 打印结果信息
    // ------------------------------------------------------------
    std::cout << "[Result] Mesh vertices:  "
              << mesh.numberOfVertices() << "\n";
    std::cout << "[Result] Mesh triangles: "
              << mesh.numberOfTriangles() << "\n";
    std::cout << "[Timing] CPU time (sec):  "
              << timer.getCPUtime() << "\n";
    std::cout << "[Timing] Wall time (sec): "
              << timer.getWallTime() << "\n";

    // ------------------------------------------------------------
    // 写入 VTK（double mesh）
    // ------------------------------------------------------------
    util::saveTriangleMesh(mesh, outFile.c_str());
}


    



int main(int argc, char* argv[]) {

    auto totalStart = std::chrono::high_resolution_clock::now(); 

    /*MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);*/


    int mpi_size = 1;
    int mpi_rank = 0;

    std::string inputFileName, variableName, indexDir;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, blockShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0, bigBlocksPerStep = 1, smallBlocksPerBig = 1 ,  smallBlockSize = 1 , totalBlocksNumber = 1, smallBlocksPerStep = 1; 
    std::vector<double> queryRange;
    double relative_error_bound = 1E-3;
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

    

    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    //index所在文件夹的根路径
    indexDir = "/home/nyan/scidx/scidx/" + std::filesystem::path(inputFileName).filename().string() + "_" +safeVarName + "_" + 
                std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) + "_newour_index/";



    std::string mySubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" +safeVarName   + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_blockZFP_originalDataCompression/";
            


    /*std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string originalSubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_originalDataCompression/";
            
    std::string mySubDir = originalSubDir + "rank_" + std::to_string(mpi_rank) + "/"; */

    std::vector<ScidxInterval<double>> allBigBlockIndices = loadBigBlockIndexFile(indexDir + "big_block_minmax");

    double globalMin = std::numeric_limits<double>::max();
    double globalMax = std::numeric_limits<double>::lowest();

    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }





    double error_bound = relative_error_bound * ( 0.597406 - 1.19061e-141 );

    


    std::cout << "Computed error_bound: " << error_bound << std::endl;
    


    std::vector<size_t> allGlobalSmallBlockIds(totalBlocksNumber);  
    for (size_t i = 0; i < totalBlocksNumber; ++i) {
        allGlobalSmallBlockIds[i] = i;
    }

    size_t blocksPerRank = (totalBlocksNumber + mpi_size - 1) / mpi_size;  // 【修改2】每个进程处理的块数
    size_t start = mpi_rank * blocksPerRank;
    size_t end = std::min(start + blocksPerRank, totalBlocksNumber);

    std::vector<size_t> myBlockIds(allGlobalSmallBlockIds.begin() + start, allGlobalSmallBlockIds.begin() + end);  // 【修改2】分配给当前 rank 的块



    auto afterPreProcessingTime = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> preProcessingTime = afterPreProcessingTime - totalStart;


    std::cout << "[Rank " << mpi_rank << "] [Time1]: preProcessingTime: " << preProcessingTime.count() << " seconds" << std::endl;

    //解压原数据小块中数据
    //std::vector<std::vector<double>> decompressedTargetOriginalBlockdata = batchDecompressBlocksAllRead(originalSubDir, allGlobalSmallBlockIds, smallBlockSize, totalBlocksNumber, error_bound);

    

    std::unordered_map<size_t, size_t> globalToLocal;
    for (size_t i = 0; i < myBlockIds.size(); ++i) {
        globalToLocal[myBlockIds[i]] = i;
    }
    size_t myLocalBlockCount = myBlockIds.size();


    /*std::vector<std::vector<double>> decompressedTargetOriginalBlockdata = batchDecompressBlocksAllRead(
        mySubDir, myBlockIds, smallBlockSize, myLocalBlockCount, error_bound, globalToLocal); */


    std::vector<std::vector<double>> decompressedTargetOriginalBlockdata = universalBatchDecompressBlocksAllRead(
            mySubDir,
            allGlobalSmallBlockIds,
            smallBlockSize,
            totalBlocksNumber,
            error_bound
        ); 



    std::cout << "Decompressed block count: " << decompressedTargetOriginalBlockdata.size() << std::endl;



    auto finishDecompreeAllOverlapedSamllBlocks = std::chrono::high_resolution_clock::now(); 

   

    /*std::vector<std::vector<double>> filtereddata  = rangeFilterData(decompressedTargetOriginalBlockdata, queryRange[0], queryRange[1]);

    std::cout << "filtered block size= " << filtereddata.size() << std::endl;




    auto finishQueryInAllOverlapedSmallBlocks= std::chrono::high_resolution_clock::now(); 

    size_t totalElements = 0;



    for (size_t i = 0; i < filtereddata.size(); ++i) {
        totalElements += filtereddata[i].size(); 
    }

    std::cerr << "Total number of filtered elements: " << totalElements << std::endl;


    //解压全部小块原数据时间
    //std::chrono::duration<double> decompreeAllOverlapedSamllBlocksTime = finishDecompreeAllOverlapedSamllBlocks - totalStart;
    
    //在解压后小块中查找时间
    std::chrono::duration<double> queryInAllOverlapedSmallBlocksTime = finishQueryInAllOverlapedSmallBlocks- finishDecompreeAllOverlapedSamllBlocks;
    


    
    std::cout << "[Rank " << mpi_rank << "] [Time4]:Query In All Overlapped Small Blocks Time: " << queryInAllOverlapedSmallBlocksTime.count() << " seconds" << std::endl;



    auto totalEnd = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> totalQueryTime = totalEnd - totalStart;
    std::cout << "[Rank " << mpi_rank << "] [Time All]:Total query time without Index: " << totalQueryTime.count() << " seconds" << std::endl;*/




    /*//计入iso surface
    RunAndSaveIsosurfaceMesh(
        decompressedTargetOriginalBlockdata,         //分块解压数据
        0.02,                                     //  isovalue值
        smallBlockShape,                                // block 大小
        stepDataShape,                           // 总volume 尺寸
        "isosurface_double_all_1231.vtk"                  // 输出ISO surface文件名
    );*/







    // ============================================================
    // 为每个step分别提取等值面
    // ============================================================

    std::cout << "\n========================================" << std::endl;
    std::cout << "Extracting isosurfaces for " << nSteps << " steps..." << std::endl;
    std::cout << "========================================\n" << std::endl;

   

    std::cout << "[Info] Blocks per step: " << smallBlocksPerStep << std::endl;
    std::cout << "[Info] Total blocks: " << decompressedTargetOriginalBlockdata.size() << std::endl;

    for (size_t step = 0; step < nSteps; ++step) {
        auto stepStart = std::chrono::high_resolution_clock::now();
        
        size_t currentStepNum = beginStepNum + step;
        size_t stepOffset = step * smallBlocksPerStep;
        
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Processing Step " << currentStepNum << std::endl;
        std::cout << "Block range: [" << stepOffset << ", " 
                << (stepOffset + smallBlocksPerStep - 1) << "]" << std::endl;
        
        // 提取当前step的块数据
        std::vector<std::vector<double>> stepBlocks;
        stepBlocks.reserve(smallBlocksPerStep);
        
        for (size_t i = 0; i < smallBlocksPerStep; ++i) {
            size_t globalBlockId = stepOffset + i;
            if (globalBlockId < decompressedTargetOriginalBlockdata.size()) {
                stepBlocks.push_back(decompressedTargetOriginalBlockdata[globalBlockId]);
            } else {
                std::cerr << "Error: Block " << globalBlockId << " out of range!" << std::endl;
                break;
            }
        }
        
        std::cout << "[Info] Extracted " << stepBlocks.size() << " blocks for Step " 
                << currentStepNum << std::endl;



        /*std::vector<double> volume = ReassembleBlocksToVolume(
            stepBlocks, smallBlockShape, stepDataShape);

        std::string vtiFile = "volume_step_" + 
            std::to_string(currentStepNum) + ".vti";

        SaveVolumeAsVTI(volume, stepDataShape, vtiFile, "B_y");*/





            
        // 生成输出文件名
        //std::string outFile = "isosurface_step_" + std::to_string(currentStepNum) + "_iso_-0.037620_compressed_noindex.vtk";
        /*std::string outFile = "/expanse/lustre/scratch/sdi/temp_project/iso_results/isosurface_step_" 
    + std::to_string(currentStepNum) 
    + "_iso_0.174875_compressed_noindex.vtk";*/

    std::string outFile = "/expanse/lustre/scratch/sdi/temp_project/iso_results/isosurface_step_" 
    + std::to_string(currentStepNum) 
    + "_iso_" + std::to_string(queryRange[0]) + "_compressed_noindex.vtk";


    // 构造 blockCountOnEachDim
std::vector<size_t> blockCountOnEachDim = {
    (stepDataShape[0] + smallBlockShape[0] - 1) / smallBlockShape[0],
    (stepDataShape[1] + smallBlockShape[1] - 1) / smallBlockShape[1],
    (stepDataShape[2] + smallBlockShape[2] - 1) / smallBlockShape[2]
};

    // 构造当前step的blockIds
    std::vector<size_t> stepBlockIds(smallBlocksPerStep);
    for (size_t i = 0; i < smallBlocksPerStep; ++i)
        stepBlockIds[i] = stepOffset + i;


    // ===== 新增：MC之前对解压数据做高斯平滑 =====
    auto smoothStart = std::chrono::high_resolution_clock::now();

    // 高斯平滑
    /*gaussianSmoothBlocks(stepBlocks, stepBlockIds, smallBlockShape,
                        blockCountOnEachDim, 1);*/

    auto smoothEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> smoothTime = smoothEnd - smoothStart;
    std::cerr << "[Time_smooth] Gaussian smoothing time: " << smoothTime.count() << " seconds" << std::endl;
    
        
        // 执行 Marching Cubes
        RunAndSaveIsosurfaceMesh(
            stepBlocks,
            queryRange[0]  ,
            smallBlockShape,
            stepDataShape,
            outFile
        );
        
        auto stepEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> stepTime = stepEnd - stepStart;
        
        std::cout << "✅ Step " << currentStepNum << " completed in " 
                << stepTime.count() << " seconds" << std::endl;
        std::cout << "   Output: " << outFile << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "All " << nSteps << " isosurfaces generated!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    

    return 0;
}

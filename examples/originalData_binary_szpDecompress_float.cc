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
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <zfp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
 

float calculateMaxError(const std::vector<float>& flatOriginal,
    const std::vector<std::vector<float>>& decompressedBlocks) {
// Flatten decompressedBlocks
std::vector<float> flatDecompressed;
for (const auto& block : decompressedBlocks) {
flatDecompressed.insert(flatDecompressed.end(), block.begin(), block.end());
}

// Check size
if (flatOriginal.size() != flatDecompressed.size()) {
throw std::runtime_error("Size mismatch between original and decompressed data.");
}

// Compute max absolute error
float maxError = 0.0f;
for (size_t i = 0; i < flatOriginal.size(); ++i) {
float error = std::abs(flatOriginal[i] - flatDecompressed[i]);
if (error > maxError) {
maxError = error;
}
}

return maxError;
}


// 计算 PSNR
float calculatePSNR(const std::vector<float>& flatOriginal,
        const std::vector<std::vector<float>>& decompressedBlocks) {
    // Flatten
    std::vector<float> flatDecompressed;
    for (const auto& block : decompressedBlocks) {
    flatDecompressed.insert(flatDecompressed.end(), block.begin(), block.end());
    }

    // Check size
    if (flatOriginal.size() != flatDecompressed.size()) {
    throw std::runtime_error("Size mismatch between original and decompressed data.");
    }

    // Compute MSE in double
    double mse = 0.0;
    for (size_t i = 0; i < flatOriginal.size(); ++i) {
    double diff = static_cast<double>(flatOriginal[i]) - static_cast<double>(flatDecompressed[i]);
    mse += diff * diff;
    }
    mse /= static_cast<double>(flatOriginal.size());

    // Get max value
    float max_val = *std::max_element(flatOriginal.begin(), flatOriginal.end());

    // Compute PSNR
    if (mse == 0.0) {
    return std::numeric_limits<float>::infinity(); // lossless
    }
    double psnr = 10.0 * std::log10((max_val * max_val) / mse);
    return static_cast<float>(psnr);
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

    std::cout << "[Debug] Number of blocks to decompress: " << blockIds.size() << std::endl;

    std::cout << "[Debug] subDir: " << subDir << std::endl;


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

    std::cout << "[Debug] after read all: " << std::endl;


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


    return decompressedOriginalresult;
}

// **读取二进制数据**
std::vector<float> readBinaryFile(const std::string& filename, size_t numElements) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Unable to open binary file " << filename << std::endl;
        return {};
    }

    std::vector<float> data(numElements);
    inFile.read(reinterpret_cast<char*>(data.data()), numElements * sizeof(float));
    inFile.close();

    return data;
}


std::vector<std::vector<float>> batchDecompressBlocksAllRead_ZFP(
    const std::string& subDir,
    const std::vector<size_t>& blockIds,
    size_t blockSize,
    size_t totalBlocks,
    double tolerance)
{
    std::string dataFile = (std::filesystem::path(subDir) / "compressed_data.bin").string();
    std::string sizeFile = (std::filesystem::path(subDir) / "compressed_sizes.bin").string();

    std::cerr << "[DEBUG] file name = " << sizeFile << std::endl;

    // 读取压缩大小
    std::ifstream sizeIn(sizeFile, std::ios::binary);
    size_t blockCount;
    sizeIn.read(reinterpret_cast<char*>(&blockCount), sizeof(size_t));
    std::cerr << "[DEBUG] blockCount in file = " << blockCount << ", totalBlocks passed in = " << totalBlocks << std::endl;

    std::vector<size_t> compressedSizes(blockCount);
    sizeIn.read(reinterpret_cast<char*>(compressedSizes.data()), blockCount * sizeof(size_t));
    sizeIn.close();

    // 计算偏移
    std::vector<size_t> offsets(blockCount);
    offsets[0] = 0;
    for (size_t i = 1; i < blockCount; ++i)
        offsets[i] = offsets[i - 1] + compressedSizes[i - 1];

    std::ifstream dataIn(dataFile, std::ios::binary);
    if (!dataIn) {
        std::cerr << "[ERROR] Cannot open compressed data file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<float>> decompressedBlocks;

    for (size_t id : blockIds) {
        if (id >= blockCount) {
            std::cerr << "[ERROR] Block ID " << id << " out of range." << std::endl;
            exit(EXIT_FAILURE);
        }

        size_t compSize = compressedSizes[id];
        std::vector<char> compBuffer(compSize);

        dataIn.seekg(offsets[id], std::ios::beg);
        dataIn.read(compBuffer.data(), compSize);

        zfp_type type = zfp_type_float;
        zfp_field* field = zfp_field_1d(nullptr, type, blockSize);
        zfp_stream* zfp = zfp_stream_open(NULL);
        zfp_stream_set_accuracy(zfp, tolerance);

        std::vector<float> output(blockSize);
        zfp_field_set_pointer(field, output.data());

        bitstream* stream = stream_open(compBuffer.data(), compSize);
        zfp_stream_set_bit_stream(zfp, stream);
        zfp_stream_rewind(zfp);

        size_t success = zfp_decompress(zfp, field);
        if (success == 0) {
            std::cerr << "[ERROR] ZFP decompression failed for block " << id << std::endl;
            exit(EXIT_FAILURE);
        }

        decompressedBlocks.push_back(std::move(output));

        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);
    }

    dataIn.close();
    return decompressedBlocks;
}



 


int main(int argc, char* argv[]) {


    std::string inputFileName, variableName;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0, bigBlocksPerStep = 1, smallBlocksPerBig = 1 ,  smallBlockSize = 1 , totalBlocksNumber = 1, smallBlocksPerStep = 1; 
    std::vector<float> queryRange;
    float relative_error_bound = 1E-3;
    size_t extraValue =0;
    std::string variableType;
    


    // **解析命令行参数**
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc) {
            inputFileName = argv[++i];
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
        }else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        } 
    }

    for (size_t d = 0; d < nDim; d++) {
        //总小块数量
        smallBlocksPerStep *= stepDataShape[d] / smallBlockShape[d];            
        // 小块大小
        smallBlockSize *= smallBlockShape[d];                             // 每个小块的数据量
    }

    size_t nSteps = 1;
    //小块总数
    totalBlocksNumber = nSteps * smallBlocksPerStep;

    
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    //std::string indexDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(extraValue) + "_" + std::to_string(extraValue) +  "_index/";

    //std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string indexDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(1) + "_" + std::to_string(1) + "_" + std::to_string(extraValue) + "_ZFP_index/";


 
    std::string subDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(1) + "_" + std::to_string(1) + "_" + std::to_string(extraValue) + "_fixlength_originalDataCompression/";


            

    std::vector<ScidxInterval<float>> allBigBlockIndices = loadBigBlockIndexFile(indexDir + "big_block_minmax");

    float globalMin = std::numeric_limits<float>::max();
    float globalMax = std::numeric_limits<float>::lowest();

    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }

    float error_bound = relative_error_bound * (globalMax - globalMin);
    std::cout << "Computed error_bound: " << error_bound << std::endl;
    


    std::vector<size_t> allGlobalSmallBlockIds(totalBlocksNumber);  
    for (size_t i = 0; i < totalBlocksNumber; ++i) {
        allGlobalSmallBlockIds[i] = i;
    }

   
    size_t myLocalBlockCount = allGlobalSmallBlockIds.size();


    std::vector<std::vector<float>> decompressedTargetOriginalBlockdata = batchDecompressBlocksAllRead_ZFP(
        subDir, allGlobalSmallBlockIds, smallBlockSize, myLocalBlockCount, error_bound);  



    std::cout << "Decompressed block count: " << decompressedTargetOriginalBlockdata.size() << std::endl;


    size_t totalElements = 1;
    for (size_t dim : stepDataShape) {
           totalElements *= dim;
    }

    // **读取二进制文件**
    std::vector<float> varData = readBinaryFile(inputFileName, totalElements);
    if (varData.empty()) {
         std::cerr << "Error: Failed to read binary file " << inputFileName << std::endl;
         return 1;
    }

    // 初始化 flatBlockData，减少后续插入扩容
    std::vector<float> flatBlockData;
    flatBlockData.reserve(totalElements);  // 预分配空间

    size_t total_blocks = totalBlocksNumber;

    std::vector<size_t> blockShape =  smallBlockShape;

    std::vector<size_t> blockCountOnEachDim(nDim);



    for (size_t i = 0; i < nDim; i++) {
        blockCountOnEachDim[i] = (stepDataShape[i] + smallBlockShape[i] - 1) / smallBlockShape[i];  // 向上取整，避免数据不整除导致丢失

    }

    // 块，和每块其中每个点
    std::vector<std::vector<float>> blocks(total_blocks, std::vector<float>());  // 初始化所有块

    // 遍历数据点，将其分配到相应的块
    for (size_t p = 0; p < totalElements; p++) {
        std::vector<size_t> elem_global_id = positionToIndices(p, stepDataShape);  // 计算全局索引坐标
        std::vector<size_t> block_global_id(nDim);
        
        for (size_t i = 0; i < nDim; i++) {
            block_global_id[i] = elem_global_id[i] / blockShape[i];  // 计算所属的块
        }

        size_t block_position = indicesToPosition(blockCountOnEachDim, block_global_id);  // 计算块的线性索引
        blocks[block_position].push_back(varData[p]);  // 将数据点加入对应块
    }

    // 依次收集所有块的数据到 flatBlockData
    for (size_t b = 0; b < total_blocks; b++) {
        flatBlockData.insert(flatBlockData.end(), blocks[b].begin(), blocks[b].end());
    }
    float maxError = calculateMaxError(flatBlockData, decompressedTargetOriginalBlockdata);
    std::cout << "Maximum absolute error: " << maxError << std::endl;
    
    
    
    float psnr = calculatePSNR(flatBlockData, decompressedTargetOriginalBlockdata);
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "PSNR: " << psnr << " dB" << std::endl;

    return 0;
}

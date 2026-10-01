#include <iostream>
#include <vector>
#include <cstdlib>
#include <adios2.h>
#include <cstddef>
#include <cstring>
#include <SZ3/api/sz.hpp> 
#include <stdexcept>
#include <cstdio>
#include <filesystem>  // 添加文件系统支持
#include <sys/stat.h>
#include <sys/types.h>
#include <cassert>
#include <set> 
#include <mpi.h> 
#include <zfp.h> 
#include <scidx_Huffman.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <filesystem>
#include <chrono>

void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
}

void createMoreDirectory(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec && !std::filesystem::is_directory(path)) {
        std::cerr << "Failed to create directory " << path << ": " << ec.message() << std::endl;
    }
}



// Function prototypes for encoding helper functions
unsigned int convertIntArray2ByteArray_fast_1b_args(const unsigned char* input, size_t inputSize, unsigned char* output);
unsigned int save_fixed_length_bits(const unsigned int* input, size_t inputSize, unsigned char* output, unsigned int bit_count);

unsigned int calculateBitCount(unsigned int maxDifference);

// Function to calculate the bit count required to represent the max difference
unsigned int calculateBitCount(unsigned int maxDifference) {
    if (maxDifference == 0) {
        return 1;  // At least 1 bit is needed to represent zero
    }
    return (unsigned int)(log2(maxDifference)) + 1;
}


std::vector<size_t> positionToIndices(size_t position, const std::vector<size_t>& shape) 
{
    std::vector<size_t> indices;
    size_t remainingPosition = position;

    for (auto dimensionSize = shape.rbegin(); dimensionSize != shape.rend(); ++dimensionSize) 
    {
        indices.insert(indices.begin(), remainingPosition % *dimensionSize);
        remainingPosition /= *dimensionSize;
    }

    return indices;
}

size_t indicesToPosition(const std::vector<size_t>& shape, const std::vector<size_t>& indices) 
{
    size_t position = indices[0];
    size_t multiplier = 1;

    for (size_t i = 1; i < shape.size(); ++i) 
    {
        multiplier *= shape[i - 1];
        position += indices[i] * multiplier;
    }

    return position;
}



void testSplitTime(const std::vector<double> &varData,
    const std::vector<size_t> &dataShape,
    const std::vector<size_t> &blockShape) {
    size_t nDim = dataShape.size();
    std::vector<size_t> blockCountOnEachDim(nDim);
    size_t total_blocks = 1;

    for (size_t i = 0; i < nDim; i++) {
    blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];
    total_blocks *= blockCountOnEachDim[i];
    }

    std::vector<std::vector<double>> blocks(total_blocks);
    std::vector<double> flatBlockData;
    flatBlockData.reserve(varData.size());

    auto splitStart = std::chrono::high_resolution_clock::now();

    // === 分块 ===
    for (size_t p = 0; p < varData.size(); p++) {
    std::vector<size_t> elem_global_id = positionToIndices(p, dataShape);
    std::vector<size_t> block_global_id(nDim);
    for (size_t i = 0; i < nDim; i++) {
    block_global_id[i] = elem_global_id[i] / blockShape[i];
    }
    size_t block_position = indicesToPosition(blockCountOnEachDim, block_global_id);
    blocks[block_position].push_back(varData[p]);
    }

    for (size_t b = 0; b < total_blocks; b++) {
    flatBlockData.insert(flatBlockData.end(), blocks[b].begin(), blocks[b].end());
    }

    auto splitEnd = std::chrono::high_resolution_clock::now();
    double splitTime = std::chrono::duration<double>(splitEnd - splitStart).count();

    std::cout << "[Split Time] Elements: " << varData.size()
    << ", Block Size: (" << blockShape[0] << "," << blockShape[1] << "," << blockShape[2] << ")"
    << ", Time: " << splitTime << " seconds" << std::endl;
}

void processDiffs(int *quant_inds, size_t pos, size_t numToCompress, 
                  unsigned char *signOutputBytes, unsigned char *otherOutputBytes,
                  size_t &signCompressedSize, size_t &otherCompressedSize) {
    //std::cout << "numToCompress: " << numToCompress << std::endl;
    if (numToCompress == 0) {
        signCompressedSize = 0;
        otherCompressedSize = 0;
        return;
    }

    unsigned char *temp_sign_arr = (unsigned char *)malloc(numToCompress );
    unsigned int *temp_predict_arr = (unsigned int *)malloc(numToCompress * sizeof(unsigned int));

    if (!temp_sign_arr || !temp_predict_arr) {
        std::cerr << "Memory allocation failed in processDiffs!" << std::endl;
        if (temp_sign_arr) free(temp_sign_arr);
        if (temp_predict_arr) free(temp_predict_arr);
        exit(1);
    }

    unsigned int max = 0;
    for (size_t i = 0; i < numToCompress; i++) {
        int current = quant_inds[pos + i];
        if (current >= 0) {
            temp_sign_arr[i] = 0;
            temp_predict_arr[i] = current;
        } else {
            temp_sign_arr[i] = 1;
            temp_predict_arr[i] = -current;
        }
        max = std::max(max, temp_predict_arr[i]);
    }

    //不会超过
    unsigned int bit_count = calculateBitCount(max);

    //temp_sign_arr转化成紧凑的8bits
    signCompressedSize = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, numToCompress, signOutputBytes);

    //一个bytes表示最大的bits数，用以表达的最大bits不能超过255，不会出问题
    otherOutputBytes[0] = static_cast<unsigned char>(bit_count);
    unsigned char *block_pointer = otherOutputBytes + 1;
    
    otherCompressedSize = save_fixed_length_bits(temp_predict_arr, numToCompress, block_pointer, bit_count) + 1;

    free(temp_sign_arr);
    free(temp_predict_arr);
}





unsigned int convertIntArray2ByteArray_fast_1b_args(const unsigned char* input, size_t inputSize, unsigned char* output) {
    size_t byteIndex = 0;
    unsigned char currentByte = 0;
    unsigned int bitsFilled = 0;

    for (size_t i = 0; i < inputSize; i++) {
        currentByte |= (input[i] & 1) << bitsFilled;
        bitsFilled++;

        if (bitsFilled == 8) {
            output[byteIndex++] = currentByte;
            currentByte = 0;
            bitsFilled = 0;
        }
    }

    if (bitsFilled > 0) {
        output[byteIndex++] = currentByte;
    }

    return byteIndex;
}



unsigned int save_fixed_length_bits(const unsigned int* input, size_t inputSize, unsigned char* output, unsigned int bit_count) {
    size_t byteIndex = 0;
    unsigned char currentByte = 0;
    unsigned int bitsFilled = 0;

    for (size_t i = 0; i < inputSize; i++) {
        unsigned int value = input[i] & ((1u << bit_count) - 1);

        size_t remainingBits = bit_count;
        while (remainingBits > 0) {

            // 转换类型以确保 std::min 的参数一致
            unsigned int bitsToWrite = std::min(static_cast<size_t>(8u - bitsFilled), remainingBits);


            currentByte |= (value & ((1u << bitsToWrite) - 1)) << bitsFilled;
            bitsFilled += bitsToWrite;
            remainingBits -= bitsToWrite;
            value >>= bitsToWrite;

            if (bitsFilled == 8) {
                output[byteIndex++] = currentByte;
                currentByte = 0;
                bitsFilled = 0;
            }
        }
    }

    if (bitsFilled > 0) {
        output[byteIndex++] = currentByte;
    }

    return byteIndex;
}



// 压缩函数，将 int 数组压缩为 10 bits 紧凑存储到 char 数组
void compressTo10Bits(const std::vector<int>& input, std::vector<char>& output) {
    size_t bitIndex = 0;        // 全局 bit 索引
    unsigned int bitBuffer = 0; // 缓存区，存储当前累积的位
    size_t bitsInBuffer = 0;   // 缓存区中有效位数

    for (int value : input) {
        // 检查是否在 10-bit 范围内
        if (value >= (1 << 10) || value < 0) {
            throw std::runtime_error("Value exceeds 10 bits or is negative.");
        }

        // 将当前值追加到缓存区
        bitBuffer |= (value << bitsInBuffer);
        bitsInBuffer += 10; // 累积 10 位

        // 将缓存区中的完整字节输出到结果数组
        while (bitsInBuffer >= 8) {
            output.push_back(static_cast<char>(bitBuffer & 0xFF)); // 提取低 8 位
            bitBuffer >>= 8;  // 缓存区右移 8 位
            bitsInBuffer -= 8; // 缓存区有效位减少 8 位
        }
    }

    // 如果最后还有未填满的位，将剩余部分存储为一个字节
    if (bitsInBuffer > 0) {
        output.push_back(static_cast<char>(bitBuffer & 0xFF));
    }
}


//将UnpredSize转化成6bits的紧凑型
void compressUnpredSizes(const std::vector<unsigned char>& original, std::vector<unsigned char>& compressed) {
    size_t bitIndex = 0;         // 当前写入的全局 bit 索引
    unsigned char currentByte = 0; // 当前正在构建的字节

    for (unsigned char value : original) {
        if (value >= 64) {
            throw std::runtime_error("Value exceeds 6 bits (must be < 64).");
        }

        size_t remainingBits = 6; // 每个值需要存储的 bit 数量
        while (remainingBits > 0) {
            size_t freeBitsInByte = 8 - (bitIndex % 8); // 当前字节剩余的位数
            size_t bitsToWrite = std::min(remainingBits, freeBitsInByte); // 本次写入的 bit 数量

            // 写入当前字节的空余位置
            currentByte |= (value & ((1 << bitsToWrite) - 1)) << (bitIndex % 8);

            bitIndex += bitsToWrite;        // 更新全局 bit 索引
            remainingBits -= bitsToWrite;  // 减少待写入的位数
            value >>= bitsToWrite;         // 移除已写入的位

            // 如果当前字节已满，写入到压缩结果中
            if (bitIndex % 8 == 0) {
                compressed.push_back(currentByte);
                currentByte = 0; // 重置字节
            }
        }
    }

    // 如果还有未填满的字节，写入到压缩结果中
    if (bitIndex % 8 != 0) {
        compressed.push_back(currentByte);
    }
}


/*inline std::pair<size_t, size_t> stmCompress_SZ3_default(
    double *input, size_t numElements, int blockSize,
    std::string subDir, double error_bound,
    double &compressTime, double &writeTime)
{
    std::string unpredDataFileName = (std::filesystem::path(subDir) / "fixed_unpredData.bin").string();
    std::string compressedFileName = (std::filesystem::path(subDir) / "fixed_compressed_data.bin").string();
    std::string HuffmanFileName = (std::filesystem::path(subDir) / "huffman.bin").string();

    std::vector<int> quant_inds(numElements);
    std::vector<size_t> metaSizes;
    std::vector<SZ3::uchar> metaDataBuffer;

    std::vector<size_t> compSizes;
    std::vector<SZ3::uchar> compDataBuffer;

    size_t pos = 0;
    size_t totalCompSize = 0;

    auto compressStart = std::chrono::high_resolution_clock::now();

    size_t bufferSize = 4 * sizeof(double) * blockSize;
    auto *buffer = new SZ3::uchar[bufferSize];

    while (pos < numElements) {
     
        size_t numToCompress = std::min(static_cast<size_t>(blockSize), numElements - pos);

        std::vector<double> data_cpy(&input[pos], &input[pos + numToCompress]);

        SZ3::Config conf(numToCompress);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;

        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);
        std::vector<int> quant_inds_block = decompose.compress(conf, data_cpy.data());

        std::copy(quant_inds_block.begin(), quant_inds_block.end(), quant_inds.begin() + pos);

        // 生成 metadata
        std::vector<SZ3::uchar> metadataBuffer(4 * sizeof(double) * blockSize);
        auto bufferp = metadataBuffer.data();
        decompose.save(bufferp);
        size_t metaSize = bufferp - metadataBuffer.data();

        metaSizes.push_back(metaSize);
        metaDataBuffer.insert(metaDataBuffer.end(), metadataBuffer.begin(), metadataBuffer.begin() + metaSize);

        pos += numToCompress;
    }

    // Huffman 编码和压缩
  
    auto *buffer_zstd = new SZ3::uchar[bufferSize];
    SZ3::uchar *buffer_pos = buffer;

    SZ3::HuffmanEncoder<int> huffman;
    huffman.preprocess_encode(quant_inds, 0);
    buffer_pos = buffer;
    huffman.save(buffer_pos);
    size_t huffmanSize = buffer_pos - buffer;

    pos = 0;
    while (pos < numElements) {
        
        size_t numToCompress = std::min(static_cast<size_t>(blockSize), numElements - pos);

        buffer_pos = buffer;
        huffman.encode(&quant_inds[pos], numToCompress, buffer_pos);

        size_t csize = SZ3::Lossless_zstd().compress(buffer, buffer_pos - buffer, buffer_zstd, bufferSize);
        compSizes.push_back(csize);
        compDataBuffer.insert(compDataBuffer.end(), buffer_zstd, buffer_zstd + csize);

        totalCompSize += csize;
        pos += numToCompress;
    }

    auto compressEnd = std::chrono::high_resolution_clock::now();
    compressTime = std::chrono::duration<double>(compressEnd - compressStart).count();

    auto writeStart = std::chrono::high_resolution_clock::now();

    // 拼接 metadata: [all_sizes][all_data]
    std::vector<SZ3::uchar> metaCombined;
    metaCombined.resize(metaSizes.size() * sizeof(size_t) + metaDataBuffer.size());
    std::memcpy(metaCombined.data(), metaSizes.data(), metaSizes.size() * sizeof(size_t));
    std::memcpy(metaCombined.data() + metaSizes.size() * sizeof(size_t), metaDataBuffer.data(), metaDataBuffer.size());
    SZ3::writefile(unpredDataFileName.c_str(), metaCombined.data(), metaCombined.size());

    // 拼接压缩数据: [all_sizes][all_data]
    std::vector<SZ3::uchar> compCombined;
    compCombined.resize(compSizes.size() * sizeof(size_t) + compDataBuffer.size());
    std::memcpy(compCombined.data(), compSizes.data(), compSizes.size() * sizeof(size_t));
    std::memcpy(compCombined.data() + compSizes.size() * sizeof(size_t), compDataBuffer.data(), compDataBuffer.size());
    SZ3::writefile(compressedFileName.c_str(), compCombined.data(), compCombined.size());


    // === 写 Huffman
    SZ3::writefile(HuffmanFileName.c_str(), buffer, huffmanSize);


    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<double>(writeEnd - writeStart).count();

    delete[] buffer;
    delete[] buffer_zstd;

    return std::make_pair(totalCompSize, compSizes.size());
}*/


inline std::pair<size_t, size_t> stmCompress_SZ3_default(
    double *input, size_t numElements, int blockSize,
    std::string subDir, double error_bound,
    double &compressTime, double &writeTime)
{
    std::string unpredDataFileName = (std::filesystem::path(subDir) / "fixed_unpredData.bin").string();
    std::string compressedFileName = (std::filesystem::path(subDir) / "fixed_compressed_data.bin").string();
    std::string HuffmanFileName = (std::filesystem::path(subDir) / "huffman.bin").string();

    std::vector<int> quant_inds;
    std::vector<size_t> metaSizes;
    std::vector<SZ3::uchar> metaDataBuffer;

    std::vector<size_t> compSizes;
    std::vector<SZ3::uchar> compDataBuffer;

    size_t pos = 0;
    size_t totalCompSize = 0;

    auto compressStart = std::chrono::high_resolution_clock::now();

    size_t bufferSize = 2 * sizeof(double) * blockSize;
    std::vector<SZ3::uchar> buffer(bufferSize);
    std::vector<SZ3::uchar> buffer_zstd(bufferSize);

    while (pos < numElements) {
        size_t numToCompress = std::min(static_cast<size_t>(blockSize), numElements - pos);

        std::vector<double> data_cpy(&input[pos], &input[pos + numToCompress]);

        SZ3::Config conf(numToCompress);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;

        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);
        std::vector<int> quant_inds_block = decompose.compress(conf, data_cpy.data());

        assert(quant_inds_block.size() == numToCompress);
        quant_inds.insert(quant_inds.end(), quant_inds_block.begin(), quant_inds_block.end());

        std::vector<SZ3::uchar> metadataBuffer(2 * sizeof(double) * blockSize);
        auto bufferp = metadataBuffer.data();
        decompose.save(bufferp);
        size_t metaSize = bufferp - metadataBuffer.data();

        metaSizes.push_back(metaSize);
        metaDataBuffer.insert(metaDataBuffer.end(), metadataBuffer.begin(), metadataBuffer.begin() + metaSize);

        pos += numToCompress;
    }

    SZ3::HuffmanEncoder<int> huffman;
    huffman.preprocess_encode(quant_inds, 0);

    

    std::cout << "[DEBUG] quant_inds size: " << quant_inds.size() << std::endl;


    SZ3::uchar *buffer_pos = buffer.data();
    huffman.save(buffer_pos);
    size_t huffmanSize = buffer_pos - buffer.data();

    pos = 0;
    bool firstIteration = true;

    while (pos < quant_inds.size()) {
        size_t numToCompress = std::min(static_cast<size_t>(blockSize), quant_inds.size() - pos);

        buffer_pos = buffer.data();


        huffman.encode(&quant_inds[pos], numToCompress, buffer_pos);
        assert((size_t)(buffer_pos - buffer.data()) <= bufferSize);

        size_t dataSizeBeforeZstd = buffer_pos - buffer.data(); // 压缩前数据大小
        size_t csize = SZ3::Lossless_zstd().compress(buffer.data(), dataSizeBeforeZstd, buffer_zstd.data(), bufferSize);

        if (firstIteration) { // 只在第一次循环时打印
            std::cout << "[DEBUG] Huffman encode count = " << numToCompress << std::endl;

            std::cout << "[DEBUG] Before Zstd: " << dataSizeBeforeZstd << " bytes" << std::endl;
            std::cout << "[DEBUG] After Zstd: " << csize << " bytes" << std::endl;
            firstIteration = false; // 标记为已打印
        }

        //size_t csize = SZ3::Lossless_zstd().compress(buffer.data(), buffer_pos - buffer.data(), buffer_zstd.data(), bufferSize);
        compSizes.push_back(csize);
        compDataBuffer.insert(compDataBuffer.end(), buffer_zstd.begin(), buffer_zstd.begin() + csize);

        totalCompSize += csize;
        pos += numToCompress;
    }

    auto compressEnd = std::chrono::high_resolution_clock::now();
    compressTime = std::chrono::duration<double>(compressEnd - compressStart).count();

    auto writeStart = std::chrono::high_resolution_clock::now();

    std::vector<SZ3::uchar> metaCombined(metaSizes.size() * sizeof(size_t) + metaDataBuffer.size());
    std::memcpy(metaCombined.data(), metaSizes.data(), metaSizes.size() * sizeof(size_t));
    std::memcpy(metaCombined.data() + metaSizes.size() * sizeof(size_t), metaDataBuffer.data(), metaDataBuffer.size());
    SZ3::writefile(unpredDataFileName.c_str(), metaCombined.data(), metaCombined.size());

    std::vector<SZ3::uchar> compCombined(compSizes.size() * sizeof(size_t) + compDataBuffer.size());
    std::memcpy(compCombined.data(), compSizes.data(), compSizes.size() * sizeof(size_t));
    std::memcpy(compCombined.data() + compSizes.size() * sizeof(size_t), compDataBuffer.data(), compDataBuffer.size());
    SZ3::writefile(compressedFileName.c_str(), compCombined.data(), compCombined.size());

    SZ3::writefile(HuffmanFileName.c_str(), buffer.data(), huffmanSize);

    std::cout << "[DEBUG] Huffman tree saved, size: " << huffmanSize << std::endl;

    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<double>(writeEnd - writeStart).count();

    return std::make_pair(totalCompSize, compSizes.size());
}



inline std::pair<size_t, size_t> stmCompress_SZ3_newHuffman(
    double *input, size_t numElements, int blockSize,
    std::string subDir, double error_bound,
    double &compressTime, double &writeTime)
{
    std::string unpredDataFileName = (std::filesystem::path(subDir) / "fixed_unpredData.bin").string();
    std::string compressedFileName = (std::filesystem::path(subDir) / "fixed_compressed_data.bin").string();
    std::string HuffmanFileName = (std::filesystem::path(subDir) / "huffman.bin").string();

    std::vector<int> quant_inds;
    std::vector<size_t> metaSizes;
    std::vector<SZ3::uchar> metaDataBuffer;

    std::vector<size_t> compSizes;
    std::vector<SZ3::uchar> compDataBuffer;

    size_t pos = 0;
    size_t totalCompSize = 0;

    auto compressStart = std::chrono::high_resolution_clock::now();

    size_t bufferSize = 40 * sizeof(double) * blockSize;
    std::vector<SZ3::uchar> buffer(bufferSize);
    std::vector<SZ3::uchar> buffer_zstd(bufferSize);

    while (pos < numElements) {
        size_t numToCompress = std::min(static_cast<size_t>(blockSize), numElements - pos);

        std::vector<double> data_cpy(&input[pos], &input[pos + numToCompress]);

        SZ3::Config conf(numToCompress);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;

        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);
        std::vector<int> quant_inds_block = decompose.compress(conf, data_cpy.data());

        assert(quant_inds_block.size() == numToCompress);
        quant_inds.insert(quant_inds.end(), quant_inds_block.begin(), quant_inds_block.end());

        std::vector<SZ3::uchar> metadataBuffer(40 * sizeof(double) * blockSize);
        auto bufferp = metadataBuffer.data();
        decompose.save(bufferp);
        size_t metaSize = bufferp - metadataBuffer.data();

        metaSizes.push_back(metaSize);
        metaDataBuffer.insert(metaDataBuffer.end(), metadataBuffer.begin(), metadataBuffer.begin() + metaSize);

        pos += numToCompress;
    }

    int stateNum = 2 * 16384;
    scidx::HuffmanTree *huffmanTreeFull = scidx::createHuffmanTree(stateNum);

    unsigned char *huffmanOut = nullptr;
    size_t huffmanOutSize = 0;
    scidx::init_and_serialize_Huffmantree(huffmanTreeFull, quant_inds.data(), quant_inds.size(), &huffmanOut, &huffmanOutSize);

    if (huffmanOut == nullptr || huffmanOutSize == 0) {
        std::cerr << "Error: huffmanOut is null or huffmanOutSize is zero." << std::endl;
        free(huffmanOut);
        return std::make_pair(0, 0);
    }

    std::ofstream outFile(HuffmanFileName.c_str(), std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file " << HuffmanFileName << " for writing." << std::endl;
        free(huffmanOut);
        return std::make_pair(0, 0);
    }
    outFile.write(reinterpret_cast<char*>(huffmanOut), huffmanOutSize);
    outFile.close();
    free(huffmanOut);

    pos = 0;
    bool firstIteration = true;

    while (pos < quant_inds.size()) {
        size_t numToCompress = std::min(static_cast<size_t>(blockSize), quant_inds.size() - pos);

        std::vector<unsigned char> singleEncodeOut(numToCompress * 4);
        size_t singleEncodeOutsize = 0;

        scidx::encode(huffmanTreeFull, &quant_inds[pos], numToCompress, singleEncodeOut.data(), &singleEncodeOutsize);

        size_t csize = SZ3::Lossless_zstd().compress(
            singleEncodeOut.data(), singleEncodeOutsize,
            buffer_zstd.data(), bufferSize);

        if (firstIteration) {
            std::cout << "[DEBUG] Huffman encode count = " << numToCompress << std::endl;
            size_t dataSizeBeforeZstd = singleEncodeOutsize;
            std::cout << "[DEBUG] Before Zstd: " << dataSizeBeforeZstd << " bytes" << std::endl;
            std::cout << "[DEBUG] After Zstd: " << csize << " bytes" << std::endl;
            firstIteration = false;
        }

        compSizes.push_back(csize);
        compDataBuffer.insert(compDataBuffer.end(), buffer_zstd.begin(), buffer_zstd.begin() + csize);
        totalCompSize += csize;
        pos += numToCompress;
    }

    auto compressEnd = std::chrono::high_resolution_clock::now();
    compressTime = std::chrono::duration<double>(compressEnd - compressStart).count();

    auto writeStart = std::chrono::high_resolution_clock::now();

    std::vector<SZ3::uchar> metaCombined(metaSizes.size() * sizeof(size_t) + metaDataBuffer.size());
    std::memcpy(metaCombined.data(), metaSizes.data(), metaSizes.size() * sizeof(size_t));
    std::memcpy(metaCombined.data() + metaSizes.size() * sizeof(size_t), metaDataBuffer.data(), metaDataBuffer.size());
    SZ3::writefile(unpredDataFileName.c_str(), metaCombined.data(), metaCombined.size());

    std::vector<SZ3::uchar> compCombined(compSizes.size() * sizeof(size_t) + compDataBuffer.size());
    std::memcpy(compCombined.data(), compSizes.data(), compSizes.size() * sizeof(size_t));
    std::memcpy(compCombined.data() + compSizes.size() * sizeof(size_t), compDataBuffer.data(), compDataBuffer.size());
    SZ3::writefile(compressedFileName.c_str(), compCombined.data(), compCombined.size());

    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<double>(writeEnd - writeStart).count();

    return std::make_pair(totalCompSize, compSizes.size());
}


inline std::pair<size_t, size_t> stmCompress_SZ3_whole(
    double *input, size_t numElements, int blockSize,
    std::string subDir, double error_bound,
    double &compressTime, double &writeTime)
{
    std::string compressedFileName = (std::filesystem::path(subDir) / "sz3_whole_compressed.bin").string();

    SZ3::Config conf(numElements);
    conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = error_bound;

    auto compressStart = std::chrono::high_resolution_clock::now();

    size_t cmpSize = 0;
    char* cmpData = SZ_compress(conf, input, cmpSize);

    auto compressEnd = std::chrono::high_resolution_clock::now();
    compressTime = std::chrono::duration<double>(compressEnd - compressStart).count();

    auto writeStart = std::chrono::high_resolution_clock::now();
    SZ3::writefile(compressedFileName.c_str(), (SZ3::uchar*)cmpData, cmpSize);
    delete[] cmpData;
    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<double>(writeEnd - writeStart).count();

    return std::make_pair(cmpSize, size_t(1));
}

inline std::pair<size_t, size_t> stmCompress_SZ3_blockwise(
    double *input, size_t numElements, int blockSize,
    std::string subDir, double error_bound,
    double &compressTime, double &writeTime)
{
    std::string compressedFileName = (std::filesystem::path(subDir) / "sz3_blockwise_compressed.bin").string();

    // 从 numElements 推导立方体边长（假设是 cubic volume）
    size_t dim = (size_t)std::round(std::cbrt((double)numElements));

    // 3D config，让SZ3内部做blockwise
    SZ3::Config conf(dim, dim, dim);
    conf.blockSize = blockSize;
    conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = error_bound;
    // conf.blockSize 是每个维度的分块大小，默认是6，可按需调整
    // conf.blockSize = 6;

    auto compressStart = std::chrono::high_resolution_clock::now();

    size_t cmpSize = 0;
    char* cmpData = SZ_compress(conf, input, cmpSize);

    auto compressEnd = std::chrono::high_resolution_clock::now();
    compressTime = std::chrono::duration<double>(compressEnd - compressStart).count();

    auto writeStart = std::chrono::high_resolution_clock::now();
    SZ3::writefile(compressedFileName.c_str(), (SZ3::uchar*)cmpData, cmpSize);
    delete[] cmpData;
    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<double>(writeEnd - writeStart).count();

    return std::make_pair(cmpSize, size_t(1));
}



inline std::pair<size_t, size_t> stmCompress_ZFP_default(
    double* input,
    size_t numElements,
    int blockSize,
    const std::string& subDir,
    double tolerance,
    double& compressTime,
    double& writeTime)
{
    std::string dataFile = (std::filesystem::path(subDir) / "compressed_data.bin").string();
    std::string sizeFile = (std::filesystem::path(subDir) / "compressed_sizes.bin").string();

    std::ofstream out(dataFile, std::ios::binary);
    std::vector<size_t> compressedSizes;

    size_t pos = 0;
    size_t totalSize = 0;
    size_t blockCount = 0;

    auto compressStart = std::chrono::high_resolution_clock::now();

    while (pos < numElements) {
        size_t numToCompress = std::min(static_cast<size_t>(blockSize), numElements - pos);

        zfp_type type = zfp_type_double;
        zfp_field* field = zfp_field_1d(input + pos, type, numToCompress);

        zfp_stream* zfp = zfp_stream_open(NULL);
        zfp_stream_set_accuracy(zfp, tolerance);

        size_t bufsize = zfp_stream_maximum_size(zfp, field);
        void* buffer = malloc(bufsize);
        if (!buffer) {
            std::cerr << "Failed to allocate compression buffer" << std::endl;
            exit(EXIT_FAILURE);
        }

        bitstream* stream = stream_open(buffer, bufsize);
        zfp_stream_set_bit_stream(zfp, stream);
        zfp_stream_rewind(zfp);

        size_t zfpsize = zfp_compress(zfp, field);
        if (zfpsize == 0) {
            std::cerr << "ZFP compression failed at block starting at position " << pos << std::endl;
            free(buffer);
            zfp_field_free(field);
            zfp_stream_close(zfp);
            stream_close(stream);
            exit(EXIT_FAILURE);
        }

        out.write(reinterpret_cast<const char*>(buffer), zfpsize);
        compressedSizes.push_back(zfpsize);
        totalSize += zfpsize;
        blockCount++;

        free(buffer);
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);

        pos += numToCompress;
    }

    auto compressEnd = std::chrono::high_resolution_clock::now();
    compressTime = std::chrono::duration<double>(compressEnd - compressStart).count();

    auto writeStart = std::chrono::high_resolution_clock::now();

    std::ofstream sizeOut(sizeFile, std::ios::binary);
    sizeOut.write(reinterpret_cast<const char*>(&blockCount), sizeof(size_t));
    sizeOut.write(reinterpret_cast<const char*>(compressedSizes.data()), blockCount * sizeof(size_t));
    sizeOut.close();
    out.close();

    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<double>(writeEnd - writeStart).count();

    return {totalSize, blockCount};
}








inline std::pair<size_t, size_t> stmCompress_SZ3(double *input, size_t numElements, int blockSize, std::string subDir, double error_bound, double &compressTime, double &writeTime) 
{
    // 创建文件名包含 step 范围
    std::string unpredDataFileName = (std::filesystem::path(subDir) / "fixed_unpredData.bin").string();
    //std::string firstValueFileName = (std::filesystem::path(subDir) / "fixed_firstValue.bin").string();
    std::string compressedFileName = (std::filesystem::path(subDir) / "fixed_compressed_data.bin").string();
    std::string signFileName = (std::filesystem::path(subDir) / "fixed_sign_data.bin").string();


    std::vector<int> quant_inds(numElements);  // 存储量化索引
    std::size_t pos = 0;
    std::size_t csize = 0;                     // 每个块的压缩大小
    std::size_t totalCompSize = 0;
    std::size_t totalSignSize = 0;


    // 创建 vector 来保存所有块的元数据
    std::vector<SZ3::uchar> allUnpredSizeBuffers; // 用单个 vector 保存所有的 unpredSize
    std::vector<SZ3::uchar> allUnpredDataBuffers; // 保持 unpredData 

    // 量化数据
    int radius = 512;


    auto compressStart = std::chrono::high_resolution_clock::now(); 

    while (pos < numElements) {
        size_t numToCompress = blockSize;
        if (numToCompress + pos > numElements) numToCompress = numElements - pos;

        std::vector<double> data_cpy(&input[pos], &input[pos + numToCompress]);

        // 配置 SZ3 参数
        SZ3::Config conf(numToCompress);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;

        // 执行压缩
        std::vector<int> quant_inds_block;
        auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);
        quant_inds_block = decompose.compress(conf, data_cpy.data());

        // 对量化索引值减去偏移
        std::transform(
            quant_inds_block.begin(), quant_inds_block.end(), quant_inds_block.begin(),
            [radius](int quant_index) { return quant_index - radius; }
        );

        //现在每个块再填充到quant_inds中
        std::copy(quant_inds_block.begin(), quant_inds_block.end(), quant_inds.begin() + pos);

        // 创建缓冲区以保存元数据
        size_t metadataSize = 2 * sizeof(double) * blockSize;
        std::vector<SZ3::uchar> metadataBuffer(metadataSize);

        // 保存元数据到缓冲区
        auto bufferp = metadataBuffer.data();
        decompose.save(bufferp);

        // 提取 unpred.size() 和 unpred.data()
        const SZ3::uchar *tempBuffer = metadataBuffer.data();

        // 跳过元数据中的前几个字段
        tempBuffer += 1; // 跳过第一个字节（标志位）
        tempBuffer += sizeof(double); // 跳过 error_bound (double 类型)
        tempBuffer += sizeof(int); // 跳过 radius (int 类型)

        size_t unpredSize = *reinterpret_cast<const size_t *>(tempBuffer);
        tempBuffer += sizeof(size_t);
        const double *unpredData = reinterpret_cast<const double *>(tempBuffer);

        /*// 确保 unpredSize 小于 64
        if (unpredSize >= 64) {
            std::cerr << "Error: unpredSize exceeds expected limit of 64." << std::endl;
            break;
        }*/

        if (unpredSize >= 256) {
            std::cerr << "Error: unpredSize exceeds expected limit of 256." << std::endl;
            break;
        }

        // 将 unpredSize 转为 unsigned char 保存
        allUnpredSizeBuffers.push_back(static_cast<SZ3::uchar>(unpredSize));

        // 将当前块的数据追加到 allUnpredDataBuffers
        size_t dataSize = unpredSize * sizeof(double);
        allUnpredDataBuffers.insert(allUnpredDataBuffers.end(),
                                    reinterpret_cast<const SZ3::uchar *>(unpredData),
                                    reinterpret_cast<const SZ3::uchar *>(unpredData) + dataSize);


        
        pos += numToCompress;
    }


    //metadata处理

    /*//unpredSize的处理
    std::vector<SZ3::uchar> compressedUnpredSize;
    compressUnpredSizes(allUnpredSizeBuffers, compressedUnpredSize);*/

    allUnpredDataBuffers.insert(
        allUnpredDataBuffers.begin(),  // 插入位置：起始位置
        allUnpredSizeBuffers.begin(), // 插入数据的起始迭代器
        allUnpredSizeBuffers.end()    // 插入数据的结束迭代器
    );
    

    pos = 0;
   
    std::vector<unsigned char> organizedCompressedData;
    std::vector<unsigned char> compressedSign;
    std::vector<int> firstValues;


    // 用于存储 bitsCounts
    std::vector<unsigned char> bitsCounts;
    // 用于存储真实数据大小
    std::vector<unsigned char> compressedDataSizes;
    // 用于存储真实数据
    std::vector<unsigned char> realCompressedData;


    // 逐块编码和压缩
    while (pos < numElements) {
        size_t numToCompress = blockSize;
        if (numToCompress + pos > numElements) numToCompress = numElements - pos;

        /*// 取出第一个值
        int firstValue = quant_inds[pos];
        firstValues.push_back(firstValue); // 将第一个值存入容器*/

        // 分配缓冲区
        size_t maxSignBufferSize = numToCompress ; // 符号数组的最大字节数
        size_t maxOtherBufferSize = numToCompress * 8; // 压缩后数据的最大字节数

        unsigned char *signOutputBytes = (unsigned char *)malloc(maxSignBufferSize);
        unsigned char *otherOutputBytes = (unsigned char *)malloc(maxOtherBufferSize);

        size_t signCompressedSize = 0;
        size_t otherCompressedSize = 0;

    
        //fixlength encoding
        //从第一个值开始处理
        processDiffs(quant_inds.data(), pos, numToCompress, signOutputBytes, otherOutputBytes, signCompressedSize, otherCompressedSize);


        //otherOutputBytes第 1 个字节存储编码bits数
        //后续存储fixlenghth encoding的数据 
        // 提取 bitsCount
        unsigned char bitsCount = otherOutputBytes[0];
        bitsCounts.push_back(bitsCount);

        //假设每个block压缩后的大小用8个bits表示（255），压缩后最大可达255bytes。若分块大小为4^3，255/（64-1）=4.04
        //则每个bin用小于4个bytes可表示的话，用一个bytes表示压缩后的大小则不会超过
        //而4bytes能表达的大小4,294,967,295。bin的最大值一定小于这个数
        //因此用一个bytes存储一个block压缩后的大小一定不会超过

        // 提取真实压缩后数据大小（63个数据，且不包含bitsCounts的一个bytes）
        //真实的fixlength压缩后数据的（字节数）
        unsigned char realSize = static_cast<unsigned char>(otherCompressedSize - 1);
        compressedDataSizes.push_back(realSize);

        // 提取真实数据
        //最后一个字节不满，在大端
        realCompressedData.insert(realCompressedData.end(), otherOutputBytes + 1, otherOutputBytes + otherCompressedSize);

        //每一个块有63个bin,有63个sign,每一个块用8个bytes来表示sign
        //compressedSign是每8个bytes用来表示一个block的全部sign，依次存储下去
        compressedSign.insert(compressedSign.end(), signOutputBytes, signOutputBytes + signCompressedSize);

        pos += numToCompress;

        //compressedSizeWithAll是每个块真实的压缩后的数据bytes和一个bytes表示bitscounts,一个bits表示压缩后的大小
        
        //size_t compressedSizeWithAll = otherCompressedSize + 1;
        
        totalCompSize += otherCompressedSize;
        totalSignSize += signCompressedSize;
      
    }

    // 将 bitsCounts 依次加入 organizedCompressedData
    //一个unchar存储一个block用来编码的bits数
    organizedCompressedData.insert(organizedCompressedData.end(), bitsCounts.begin(), bitsCounts.end());

    // 将 dataSizes 依次加入 organizedCompressedData
    //一个unchar存储一个block编码后数据的字节数
    organizedCompressedData.insert(organizedCompressedData.end(), compressedDataSizes.begin(), compressedDataSizes.end());

    // 将真实数据依次加入 organizedCompressedData
    //以unchar的形式紧凑存储编码后的数据
    organizedCompressedData.insert(organizedCompressedData.end(), realCompressedData.begin(), realCompressedData.end());


    /*//如果radius是512，可以10bits来表示第一个值
    std::vector<char> compressedFirstData;              // 用于存储压缩后的数据
    compressTo10Bits(firstValues, compressedFirstData);

    //10bits表示一个firstdata
    SZ3::writefile(firstValueFileName.c_str(), compressedFirstData.data(), compressedFirstData.size());*/

    auto compressEnd = std::chrono::high_resolution_clock::now(); 

    compressTime = std::chrono::duration<double>(compressEnd - compressStart).count(); 

    
    
    auto writeStart = std::chrono::high_resolution_clock::now();
    /*//每6个bits表示一个block的unpredictedsize，总共用前，block个数*6/8（向上取整）个bytes表示了大小*/

    //前block个数个uchar记录unpredictedsize
    //后面依次存储double类型的unpredicted data（个数根据unpredSize确定
    SZ3::writefile(unpredDataFileName.c_str(), allUnpredDataBuffers.data(), allUnpredDataBuffers.size());

    // 使用 SZ3::writefile 将 compressedData 写入文件
    //bitsCounts（块的个数个，每个1bytes）+压缩后数据大小（块的个数个，每个1bytes）+压缩后数据（块的个数个，大小根据前面的大小确定）
    SZ3::writefile(compressedFileName.c_str(), organizedCompressedData.data(), organizedCompressedData.size());

    // 每8个bytes用来表示一个block的全部sign，依次存储下去
    SZ3::writefile(signFileName.c_str(), compressedSign.data(), compressedSign.size());

    auto writeEnd = std::chrono::high_resolution_clock::now();  
    writeTime = std::chrono::duration<double>(writeEnd - writeStart).count(); 

   
    std::cout << "[Time] Data Compression Time: " << compressTime << " seconds" << std::endl;
    std::cout << "[Time] File Write Time: " << writeTime << " seconds" << std::endl;
    
    return {totalCompSize, totalSignSize};
}



// 通用分块压缩代码，支持 4³, 16³, 64³, 256³, 1024³



// 配置结构，根据分块大小自动计算参数
struct CompressionConfig {
    size_t blockSize;                    // 分块大小
    size_t unpredSizeBits;              // unpredSize需要的位数
    size_t dataSizeBytes;               // 压缩数据大小字段需要的字节数
    size_t maxUnpredSize;               // unpredSize的最大值
    size_t maxDataSize;                 // 压缩数据大小的最大值
    size_t signBytesPerBlock;           // 每个块的sign数据字节数
    
    CompressionConfig(size_t blockSize) : blockSize(blockSize) {
        // 根据分块大小计算所需参数
        calculateParameters();
    }
    
private:
    void calculateParameters() {
        // unpredSize 位数：能容纳 blockSize-1 的最小位数
        unpredSizeBits = static_cast<size_t>(std::ceil(std::log2(blockSize)));
        //unpredSizeBits = static_cast<size_t>(std::ceil(std::log2(blockSize + 1)));
        maxUnpredSize = (1ULL << unpredSizeBits) - 1;
        
        // sign 数据字节数：向上取整到字节
        signBytesPerBlock = (blockSize - 1 + 7) / 8; // blockSize-1 个sign位
        
        // 估算压缩后数据大小需要的字节数
        // 保守估算：每个元素最多占用4字节
        //因为quantbinCnt = 1024，实际量化索引范围在 [-512, 512]，去掉符号[0, 512],需要10个bits来表示压缩后的数据大小
        //blockSize*10/8个bytes来表示每个块压缩后需要的bytes数，需要存储的是blockSize*10/8这个值
        //考量blockSize*10/8这个值需要几个bytes来存储
        //4*4*4大小，则是64*10/8=80bytes,80用一个byte来存储就足够
        /*size_t estimatedMaxCompressedSize = blockSize * 4;
        if (estimatedMaxCompressedSize <= 255) {
            dataSizeBytes = 1;
            maxDataSize = 255;
        } else if (estimatedMaxCompressedSize <= 65535) {
            dataSizeBytes = 2;
            maxDataSize = 65535;
        } else if (estimatedMaxCompressedSize <= 16777215) {
            dataSizeBytes = 3;
            maxDataSize = 16777215;
        } else {
            dataSizeBytes = 4;
            maxDataSize = 4294967295ULL;
        }*/
        dataSizeBytes = 1;
        maxDataSize = 255;
        std::cout << "Block Configuration:" << std::endl;
        std::cout << "  Block Size: " << blockSize << std::endl;
        std::cout << "  UnpredSize Bits: " << unpredSizeBits << std::endl;
        std::cout << "  DataSize Bytes: " << dataSizeBytes << std::endl;
        std::cout << "  Sign Bytes Per Block: " << signBytesPerBlock << std::endl;
    }
};

// 通用位压缩函数：将数值按指定位数紧凑存储
class UniversalBitPacker {
public:
    // 压缩：将数值数组按指定位数紧凑存储
    static void packBits(const std::vector<uint64_t>& input, 
                        std::vector<uint8_t>& output, 
                        size_t bitsPerValue) {
        if (input.empty()) return;
        
        output.clear();
        size_t bitIndex = 0;
        uint64_t bitBuffer = 0;
        size_t bitsInBuffer = 0;
        
        uint64_t mask = (1ULL << bitsPerValue) - 1;
        
        for (uint64_t value : input) {
            if (value > mask) {
                throw std::runtime_error("Value exceeds specified bit width");
            }
            
            // 将值加入缓冲区
            bitBuffer |= (value << bitsInBuffer);
            bitsInBuffer += bitsPerValue;
            
            // 输出完整的字节
            while (bitsInBuffer >= 8) {
                output.push_back(static_cast<uint8_t>(bitBuffer & 0xFF));
                bitBuffer >>= 8;
                bitsInBuffer -= 8;
            }
        }
        
        // 输出剩余位
        if (bitsInBuffer > 0) {
            output.push_back(static_cast<uint8_t>(bitBuffer & 0xFF));
        }
    }
    
    // 解压缩：从紧凑存储中恢复数值数组
    static void unpackBits(const std::vector<uint8_t>& input, 
                          std::vector<uint64_t>& output, 
                          size_t bitsPerValue, 
                          size_t numValues) {
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
            
            output.push_back(value & mask);
        }
    }
};

// 通用多字节数值存储
class UniversalValueWriter {
public:
    // 写入指定字节数的数值（小端序）
    static void writeValue(std::vector<uint8_t>& output, uint64_t value, size_t numBytes) {
        if (numBytes > 8) {
            throw std::runtime_error("Cannot write more than 8 bytes");
        }
        
        for (size_t i = 0; i < numBytes; i++) {
            output.push_back(static_cast<uint8_t>(value & 0xFF));
            value >>= 8;
        }
    }
    
    // 读取指定字节数的数值（小端序）
    static uint64_t readValue(const uint8_t* data, size_t numBytes) {
        if (numBytes > 8) {
            throw std::runtime_error("Cannot read more than 8 bytes");
        }
        
        uint64_t value = 0;
        for (size_t i = 0; i < numBytes; i++) {
            value |= (static_cast<uint64_t>(data[i]) << (i * 8));
        }
        return value;
    }
};

// 通用差分处理函数
void processQuantizedDifferences(const int* quantIndices, size_t pos, size_t numElements,
                               uint8_t* signOutputBytes, uint8_t* otherOutputBytes,
                               size_t& signCompressedSize, size_t& otherCompressedSize,
                               const CompressionConfig& config) {
    if (numElements == 0) {
        signCompressedSize = 0;
        otherCompressedSize = 0;
        return;
    }

    // 分配临时数组
    std::vector<uint8_t> tempSignArray(numElements);
    std::vector<uint32_t> tempValueArray(numElements);

    uint32_t maxValue = 0;
    for (size_t i = 0; i < numElements; i++) {
        int current = quantIndices[pos + i];
        if (current >= 0) {
            tempSignArray[i] = 0;
            tempValueArray[i] = static_cast<uint32_t>(current);
        } else {
            tempSignArray[i] = 1;
            tempValueArray[i] = static_cast<uint32_t>(-current);
        }
        maxValue = std::max(maxValue, tempValueArray[i]);
    }

    // 计算需要的位数
    uint32_t bitsNeeded = 0;
    if (maxValue > 0) {
        bitsNeeded = static_cast<uint32_t>(std::ceil(std::log2(maxValue + 1)));
    }

    // 压缩符号数组
    signCompressedSize = 0;
    uint8_t currentByte = 0;
    uint32_t bitsInByte = 0;

    for (size_t i = 0; i < numElements; i++) {
        currentByte |= (tempSignArray[i] & 1) << bitsInByte;
        bitsInByte++;

        if (bitsInByte == 8) {
            signOutputBytes[signCompressedSize++] = currentByte;
            currentByte = 0;
            bitsInByte = 0;
        }
    }

    if (bitsInByte > 0) {
        signOutputBytes[signCompressedSize++] = currentByte;
    }

    // 存储位数信息
    otherOutputBytes[0] = static_cast<uint8_t>(bitsNeeded);
    
    // 压缩数值数组
    uint8_t* dataPtr = otherOutputBytes + 1;
    size_t dataCompressedSize = 0;
    
    if (bitsNeeded > 0) {
        size_t byteIndex = 0;
        uint8_t currentDataByte = 0;
        uint32_t bitsInDataByte = 0;

        for (size_t i = 0; i < numElements; i++) {
            uint32_t value = tempValueArray[i] & ((1u << bitsNeeded) - 1);
            size_t remainingBits = bitsNeeded;
            
            while (remainingBits > 0) {
                uint32_t bitsToWrite = std::min(8u - bitsInDataByte, static_cast<uint32_t>(remainingBits));
                
                currentDataByte |= (value & ((1u << bitsToWrite) - 1)) << bitsInDataByte;
                bitsInDataByte += bitsToWrite;
                remainingBits -= bitsToWrite;
                value >>= bitsToWrite;

                if (bitsInDataByte == 8) {
                    dataPtr[byteIndex++] = currentDataByte;
                    currentDataByte = 0;
                    bitsInDataByte = 0;
                }
            }
        }

        if (bitsInDataByte > 0) {
            dataPtr[byteIndex++] = currentDataByte;
        }
        
        dataCompressedSize = byteIndex;
    }

    otherCompressedSize = 1 + dataCompressedSize;
}

// 通用压缩主函数
std::pair<size_t, size_t> universalStmCompress_SZ3(
    double* input, size_t numElements, size_t blockSize, 
    const std::string& subDir, double errorBound, 
    double& compressTime, double& writeTime) {
    
    // 创建配置
    CompressionConfig config(blockSize);
    
    // 验证分块大小
    size_t validSizes[] = {64, 4096, 262144, 16777216, 1073741824}; // 4³, 16³, 64³, 256³, 1024³
    bool validSize = false;
    for (size_t size : validSizes) {
        if (blockSize == size) {
            validSize = true;
            break;
        }
    }
    if (!validSize) {
        throw std::runtime_error("Unsupported block size. Supported: 4³, 16³, 64³, 256³, 1024³");
    }
    
    // 文件名
    std::string unpredDataFileName = (std::filesystem::path(subDir) / "universal_unpredData.bin").string();
    std::string compressedFileName = (std::filesystem::path(subDir) / "universal_compressed_data.bin").string();
    std::string signFileName = (std::filesystem::path(subDir) / "universal_sign_data.bin").string();

    std::vector<int> quantIndices(numElements);
    size_t totalCompSize = 0;
    size_t totalSignSize = 0;

    // 存储元数据
    std::vector<uint64_t> allUnpredSizes;
    std::vector<uint8_t> allUnpredDataBuffers;

    int radius = 512;

    auto compressStart = std::chrono::high_resolution_clock::now();

    // 分块处理
    size_t pos = 0;
    while (pos < numElements) {
        size_t numToCompress = std::min(blockSize, numElements - pos);
        
        std::vector<double> dataCopy(input + pos, input + pos + numToCompress);

        // SZ3 压缩配置
        SZ3::Config conf(numToCompress);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = errorBound;
        conf.quantbinCnt = 1024;

        // 执行压缩
        std::vector<int> quantIndicesBlock;
        auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);
        quantIndicesBlock = decompose.compress(conf, dataCopy.data());

        // 减去半径偏移
        std::transform(quantIndicesBlock.begin(), quantIndicesBlock.end(), 
                      quantIndicesBlock.begin(),
                      [radius](int idx) { return idx - radius; });

        std::copy(quantIndicesBlock.begin(), quantIndicesBlock.end(), 
                  quantIndices.begin() + pos);

        // 处理元数据
        size_t metadataSize = 2 * sizeof(double) * blockSize;
        std::vector<uint8_t> metadataBuffer(metadataSize);
        auto bufferp = metadataBuffer.data();
        decompose.save(bufferp);

        // 提取unpredicted数据
        const uint8_t* tempBuffer = metadataBuffer.data();
        tempBuffer += 1 + sizeof(double) + sizeof(int); // 跳过标志位、error_bound、radius

        size_t unpredSize = *reinterpret_cast<const size_t*>(tempBuffer);
        tempBuffer += sizeof(size_t);
        const double* unpredData = reinterpret_cast<const double*>(tempBuffer);

        if (unpredSize > config.maxUnpredSize) {
            throw std::runtime_error("UnpredSize exceeds configuration limit");
        }

        allUnpredSizes.push_back(unpredSize);
        size_t dataSize = unpredSize * sizeof(double);
        allUnpredDataBuffers.insert(allUnpredDataBuffers.end(),
                                   reinterpret_cast<const uint8_t*>(unpredData),
                                   reinterpret_cast<const uint8_t*>(unpredData) + dataSize);

        pos += numToCompress;
    }

    // 压缩unpredicted sizes
    std::vector<uint8_t> compressedUnpredSizes;
    UniversalBitPacker::packBits(allUnpredSizes, compressedUnpredSizes, config.unpredSizeBits);

    // 将压缩后的sizes插入到数据前面
    allUnpredDataBuffers.insert(allUnpredDataBuffers.begin(),
                               compressedUnpredSizes.begin(),
                               compressedUnpredSizes.end());

    // 第二轮：处理量化差分
    pos = 0;
    std::vector<uint8_t> organizedCompressedData;
    std::vector<uint8_t> compressedSign;
    std::vector<uint8_t> bitsCounts;
    std::vector<uint64_t> compressedDataSizes;
    std::vector<uint8_t> realCompressedData;

    while (pos < numElements) {
        size_t numToCompress = std::min(blockSize, numElements - pos);

        // 分配缓冲区
        std::vector<uint8_t> signOutputBytes(config.signBytesPerBlock);
        std::vector<uint8_t> otherOutputBytes(numToCompress * 4 + 100); // 保守估算

        size_t signCompressedSize = 0;
        size_t otherCompressedSize = 0;

        processQuantizedDifferences(quantIndices.data(), pos, numToCompress,
                                   signOutputBytes.data(), otherOutputBytes.data(),
                                   signCompressedSize, otherCompressedSize, config);

        // 提取bits count
        uint8_t bitsCount = otherOutputBytes[0];
        bitsCounts.push_back(bitsCount);

        // 提取压缩数据大小
        uint64_t realSize = otherCompressedSize - 1;
        if (realSize > config.maxDataSize) {
            throw std::runtime_error("Compressed data size exceeds configuration limit");
        }
        compressedDataSizes.push_back(realSize);

        // 提取压缩数据
        realCompressedData.insert(realCompressedData.end(),
                                 otherOutputBytes.begin() + 1,
                                 otherOutputBytes.begin() + otherCompressedSize);

        // 存储sign数据
        compressedSign.insert(compressedSign.end(),
                             signOutputBytes.begin(),
                             signOutputBytes.begin() + signCompressedSize);

        pos += numToCompress;
        totalCompSize += otherCompressedSize;
        totalSignSize += signCompressedSize;
    }

    // 组织压缩数据
    organizedCompressedData.insert(organizedCompressedData.end(),
                                  bitsCounts.begin(), bitsCounts.end());

    // 写入数据大小（多字节）
    for (uint64_t size : compressedDataSizes) {
        UniversalValueWriter::writeValue(organizedCompressedData, size, config.dataSizeBytes);
    }

    // 写入实际数据
    organizedCompressedData.insert(organizedCompressedData.end(),
                                  realCompressedData.begin(), realCompressedData.end());

    auto compressEnd = std::chrono::high_resolution_clock::now();
    compressTime = std::chrono::duration<double>(compressEnd - compressStart).count();

    // 写入文件
    auto writeStart = std::chrono::high_resolution_clock::now();

    SZ3::writefile(unpredDataFileName.c_str(), allUnpredDataBuffers.data(), allUnpredDataBuffers.size());
    SZ3::writefile(compressedFileName.c_str(), organizedCompressedData.data(), organizedCompressedData.size());
    SZ3::writefile(signFileName.c_str(), compressedSign.data(), compressedSign.size());

    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<double>(writeEnd - writeStart).count();

    std::cout << "[Time] Data Compression Time: " << compressTime << " seconds" << std::endl;
    std::cout << "[Time] File Write Time: " << writeTime << " seconds" << std::endl;
    
    return {totalCompSize, totalSignSize};
}





int main(int argc, char *argv[])
{
    double totalCompressTime = 0.0;
    double totalWriteTime = 0.0;


    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    std::string inputFileName;
    std::string variableName;
    std::string variableType;
    size_t nDim = 0;
    size_t beginStepNum = 0;
    size_t endStepNum = 0;
    std::vector<size_t> blockShape;
    std::vector<size_t> targetSteps; 
    double relative_error_bound = 1E-3;
    size_t extraValue =0;

    // 解析命令行参数
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc)
        {
            inputFileName = argv[++i];
        }
        else if (arg == "--variable_name" && i + 1 < argc)
        {
            variableName = argv[++i];
        }
        else if (arg == "--dimensions" && i + 1 < argc)
        {
            nDim = std::stoul(argv[++i]);
        }else if (arg == "--begin_step")
        {
            if (i + 1 < argc)
            {
                beginStepNum = atoi(argv[i + 1]);
            }
            else
            {
                std::cerr << "--begin_step option requires one argument." << std::endl;
                return 1;
            }
        }
        else if (arg == "--end_step")
        {
            if (i + 1 < argc)
            {
                endStepNum = atoi(argv[i + 1]);
            }
            else
            {
                std::cerr << "--end_step option requires one argument." << std::endl;
                return 1;
            }
        }
        else if (arg == "--target_steps")
        {
            while (i + 1 < argc && std::isdigit(argv[i + 1][0])) {
                targetSteps.push_back(std::stoul(argv[++i]));
            }
        }
        else if (arg == "--block_shape" && i + nDim < argc)
        {
            for (size_t j = 0; j < nDim; j++)
            {
                blockShape.push_back(std::stoul(argv[++i]));
            }
        }else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);

        } 
    }

    if (inputFileName.empty() || variableName.empty() || nDim == 0 || blockShape.size() != nDim) 
    {
        std::cerr << "Error: missing or invalid arguments." << std::endl;
        return 1;
    }

    // 计算每个块的数据点个数 blockSize
    int blockSize = 1;
    for (const auto& dim : blockShape)
    {
        blockSize *= dim;
    }

    /*//test split time
    size_t mockDimSize = 64;
    std::vector<double> mockVarData(mockDimSize * mockDimSize * mockDimSize, 1.0);
    std::vector<size_t> mockDataShape = {mockDimSize, mockDimSize, mockDimSize};
    std::vector<size_t> mockBlockShape = {4, 4, 4};
    
    testSplitTime(mockVarData, mockDataShape, mockBlockShape);*/
    


    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);

    // 定义 flatBlockData 用于保存所有 step 的数据
    //全部step，按照一维小块的顺序排列
    std::vector<double> flatBlockData;
    size_t step = 0;

    double totalSplitTime = 0.0;

    std::set<size_t> targetSet(targetSteps.begin(), targetSteps.end());



    while (reader_engine.BeginStep() == adios2::StepStatus::OK)
    {
        //固定step才读取
        if (!targetSet.empty() && targetSet.find(step) == targetSet.end()) 
        {
            reader_engine.EndStep();
            step++;
            continue;
        }
        auto var = reader_io.InquireVariable<double>(variableName);

        variableType= reader_io.VariableType(variableName);
       
        size_t varElements = 1;
        for (size_t i = 0; i < nDim; i++)
        {
            varElements *= var.Shape()[i];
        }

        // 计算每个维度的块数量
        std::vector<size_t> blockCountOnEachDim;
        size_t total_blocks = 1;
        for (size_t i = 0; i < nDim; i++)
        {
            blockCountOnEachDim.push_back(var.Shape()[i] / blockShape[i]);
            total_blocks *= var.Shape()[i] / blockShape[i];
        }

        if (variableType == "double")
        {

            std::vector<double> varData(varElements);
            reader_engine.Get(var, varData.data(), adios2::Mode::Sync);

            // === 开始数据分块 ===
            auto splitStart = std::chrono::high_resolution_clock::now();

            std::vector<std::vector<double>> blocks(total_blocks);
            // 为当前 step 的数据分配空间，并将其追加到 flatBlockData 中
            flatBlockData.reserve(flatBlockData.size() + varElements);

            //处理每一个点，将对应的点放入对应的块中去
            for (size_t p = 0; p < varElements; p++)
            {
                std::vector<size_t> elem_global_id = positionToIndices(p, var.Shape());
                std::vector<size_t> block_global_id(nDim);
                for (size_t i = 0; i < nDim; i++)
                {
                    block_global_id[i] = elem_global_id[i]/blockShape[i];
                }                
                size_t block_position = indicesToPosition(blockCountOnEachDim, block_global_id);
                blocks[block_position].push_back(varData[p]);
            }

            for (size_t b = 0; b < total_blocks; b++)
            {
                // 将每个块的数据追加到 flatBlockData
                flatBlockData.insert(flatBlockData.end(), blocks[b].begin(), blocks[b].end());
            }

            auto splitEnd = std::chrono::high_resolution_clock::now();
            totalSplitTime += std::chrono::duration<double>(splitEnd - splitStart).count();


        }

        reader_engine.EndStep();
        step++;
    }

    reader_engine.Close();

    std::cout << "[Time] Total Split Time (All Steps): " << totalSplitTime << " seconds" << std::endl;


    double Global_min_value = *std::min_element(flatBlockData.begin(), flatBlockData.end());
    double Global_max_value = *std::max_element(flatBlockData.begin(), flatBlockData.end());
    //已经通过相对误差，计算出绝对误差
    double error_bound = relative_error_bound * (Global_max_value - Global_min_value);



  
    std::vector<double> localFlatBlockData;

    if (world_rank == 0) {
        size_t total_blocks = flatBlockData.size() / blockSize;
        size_t blocks_per_proc = (total_blocks + world_size - 1) / world_size;

        for (int rank = 1; rank < world_size; ++rank) {
            size_t start_block = rank * blocks_per_proc;
            size_t end_block = std::min(start_block + blocks_per_proc, total_blocks);
            size_t num_vals = (end_block - start_block) * blockSize;

            MPI_Send(flatBlockData.data() + start_block * blockSize, num_vals, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
        }

        // 自己的数据保留
        size_t start_block = 0;
        size_t end_block = std::min(blocks_per_proc, flatBlockData.size() / blockSize);
        localFlatBlockData.assign(flatBlockData.begin(), flatBlockData.begin() + end_block * blockSize);

    } else {
       
        size_t total_blocks = flatBlockData.size() / blockSize;  
        size_t blocks_per_proc = (total_blocks + world_size - 1) / world_size;
        size_t start_block = world_rank * blocks_per_proc;
        size_t end_block = std::min(start_block + blocks_per_proc, total_blocks);
        size_t num_vals = (end_block - start_block) * blockSize;

        localFlatBlockData.resize(num_vals);
        MPI_Recv(localFlatBlockData.data(), num_vals, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string subDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" +safeVarName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_blockZFP_originalDataCompression/";





    /*std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string baseSubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "__originalDataCompression/";

    std::string subDir = baseSubDir + "rank_" + std::to_string(world_rank) + "/";*/
    
    createMoreDirectory(subDir);

    std::cout << "Error Bound Set: " << error_bound << std::endl;

    double thisCompressTime = 0.0;
    double thisWriteTime = 0.0;


    //压缩全部小块数据
    auto [totalCompSize, totalSignSize] = universalStmCompress_SZ3(localFlatBlockData.data(), localFlatBlockData.size(), blockSize, subDir, error_bound, thisCompressTime, thisWriteTime);
    totalCompressTime += thisCompressTime;
    totalWriteTime += thisWriteTime;

    std::cout << "[Rank " << world_rank << "] Total Compress Time: " << totalCompressTime << " seconds\n";
    std::cout << "[Rank " << world_rank << "] Total Write Time: " << totalWriteTime << " seconds\n";


    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}  
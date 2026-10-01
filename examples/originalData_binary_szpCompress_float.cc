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
#include <zfp.h> 



// Function prototypes for encoding helper functions
unsigned int convertIntArray2ByteArray_fast_1b_args(const unsigned char* input, size_t inputSize, unsigned char* output);
unsigned int save_fixed_length_bits(const unsigned int* input, size_t inputSize, unsigned char* output, unsigned int bit_count);

unsigned int calculateBitCount(unsigned int maxDifference);



void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
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

//fix_length encoding
inline std::pair<size_t, size_t> stmCompress_SZ3(float *input, size_t numElements, int blockSize, std::string subDir, float error_bound, float &compressTime, float &writeTime) 
{
    // 创建文件名包含 step 范围
    std::string unpredDataFileName = (std::filesystem::path(subDir) / "fixed_unpredData.bin").string();
    std::string firstValueFileName = (std::filesystem::path(subDir) / "fixed_firstValue.bin").string();
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

    //开始压缩
    auto compressStart = std::chrono::high_resolution_clock::now();

    while (pos < numElements) {
        size_t numToCompress = blockSize;
        if (numToCompress + pos > numElements) numToCompress = numElements - pos;

        std::vector<float> data_cpy(&input[pos], &input[pos + numToCompress]);

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
        auto predictor = SZ3::LorenzoPredictor<float, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<float, 1>(conf, quantizer);
        quant_inds_block = decompose.compress(conf, data_cpy.data());

        // 对量化索引值减去偏移
        std::transform(
            quant_inds_block.begin(), quant_inds_block.end(), quant_inds_block.begin(),
            [radius](int quant_index) { return quant_index - radius; }
        );


        //现在每个块再填充到quant_inds中
        std::copy(quant_inds_block.begin(), quant_inds_block.end(), quant_inds.begin() + pos);

        // 创建缓冲区以保存元数据
        size_t metadataSize = 2 * sizeof(float) * blockSize;
        std::vector<SZ3::uchar> metadataBuffer(metadataSize);

        // 保存元数据到缓冲区
        auto bufferp = metadataBuffer.data();
        decompose.save(bufferp);

        // 提取 unpred.size() 和 unpred.data()
        const SZ3::uchar *tempBuffer = metadataBuffer.data();

        // 跳过元数据中的前几个字段
        tempBuffer += 1; // 跳过第一个字节（标志位）
        tempBuffer += sizeof(double); // 跳过 error_bound (float 类型)
        tempBuffer += sizeof(int); // 跳过 radius (int 类型)

        size_t unpredSize = *reinterpret_cast<const size_t *>(tempBuffer);
        tempBuffer += sizeof(size_t);
        const float *unpredData = reinterpret_cast<const float *>(tempBuffer);

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
        size_t dataSize = unpredSize * sizeof(float);
        allUnpredDataBuffers.insert(allUnpredDataBuffers.end(),
                                    reinterpret_cast<const SZ3::uchar *>(unpredData),
                                    reinterpret_cast<const SZ3::uchar *>(unpredData) + dataSize);


        
        pos += numToCompress;
    }


    //metadata处理

    /*//unpredSize的处理
    //每个块保存成一个uchar
    std::vector<SZ3::uchar> compressedUnpredSize;
    compressUnpredSizes(allUnpredSizeBuffers, compressedUnpredSize);*/

    allUnpredDataBuffers.insert(
        allUnpredDataBuffers.begin(),  // 插入位置：起始位置
        allUnpredSizeBuffers.begin(), // 插入数据的起始迭代器
        allUnpredSizeBuffers.end()    // 插入数据的结束迭代器
    );
    
    /*//每6个bits表示一个block的unpredictedsize，总共用前，block个数*6/8（向上取整）个bytes表示了大小*/
    

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

        //每块第一个值通过0余预测，通过0解压即可，不用记录啊
        
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

        //提取真实压缩的数据
        //最后一个字节不满，在大端
        realCompressedData.insert(realCompressedData.end(), otherOutputBytes + 1, otherOutputBytes + otherCompressedSize);

        //每一个块有63个bin,有63个sign,每一个块用8个bytes来表示sign
        //compressedSign是每8个bytes用来表示一个block的全部sign，依次存储下去
        compressedSign.insert(compressedSign.end(), signOutputBytes, signOutputBytes + signCompressedSize);

        pos += numToCompress;


        
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

    auto compressEnd = std::chrono::high_resolution_clock::now();
    compressTime = std::chrono::duration<float>(compressEnd - compressStart).count();

    auto writeStart = std::chrono::high_resolution_clock::now();


    /*//如果radius是512，可以10bits来表示第一个值
    std::vector<char> compressedFirstData;              // 用于存储压缩后的数据
    compressTo10Bits(firstValues, compressedFirstData);
    //10bits表示一个firstdata
    SZ3::writefile(firstValueFileName.c_str(), compressedFirstData.data(), compressedFirstData.size());*/

    //前block个数个uchar记录unpredictedsize
    //后面依次存储float类型的unpredicted data（个数根据unpredSize确定
    SZ3::writefile(unpredDataFileName.c_str(), allUnpredDataBuffers.data(), allUnpredDataBuffers.size());
 
    // 使用 SZ3::writefile 将 compressedData 写入文件
    //bitsCounts（块的个数个，每个1bytes）+压缩后数据大小（块的个数个，每个1bytes）+压缩后数据（块的个数个，大小根据前面的大小确定）
    SZ3::writefile(compressedFileName.c_str(), organizedCompressedData.data(), organizedCompressedData.size());

    // 每8个bytes用来表示一个block的全部sign，依次存储下去
    SZ3::writefile(signFileName.c_str(), compressedSign.data(), compressedSign.size());

    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<float>(writeEnd - writeStart).count();

    std::cout << "[Time] Data Compression Time: " << compressTime << " seconds" << std::endl;
    std::cout << "[Time] File Write Time: " << writeTime << " seconds" << std::endl;
  
    return {totalCompSize, totalSignSize};
}


inline std::pair<size_t, size_t> stmCompress_SZ3_default(
    float *input, size_t numElements, int blockSize,
    std::string subDir, float error_bound,
    float &compressTime, float &writeTime)
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

    // === 分块压缩 & 保存metadata ===
    while (pos < numElements) {
        size_t numToCompress = std::min(static_cast<size_t>(blockSize), numElements - pos);
        std::vector<float> data_cpy(&input[pos], &input[pos + numToCompress]);

        SZ3::Config conf(numToCompress);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;

        auto quantizer = SZ3::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<float, 1>(conf, quantizer);
        std::vector<int> quant_inds_block = decompose.compress(conf, data_cpy.data());

        std::copy(quant_inds_block.begin(), quant_inds_block.end(), quant_inds.begin() + pos);

        std::vector<SZ3::uchar> metadataBuffer(60 * sizeof(float) * blockSize);
        SZ3::uchar *bufferp = metadataBuffer.data();
        decompose.save(bufferp);
        size_t metaSize = bufferp - metadataBuffer.data();

        metaSizes.push_back(metaSize);
        metaDataBuffer.insert(metaDataBuffer.end(), metadataBuffer.begin(), metadataBuffer.begin() + metaSize);

        pos += numToCompress;
    }

    // === 构建 Huffman 编码器 ===
    size_t bufferSize = 60 * sizeof(float) * blockSize;
    std::vector<SZ3::uchar> buffer(bufferSize);
    std::vector<SZ3::uchar> buffer_zstd(bufferSize);

    SZ3::HuffmanEncoder<int> huffman;
    huffman.preprocess_encode(quant_inds, 0);

    SZ3::uchar *buffer_pos = buffer.data();
    huffman.save(buffer_pos);
    size_t huffmanSize = buffer_pos - buffer.data();

    // === 编码每个块 ===
    pos = 0;
    while (pos < numElements) {
        size_t numToCompress = std::min(static_cast<size_t>(blockSize), numElements - pos);

        buffer_pos = buffer.data();
        huffman.encode(&quant_inds[pos], numToCompress, buffer_pos);

        size_t csize = SZ3::Lossless_zstd().compress(buffer.data(), buffer_pos - buffer.data(), buffer_zstd.data(), bufferSize);
        compSizes.push_back(csize);
        compDataBuffer.insert(compDataBuffer.end(), buffer_zstd.begin(), buffer_zstd.begin() + csize);

        totalCompSize += csize;
        pos += numToCompress;
    }

    auto compressEnd = std::chrono::high_resolution_clock::now();
    compressTime = std::chrono::duration<float>(compressEnd - compressStart).count();

    auto writeStart = std::chrono::high_resolution_clock::now();

    // === 写 metadata ===
    std::vector<SZ3::uchar> metaCombined(metaSizes.size() * sizeof(size_t) + metaDataBuffer.size());
    std::memcpy(metaCombined.data(), metaSizes.data(), metaSizes.size() * sizeof(size_t));
    std::memcpy(metaCombined.data() + metaSizes.size() * sizeof(size_t), metaDataBuffer.data(), metaDataBuffer.size());
    SZ3::writefile(unpredDataFileName.c_str(), metaCombined.data(), metaCombined.size());

    // === 写压缩数据 ===
    std::vector<SZ3::uchar> compCombined(compSizes.size() * sizeof(size_t) + compDataBuffer.size());
    std::memcpy(compCombined.data(), compSizes.data(), compSizes.size() * sizeof(size_t));
    std::memcpy(compCombined.data() + compSizes.size() * sizeof(size_t), compDataBuffer.data(), compDataBuffer.size());
    SZ3::writefile(compressedFileName.c_str(), compCombined.data(), compCombined.size());

    // === 写 Huffman 文件 ===
    SZ3::writefile(HuffmanFileName.c_str(), buffer.data(), huffmanSize);

    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<float>(writeEnd - writeStart).count();

    return std::make_pair(totalCompSize, compSizes.size());
}


inline std::pair<size_t, size_t> stmCompress_ZFP_default(
    float* input,
    size_t numElements,
    int blockSize,
    const std::string& subDir,
    float tolerance,
    float& compressTime,
    float& writeTime)
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

        zfp_type type = zfp_type_float;
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
    compressTime = std::chrono::duration<float>(compressEnd - compressStart).count();

    auto writeStart = std::chrono::high_resolution_clock::now();

    std::ofstream sizeOut(sizeFile, std::ios::binary);
    sizeOut.write(reinterpret_cast<const char*>(&blockCount), sizeof(size_t));
    sizeOut.write(reinterpret_cast<const char*>(compressedSizes.data()), blockCount * sizeof(size_t));
    sizeOut.close();
    out.close();

    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<float>(writeEnd - writeStart).count();

    return {totalSize, blockCount};
}










int main(int argc, char *argv[])
{
    std::string inputFileName;
    //std::string variableName;
    std::string variableType;
    size_t nDim = 0;
    std::vector<size_t> dataShape;
    std::vector<size_t> blockShape;
    size_t beginStepNum = 0;
    size_t endStepNum = 0;

    float relative_error_bound = 1E-3;  // 默认相对误差
    size_t extraValue =0; //用来标识error_bound量级的数值

    // 解析命令行参数
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc)
        {
            inputFileName = argv[++i];
        }
        else if (arg == "--dimensions" && i + 1 < argc)
        {
            nDim = std::stoul(argv[++i]);
            for (size_t j = 0; j < nDim; j++) {
                dataShape.push_back(std::stoul(argv[++i]));
            }

        }
        else if (arg == "--block_shape" && i + nDim < argc)
        {
            for (size_t j = 0; j < nDim; j++)
            {
                blockShape.push_back(std::stoul(argv[++i]));
            }
        }
        else if (arg == "--begin_step")
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
        else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);

        }   

    }

    if (inputFileName.empty() || nDim == 0 || blockShape.size() != nDim) 
    {
        std::cerr << "Error: missing or invalid arguments." << std::endl;
        return 1;
    }

    //数据大小
    size_t totalElements = 1;
    for (size_t dim : dataShape) {
           totalElements *= dim;
    }

    // 计算每个块的数据点个数 blockSize
    int blockSize = 1;
    for (const auto& dim : blockShape)
    {
        blockSize *= dim;
    }

    double totalSplitTime = 0.0;

    // **读取二进制文件**
    std::vector<float> varData = readBinaryFile(inputFileName, totalElements);
    if (varData.empty()) {
         std::cerr << "Error: Failed to read binary file " << inputFileName << std::endl;
         return 1;
    }

    float Global_min_value = *std::min_element(varData.begin(), varData.end());
    float Global_max_value = *std::max_element(varData.begin(), varData.end());
    //已经通过相对误差，计算出绝对误差
    float error_bound = relative_error_bound * (Global_max_value - Global_min_value);


    /*// debug
    for (size_t i = 0; i < std::min<size_t>(300, varData.size()); ++i) {
        std::cout << "varData[" << i << "] = " << std::setprecision(6) << varData[i] << std::endl;
    }*/

    // 计算每个维度的块数量
    std::vector<size_t> blockCountOnEachDim(nDim);
    size_t total_blocks = 1;
    for (size_t i = 0; i < nDim; i++) {
        blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];  // 向上取整，避免数据不整除导致丢失
        total_blocks *= blockCountOnEachDim[i];
    }

    auto splitStart = std::chrono::high_resolution_clock::now();

    // 初始化 flatBlockData，减少后续插入扩容
    std::vector<float> flatBlockData;
    flatBlockData.reserve(totalElements);  // 预分配空间

    // 块，和每块其中每个点
    std::vector<std::vector<float>> blocks(total_blocks, std::vector<float>());  // 初始化所有块

    // 遍历数据点，将其分配到相应的块
    // 遍历数据点，将其分配到相应的块（并确保块内为 z-y-x 顺序）
    //处理每一个点，将对应的点放入对应的块中去
    for (size_t p = 0; p < totalElements; p++)
    {
        std::vector<size_t> elem_global_id = positionToIndices(p, dataShape);
        std::vector<size_t> block_global_id(nDim);
        for (size_t i = 0; i < nDim; i++)
        {
            block_global_id[i] = elem_global_id[i]/blockShape[i];
        }                
        size_t block_position = indicesToPosition(blockCountOnEachDim, block_global_id);
        blocks[block_position].push_back(varData[p]);
    }


    // 依次收集所有块的数据到 flatBlockData
    for (size_t b = 0; b < total_blocks; b++) {
        flatBlockData.insert(flatBlockData.end(), blocks[b].begin(), blocks[b].end());
    }

    /*//debug
    for (size_t i = 0; i < std::min<size_t>(300, flatBlockData.size()); ++i) {
        std::cout << "flatBlockData[" << i << "] = " << std::setprecision(6) << flatBlockData[i] << std::endl;
    }*/
    

    auto splitEnd = std::chrono::high_resolution_clock::now();
    totalSplitTime += std::chrono::duration<double>(splitEnd - splitStart).count();

    std::cout << "[Time] Total Split Time (All Steps): " << totalSplitTime << " seconds" << std::endl;


    //std::cout << "blockSize: " << blockSize << std::endl;

    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string subDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +   "_iso_fixlength_originalDataCompression/";

    createDirectory(subDir);

    std::cout << "Error Bound Set: " << error_bound << std::endl;

    float thisCompressTime = 0.0;
    float thisWriteTime = 0.0;

    

    auto [totalCompSize, totalSignSize] = stmCompress_SZ3(flatBlockData.data(), flatBlockData.size(), blockSize, subDir, error_bound,thisCompressTime, thisWriteTime);


    // 输出压缩结果的信息
    std::cout << "Compressed Data Size: " << totalCompSize << std::endl;

    std::cout << "Sign Data Size: " << totalSignSize << std::endl;

    return 0;
}  
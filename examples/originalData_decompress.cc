#include <iostream>
#include <vector>
#include <cstdlib>
#include <adios2.h>
#include <cstddef>
#include <cstring>
#include <SZ3/api/sz.hpp>  



// 读取二进制文件内容到缓冲区
std::vector<SZ3::uchar> readBinaryFile(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return {};
    }

    // 获取文件大小
    inFile.seekg(0, std::ios::end);
    size_t fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    // 读取文件到缓冲区
    std::vector<SZ3::uchar> buffer(fileSize);
    inFile.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    inFile.close();

    if (!inFile.good()) {
        std::cerr << "Error occurred while reading from the file!" << std::endl;
    }

    return buffer;
}


inline std::vector<double> stmDecompress_SZ3(const std::string& huffmanFileName, const std::string& compressedFileName, size_t numElements, int blockSize) 
{
    std::vector<int> quant_inds(numElements); // To store decompressed quantization indices
    size_t pos = 0;

    // Step 1: Load Huffman Tree
    std::vector<SZ3::uchar> huffmanBuffer = readBinaryFile(huffmanFileName);
    
    if (huffmanBuffer.empty()) {
        std::cerr << "Huffman file could not be read." << std::endl;
        return {};
    }
    
    const SZ3::uchar* buffer_pos = huffmanBuffer.data();
    size_t remaining_length = huffmanBuffer.size();
    //std::cout << "Huffman buffer size: " << remaining_length << std::endl;
    
    SZ3::HuffmanEncoder<int> huffman;
    huffman.load(buffer_pos, remaining_length);


    // Step 2: Read Compressed Data
    std::vector<SZ3::uchar> compressedData = readBinaryFile(compressedFileName);

    if (compressedData.empty()) {
        std::cerr << "Compressed file could not be read." << std::endl;
        return {};
    }
    auto *data_pos = compressedData.data();

    // Extract compressedSizes from the beginning of compressedData
    size_t numBlocks = (numElements + blockSize - 1) / blockSize;


    // 提取每个块的元数据大小和压缩总大小
    std::vector<size_t> metadataSizes(numBlocks), totalSizes(numBlocks);


    for (size_t i = 0; i < numBlocks; ++i) {
        totalSizes[i] = (static_cast<size_t>(data_pos[4 * i]) << 8) |
                        static_cast<size_t>(data_pos[4 * i + 1]);

        metadataSizes[i] = (static_cast<size_t>(data_pos[4 * i + 2]) << 8) |
                        static_cast<size_t>(data_pos[4 * i + 3]);
    }
    data_pos += numBlocks * 4;




    // Step 3: Decompress each block
    SZ3::Lossless_zstd zstd;
    auto *buffer = new SZ3::uchar[2 * sizeof(double) * blockSize];
    std::vector<double> decompressedData(numElements); // Final decompressed data output

    for (size_t i = 0; i < numBlocks; ++i) {
        // Extract compressed block data
        size_t csize = totalSizes[i];
        size_t metadataSize = metadataSizes[i];


        /*std::cout << "Decompressing block " << i << ": totalSize = " << csize
                  << ", metadataSize = " << metadataSize << std::endl;*/



        //设置了8倍大小（足够大），但可调整
        size_t decompressionBufferSize = 2 * sizeof(double) * blockSize;

      
        size_t realDecompressedSize = zstd.decompress(data_pos, csize, buffer, decompressionBufferSize);

        size_t quant_data_size = realDecompressedSize - metadataSize;

        data_pos += csize;

        const SZ3::uchar* quant_data_pos = buffer;  // 量化数据起始
        const SZ3::uchar* metadata_pos = buffer + quant_data_size;  // 元数据起始


        SZ3::Config conf(blockSize);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        //conf.errorBoundMode = SZ3::EB_ABS;
        //conf.absErrorBound = 1e-4;
        //SZ3::calAbsErrorBound(conf, data_cpy.data());
    
    
        auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);


        size_t remaining_length = metadataSize; 
        decompose.load(metadata_pos, remaining_length);



        // Decode with Huffman
        size_t numToDecompress = blockSize;
        if (numToDecompress + pos > numElements) numToDecompress = numElements - pos;
        
        
        std::vector<int> decodedQuantData = huffman.decode(quant_data_pos, numToDecompress);


        decompose.decompress(conf, decodedQuantData, decompressedData.data() + pos);
   

        pos += numToDecompress;

    }

    delete[] buffer;
    return decompressedData; // Return the decompressed data
}



// 保存数据为可读的文本文件
void saveDataAsText(const std::vector<double>& data, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (const double& value : data) {
        outFile << value << "\n";
    }

    outFile.close();
    if (!outFile.good()) {
        std::cerr << "Error occurred while writing data to the file!" << std::endl;
    } else {
        std::cout << "Data has been saved as readable text to " << filename << std::endl;
    }
}

std::vector<double> loadDataFromText(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return {};
    }

    // 获取文件大小
    inFile.seekg(0, std::ios::end);
    size_t fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    // 计算 double 数量
    size_t numElements = fileSize / sizeof(double);
    std::vector<double> flatBlockData(numElements);

    // 读取二进制数据
    inFile.read(reinterpret_cast<char*>(flatBlockData.data()), fileSize);

    inFile.close();
    if (!inFile.good() && !inFile.eof()) {
        std::cerr << "Error occurred while reading flatBlockData from the binary file!" << std::endl;
    } else {
        std::cout << "flatBlockData has been successfully loaded from binary file: " << filename << std::endl;
    }

    return flatBlockData;
}


// 比较解压前后的数据是否在误差范围内（std::setprecision(15) 保留精度）
bool compareDataWithinErrorRange(const std::vector<double>& original, const std::vector<double>& decompressed, double errorBound) {
    if (original.size() != decompressed.size()) {
        std::cerr << "Data size mismatch between original and decompressed data." << std::endl;
        return false;
    }

    for (size_t i = 0; i < original.size(); ++i) {
        double absError = std::abs(original[i] - decompressed[i]);
       


       std::cerr << std::setprecision(15) 
          << "Original: " << original[i]
          << ", Decompressed: " << decompressed[i]
          << ", Abs Error: " << absError
          << ", Error Bound: " << errorBound
          << std::endl;


        if (absError > errorBound) {
            std::cerr << "Data at index " << i << " is out of error range: "
                      << "original = " << original[i] << ", decompressed = " << decompressed[i] << std::endl;
            return false;
        }
    }
    return true;
}


int main(int argc, char* argv[]) {
    // 检查命令行参数数量是否正确
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <huffmanFileName> <compressedFileName> <numElements> <blockSize>" << std::endl;
        return 1;
    }

    // 解析命令行参数
    std::string huffmanFileName = argv[1];
    std::string compressedFileName = argv[2];
    size_t numElements = std::stoul(argv[3]); // 将参数转换为 size_t
    int blockSize = std::stoi(argv[4]);       // 将参数转换为 int

    // 调用解压函数
    std::vector<double> decompressedData = stmDecompress_SZ3(huffmanFileName, compressedFileName, numElements, blockSize);

    // 检查解压数据是否成功
    if (decompressedData.empty()) {
        std::cerr << "Decompression failed or returned empty data." << std::endl;
        return -1;
    }



    //std::string outputFileName = "original_data_readable.txt";
    //std::vector<double> flatBlockData = loadDataFromText(outputFileName);
    //double errorBound = 1e-5;

    /*if (compareDataWithinErrorRange(flatBlockData, decompressedData, errorBound)) {
        std::cout << "Decompressed data is within the error bound." << std::endl;
    } else {
        std::cout << "Decompressed data is outside the error bound!" << std::endl;
    }*/

    return 0;
}


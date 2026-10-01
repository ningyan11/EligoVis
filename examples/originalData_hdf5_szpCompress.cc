#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <SZ3/api/sz.hpp> 
#include <stdexcept>
#include <cstdio>
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <chrono>
#include <zfp.h> 
#include <H5Cpp.h>   // HDF5 C++ API



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

// ============================================================
// 新增：直接从 HDF5 文件里读取一个 float32 dataset
// ============================================================
// filename    : hdf5 文件路径
// datasetName : dataset 在文件内的完整路径, 例如 "native_fields/baryon_density"
// outShape    : 输出参数, 读到的 dataset 各维度大小 (会覆盖传入内容)
//
// 注意: 这个函数会把整个 dataset 一次性读进内存 (跟原来的
// readBinaryFile 行为一致), 2048^3 的 float32 大约是 32GB,
// 请在有足够内存的计算节点上运行, 不要在登录节点跑。
std::vector<float> readHDF5Dataset(const std::string& filename,
                                    const std::string& datasetName,
                                    std::vector<size_t>& outShape) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(datasetName);
        H5::DataSpace dataspace = dataset.getSpace();

        int rank = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        dataspace.getSimpleExtentDims(dims.data(), nullptr);

        outShape.resize(rank);
        size_t totalElements = 1;
        for (int i = 0; i < rank; i++) {
            outShape[i] = static_cast<size_t>(dims[i]);
            totalElements *= outShape[i];
        }

        // 检查数据类型大小，确保是 4 字节 (float32)
        H5::DataType dtype = dataset.getDataType();
        if (dtype.getSize() != sizeof(float)) {
            throw std::runtime_error(
                "Dataset '" + datasetName + "' element size is " +
                std::to_string(dtype.getSize()) +
                " bytes, expected 4 bytes (float32). "
                "如果是 float64, 请改用 double 版本的压缩代码。");
        }

        std::cout << "[HDF5] Opening dataset: " << datasetName << std::endl;
        std::cout << "[HDF5] Shape: (";
        for (size_t i = 0; i < outShape.size(); i++) {
            std::cout << outShape[i] << (i + 1 < outShape.size() ? ", " : "");
        }
        std::cout << "), total elements = " << totalElements
                  << " (~" << (totalElements * sizeof(float)) / (1024.0*1024.0*1024.0)
                  << " GB)" << std::endl;

        std::vector<float> data(totalElements);

        auto readStart = std::chrono::high_resolution_clock::now();
        dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);
        auto readEnd = std::chrono::high_resolution_clock::now();
        double readTime = std::chrono::duration<double>(readEnd - readStart).count();
        std::cout << "[HDF5] Read Time: " << readTime << " seconds" << std::endl;

        return data;

    } catch (H5::FileIException& e) {
        std::cerr << "HDF5 File Error: " << e.getCDetailMsg() << std::endl;
        throw;
    } catch (H5::DataSetIException& e) {
        std::cerr << "HDF5 DataSet Error (dataset name wrong?): " << e.getCDetailMsg() << std::endl;
        throw;
    } catch (H5::DataSpaceIException& e) {
        std::cerr << "HDF5 DataSpace Error: " << e.getCDetailMsg() << std::endl;
        throw;
    }
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

    unsigned int bit_count = calculateBitCount(max);

    signCompressedSize = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, numToCompress, signOutputBytes);

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


//fix_length encoding
inline std::pair<size_t, size_t> stmCompress_SZ3(float *input, size_t numElements, int blockSize, std::string subDir, float error_bound, float &compressTime, float &writeTime) 
{
    std::string unpredDataFileName = (std::filesystem::path(subDir) / "fixed_unpredData.bin").string();
    std::string compressedFileName = (std::filesystem::path(subDir) / "fixed_compressed_data.bin").string();
    std::string signFileName = (std::filesystem::path(subDir) / "fixed_sign_data.bin").string();

    std::vector<int> quant_inds(numElements);
    std::size_t pos = 0;
    std::size_t totalCompSize = 0;
    std::size_t totalSignSize = 0;

    std::vector<SZ3::uchar> allUnpredSizeBuffers;
    std::vector<SZ3::uchar> allUnpredDataBuffers;

    int radius = 512;

    auto compressStart = std::chrono::high_resolution_clock::now();

    while (pos < numElements) {
        size_t numToCompress = blockSize;
        if (numToCompress + pos > numElements) numToCompress = numElements - pos;

        std::vector<float> data_cpy(&input[pos], &input[pos + numToCompress]);

        SZ3::Config conf(numToCompress);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;

        std::vector<int> quant_inds_block;
        auto predictor = SZ3::LorenzoPredictor<float, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<float, 1>(conf, quantizer);
        quant_inds_block = decompose.compress(conf, data_cpy.data());

        std::transform(
            quant_inds_block.begin(), quant_inds_block.end(), quant_inds_block.begin(),
            [radius](int quant_index) { return quant_index - radius; }
        );

        std::copy(quant_inds_block.begin(), quant_inds_block.end(), quant_inds.begin() + pos);

        size_t metadataSize = 2 * sizeof(float) * blockSize;
        std::vector<SZ3::uchar> metadataBuffer(metadataSize);

        auto bufferp = metadataBuffer.data();
        decompose.save(bufferp);

        const SZ3::uchar *tempBuffer = metadataBuffer.data();

        tempBuffer += 1;               // 跳过标志位
        tempBuffer += sizeof(double);   // 跳过 error_bound (永远是double)
        tempBuffer += sizeof(int);     // 跳过 radius (int 类型)

        size_t unpredSize = *reinterpret_cast<const size_t *>(tempBuffer);
        tempBuffer += sizeof(size_t);
        const float *unpredData = reinterpret_cast<const float *>(tempBuffer);

        if (unpredSize >= 256) {
            std::cerr << "Error: unpredSize exceeds expected limit of 256." << std::endl;
            break;
        }

        allUnpredSizeBuffers.push_back(static_cast<SZ3::uchar>(unpredSize));

        size_t dataSize = unpredSize * sizeof(float);
        allUnpredDataBuffers.insert(allUnpredDataBuffers.end(),
                                    reinterpret_cast<const SZ3::uchar *>(unpredData),
                                    reinterpret_cast<const SZ3::uchar *>(unpredData) + dataSize);

        pos += numToCompress;
    }

    allUnpredDataBuffers.insert(
        allUnpredDataBuffers.begin(),
        allUnpredSizeBuffers.begin(),
        allUnpredSizeBuffers.end()
    );

    pos = 0;

    std::vector<unsigned char> organizedCompressedData;
    std::vector<unsigned char> compressedSign;

    std::vector<unsigned char> bitsCounts;
    std::vector<unsigned char> compressedDataSizes;
    std::vector<unsigned char> realCompressedData;

    while (pos < numElements) {
        size_t numToCompress = blockSize;
        if (numToCompress + pos > numElements) numToCompress = numElements - pos;

        size_t maxSignBufferSize = numToCompress;
        size_t maxOtherBufferSize = numToCompress * 8;

        unsigned char *signOutputBytes = (unsigned char *)malloc(maxSignBufferSize);
        unsigned char *otherOutputBytes = (unsigned char *)malloc(maxOtherBufferSize);

        size_t signCompressedSize = 0;
        size_t otherCompressedSize = 0;

        processDiffs(quant_inds.data(), pos, numToCompress, signOutputBytes, otherOutputBytes, signCompressedSize, otherCompressedSize);

        unsigned char bitsCount = otherOutputBytes[0];
        bitsCounts.push_back(bitsCount);

        unsigned char realSize = static_cast<unsigned char>(otherCompressedSize - 1);
        compressedDataSizes.push_back(realSize);

        realCompressedData.insert(realCompressedData.end(), otherOutputBytes + 1, otherOutputBytes + otherCompressedSize);

        compressedSign.insert(compressedSign.end(), signOutputBytes, signOutputBytes + signCompressedSize);

        free(signOutputBytes);
        free(otherOutputBytes);

        pos += numToCompress;

        totalCompSize += otherCompressedSize;
        totalSignSize += signCompressedSize;
    }

    organizedCompressedData.insert(organizedCompressedData.end(), bitsCounts.begin(), bitsCounts.end());
    organizedCompressedData.insert(organizedCompressedData.end(), compressedDataSizes.begin(), compressedDataSizes.end());
    organizedCompressedData.insert(organizedCompressedData.end(), realCompressedData.begin(), realCompressedData.end());

    auto compressEnd = std::chrono::high_resolution_clock::now();
    compressTime = std::chrono::duration<float>(compressEnd - compressStart).count();

    auto writeStart = std::chrono::high_resolution_clock::now();

    SZ3::writefile(unpredDataFileName.c_str(), allUnpredDataBuffers.data(), allUnpredDataBuffers.size());
    SZ3::writefile(compressedFileName.c_str(), organizedCompressedData.data(), organizedCompressedData.size());
    SZ3::writefile(signFileName.c_str(), compressedSign.data(), compressedSign.size());

    auto writeEnd = std::chrono::high_resolution_clock::now();
    writeTime = std::chrono::duration<float>(writeEnd - writeStart).count();

    std::cout << "[Time] Data Compression Time: " << compressTime << " seconds" << std::endl;
    std::cout << "[Time] File Write Time: " << writeTime << " seconds" << std::endl;

    return {totalCompSize, totalSignSize};
}



int main(int argc, char *argv[])
{
    std::string inputFileName;
    std::string datasetName = "native_fields/baryon_density";  // 新增: HDF5内部路径, 可用 --dataset_name 覆盖
    std::vector<size_t> blockShape;
    size_t beginStepNum = 0;
    size_t endStepNum = 0;

    float relative_error_bound = 1E-3;
    size_t extraValue = 0;

    // 解析命令行参数
    // 注意: --block_shape 的解析依赖 nDim, 而 nDim 现在来自 HDF5 文件本身,
    // 所以这里先用一个占位符方式: 要求用户显式传 --ndim 告诉程序维度数
    // (通常宇宙学数据都是3维, 直接写 --ndim 3 即可)
    size_t nDim = 0;

    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc)
        {
            inputFileName = argv[++i];
        }
        else if (arg == "--dataset_name" && i + 1 < argc)
        {
            datasetName = argv[++i];
        }
        else if (arg == "--ndim" && i + 1 < argc)
        {
            nDim = std::stoul(argv[++i]);
        }
        else if (arg == "--block_shape" && nDim > 0 && i + nDim < argc)
        {
            for (size_t j = 0; j < nDim; j++)
            {
                blockShape.push_back(std::stoul(argv[++i]));
            }
        }
        else if (arg == "--begin_step")
        {
            if (i + 1 < argc) beginStepNum = atoi(argv[i + 1]);
        }
        else if (arg == "--end_step")
        {
            if (i + 1 < argc) endStepNum = atoi(argv[i + 1]);
        }
        else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        }
    }

    if (inputFileName.empty() || nDim == 0 || blockShape.size() != nDim)
    {
        std::cerr << "Error: missing or invalid arguments." << std::endl;
        std::cerr << "Usage: " << argv[0]
                  << " --input_file <path.hdf5> --dataset_name <group/dataset> --ndim 3"
                  << " --block_shape 4 4 4 --relative_error 1e-3 0" << std::endl;
        return 1;
    }

    double totalSplitTime = 0.0;

    // === 直接从 HDF5 读取数据, 替代原来的 readBinaryFile ===
    std::vector<size_t> dataShape;
    std::vector<float> varData;
    try {
        varData = readHDF5Dataset(inputFileName, datasetName, dataShape);
    } catch (...) {
        std::cerr << "Failed to read HDF5 dataset. Aborting." << std::endl;
        return 1;
    }

    if (dataShape.size() != nDim) {
        std::cerr << "Error: --ndim " << nDim << " does not match actual HDF5 dataset rank "
                  << dataShape.size() << std::endl;
        return 1;
    }

    size_t totalElements = varData.size();

    float Global_min_value = *std::min_element(varData.begin(), varData.end());
    float Global_max_value = *std::max_element(varData.begin(), varData.end());
    float error_bound = relative_error_bound * (Global_max_value - Global_min_value);

    std::cout << "[Data Range] Global_min_value = " << Global_min_value 
          << ", Global_max_value = " << Global_max_value << std::endl;
          
    // 计算每个维度的块数量
    std::vector<size_t> blockCountOnEachDim(nDim);
    size_t total_blocks = 1;
    for (size_t i = 0; i < nDim; i++) {
        blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];
        total_blocks *= blockCountOnEachDim[i];
    }

    int blockSize = 1;
    for (const auto& dim : blockShape) blockSize *= dim;

    auto splitStart = std::chrono::high_resolution_clock::now();

    std::vector<float> flatBlockData;
    flatBlockData.reserve(totalElements);

    std::vector<std::vector<float>> blocks(total_blocks, std::vector<float>());

    for (size_t p = 0; p < totalElements; p++)
    {
        std::vector<size_t> elem_global_id = positionToIndices(p, dataShape);
        std::vector<size_t> block_global_id(nDim);
        for (size_t i = 0; i < nDim; i++)
        {
            block_global_id[i] = elem_global_id[i] / blockShape[i];
        }
        size_t block_position = indicesToPosition(blockCountOnEachDim, block_global_id);
        blocks[block_position].push_back(varData[p]);
    }

    for (size_t b = 0; b < total_blocks; b++) {
        flatBlockData.insert(flatBlockData.end(), blocks[b].begin(), blocks[b].end());
    }

    // 提前释放不再需要的内存
    std::vector<float>().swap(varData);
    std::vector<std::vector<float>>().swap(blocks);

    auto splitEnd = std::chrono::high_resolution_clock::now();
    totalSplitTime += std::chrono::duration<double>(splitEnd - splitStart).count();

    std::cout << "[Time] Total Split Time: " << totalSplitTime << " seconds" << std::endl;

    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string safeDatasetName = datasetName;
    std::replace(safeDatasetName.begin(), safeDatasetName.end(), '/', '_');

    std::string subDir = "/expanse/lustre/scratch/sdi/temp_project/nyx_original_compress/" + inputFileBaseName + "_" + safeDatasetName + "_"
                        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                        + std::to_string(extraValue) + "_hdf5_fixlength_originalDataCompression/";

    createDirectory(subDir);

    std::cout << "Error Bound Set: " << error_bound << std::endl;

    float thisCompressTime = 0.0;
    float thisWriteTime = 0.0;

    auto [totalCompSize, totalSignSize] = stmCompress_SZ3(flatBlockData.data(), flatBlockData.size(), blockSize, subDir, error_bound, thisCompressTime, thisWriteTime);

    std::cout << "Compressed Data Size: " << totalCompSize << std::endl;
    std::cout << "Sign Data Size: " << totalSignSize << std::endl;

    return 0;
}

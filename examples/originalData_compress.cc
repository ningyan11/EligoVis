#include <iostream>
#include <vector>
#include <cstdlib>
#include <adios2.h>
#include <cstddef>
#include <cstring>
#include <SZ3/api/sz.hpp>  
#include <chrono>

template<typename Func>
double benchmark(Func f) {
    auto start = std::chrono::high_resolution_clock::now();
    f(); // 执行访问模式
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

double continuous_access(const std::vector<double>& data) {
    volatile double sum = 0; // volatile 防止编译器优化掉
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i];
    }
    return sum;
}

#include <random>

double random_access(const std::vector<double>& data, double ratio) {
    size_t N = data.size();
    size_t M = static_cast<size_t>(N * ratio);

    std::vector<size_t> indices(M);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, N-1);
    for (size_t i = 0; i < M; i++) {
        indices[i] = dist(rng);
    }

    volatile double sum = 0;
    for (size_t i = 0; i < M; i++) {
        sum += data[indices[i]];
    }
    return sum;
}

double random_block_access(const std::vector<double>& data, size_t blockSize, double ratio) {
    size_t numBlocks = data.size() / blockSize;
    size_t blocksToAccess = static_cast<size_t>(numBlocks * ratio);

    // 随机选择 blocksToAccess 个块
    std::vector<size_t> blockIndices(blocksToAccess);
    std::mt19937 rng(42);  // 固定种子，保证可复现
    std::uniform_int_distribution<size_t> dist(0, numBlocks - 1);
    for (size_t i = 0; i < blocksToAccess; i++) {
        blockIndices[i] = dist(rng);
    }

    volatile double sum = 0;
    for (auto b : blockIndices) {
        size_t start = b * blockSize;
        for (size_t j = 0; j < blockSize; j++) {
            sum += data[start + j];  // 访问整块
        }
    }
    return sum;
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

// 计算数据范围
template <class T>
T data_range(const T *data, size_t num) {
    if (num == 0) {
        throw std::invalid_argument("Error: Data size is zero.");
    }

    T max = data[0];
    T min = data[0];
    for (size_t i = 1; i < num; ++i) {
        if (max < data[i]) max = data[i];
        if (min > data[i]) min = data[i];
    }

    T range = max - min;

    // 防止范围过小导致误差界限为零
    if (range < std::numeric_limits<T>::epsilon()) {
        //std::cerr << "Warning:ßß Data range too small, adjusted to epsilon." << std::endl;
        range = std::numeric_limits<T>::epsilon();
    }

    return range;
}

//保存（以binary)
void saveFlatBlockDataAsText(const std::vector<double>& flatBlockData, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // 写入每个 double 值，以二进制格式保存
    outFile.write(reinterpret_cast<const char*>(flatBlockData.data()), flatBlockData.size() * sizeof(double));

    outFile.close();
    if (!outFile.good()) {
        std::cerr << "Error occurred while writing flatBlockData to the binary file!" << std::endl;
    } else {
        std::cout << "flatBlockData has been saved as binary data to " << filename << std::endl;
    }
}



inline size_t stmCompress_SZ3(double *input, size_t numElements, int blockSize, int beginStepNum, int endStepNum) 
{
    // 创建文件名包含 step 范围

    std::string huffmanFileName = "./gray_onestep_d1e3/gray_d1e3_huffman_tree_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum);
    std::string compressedFileName = "./gray_onestep_d1e3/gray_d1e3_compressed_data_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + ".bin";

    std::vector<int> quant_inds(numElements);  // 存储量化索引
    std::size_t pos = 0;
    std::size_t csize = 0;                     // 每个块的压缩大小
    std::size_t totalCompSize = 0;

    // 设置压缩缓冲区大小为 2 倍的块大小，用于每个块的压缩数据
    size_t bufferSize = 2 * sizeof(double) * blockSize;
    auto *buffer = new SZ3::uchar[bufferSize];
    auto *buffer_zstd = new SZ3::uchar[bufferSize];

     // 创建一个 vector 来保存所有块的元数据
    std::vector<std::vector<SZ3::uchar>> allMetadataBuffers;

    // 量化数据，准备 Huffman Tree
    while (pos < numElements) {
        size_t numToCompress = blockSize;
        if (numToCompress + pos > numElements) numToCompress = numElements - pos;

        std::vector<double> data_cpy(&input[pos], &input[pos + numToCompress]);

        //改为3D compression
        //SZ3::Config conf(4, 4, 4);
        SZ3::Config conf(numToCompress);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = 1e-3;
        
        //SZ3::calAbsErrorBound(conf, data_cpy.data());
        

        std::vector<int> quant_inds_block;
        auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);
        quant_inds_block = decompose.compress(conf, data_cpy.data());


        // 创建缓冲区以保存元数据
        size_t metadataSize = 2 * sizeof(double) * blockSize;
        std::vector<SZ3::uchar> metadataBuffer(metadataSize);

        // 保存元数据到缓冲区
        auto bufferp = metadataBuffer.data();
        decompose.save(bufferp);
        size_t realMetadatasize = bufferp - metadataBuffer.data();
        //metadataBuffer.resize(realMetadatasize); // 调整 metadataBuffer 大小为实际元数据大小

        //提取 unpred.size() 和 unpred.data() 的位置
        const SZ3::uchar *tempBuffer = metadataBuffer.data();

        // 跳过第一个字节（标志位）
        tempBuffer += 1;

        // 跳过 error_bound (double 类型)
        tempBuffer += sizeof(double);

        // 跳过 radius (int 类型)
        tempBuffer += sizeof(int);

        // 提取 unpred.size()
        size_t unpredSize = *reinterpret_cast<const size_t *>(tempBuffer);
        tempBuffer += sizeof(size_t);

        // 提取 unpred.data() (double 数组)
        const double *unpredData = reinterpret_cast<const double *>(tempBuffer);

        // 4. 创建新的缓冲区，只保存 unpred.size() 和 unpred.data()
        std::vector<SZ3::uchar> modifiedMetadataBuffer(sizeof(size_t) + unpredSize * sizeof(double));
        SZ3::uchar *modifiedBufferPointer = modifiedMetadataBuffer.data();

        // 写入 unpred.size() 到新的缓冲区
        *reinterpret_cast<size_t *>(modifiedBufferPointer) = unpredSize;
        modifiedBufferPointer += sizeof(size_t);

        // 写入 unpred.data() 到新的缓冲区
        memcpy(modifiedBufferPointer, unpredData, unpredSize * sizeof(double));

        //替换原来的 metadataBuffer，释放多余的内存
        metadataBuffer = std::move(modifiedMetadataBuffer);

         // 将当前块的 metadataBuffer 添加到 allMetadataBuffers
        allMetadataBuffers.push_back(std::move(metadataBuffer));


        memcpy(&quant_inds[pos], quant_inds_block.data(), sizeof(int) * numToCompress);

        pos += numToCompress;
    }

    std::cout << "Total metadata buffers: " << allMetadataBuffers[0].size() << std::endl;

    pos = 0;

    // 设置 Huffman 树的缓冲区大小为原始数据大小的两倍
    size_t huffmanBufferSize = 2 * sizeof(double) * numElements;
    auto *huffmanBuffer = new SZ3::uchar[huffmanBufferSize];
    SZ3::uchar *buffer_pos = huffmanBuffer;


    // 创建并存储 Huffman Tree
    SZ3::HuffmanEncoder<int> huffman;
    huffman.preprocess_encode(quant_inds, 0);

    huffman.save(buffer_pos);
    // 将 Huffman 树写入文件
    SZ3::writefile(huffmanFileName.c_str(), huffmanBuffer, buffer_pos - huffmanBuffer);
    
    // 删除 Huffman Tree 缓冲区，释放内存
    delete[] huffmanBuffer;

    // 用于存储每个块的大小
    std::vector<uint8_t> compressedSizes;
    std::vector<uint8_t> compressedData;

    size_t metadataIndex = 0;
    // 逐块编码和压缩
    while (pos < numElements) {
        size_t numToCompress = blockSize;
        if (numToCompress + pos > numElements) numToCompress = numElements - pos;

        SZ3::Lossless_zstd zstd;
        SZ3::uchar *buffer_pos = buffer;

        huffman.encode(&quant_inds[pos], numToCompress, buffer_pos);
        
        size_t metadataSize = allMetadataBuffers[metadataIndex].size();

        // 将当前块的元数据附加到编码后的量化数据之后
        std::memcpy(buffer_pos, allMetadataBuffers[metadataIndex].data(), allMetadataBuffers[metadataIndex].size());
        buffer_pos += allMetadataBuffers[metadataIndex].size();


        csize = zstd.compress(buffer, buffer_pos - buffer, buffer_zstd, bufferSize);

        if (metadataIndex == 0) {
        std::cout << "before Zstd compress: " << (buffer_pos - buffer) << " bytes" << std::endl;
        std::cout << "Compressed size for current block: " << csize << " bytes" << std::endl;}


        compressedSizes.push_back(static_cast<uint8_t>((csize >> 8) & 0xFF));
        compressedSizes.push_back(static_cast<uint8_t>(csize & 0xFF));

        compressedSizes.push_back(static_cast<uint8_t>((metadataSize >> 8) & 0xFF));
        compressedSizes.push_back(static_cast<uint8_t>(metadataSize & 0xFF));





        // 将压缩后的块数据追加到 compressedData
        compressedData.insert(compressedData.end(), buffer_zstd, buffer_zstd + csize);

        pos += numToCompress;
        totalCompSize += csize;
        metadataIndex++; 
    }

    // 将 compressedSizes 数据插入到 compressedData 的开头
    compressedData.insert(compressedData.begin(), compressedSizes.begin(), compressedSizes.end());

    delete[] buffer;
    delete[] buffer_zstd;
    // 使用 SZ3::writefile 将 compressedData 写入文件
    SZ3::writefile(compressedFileName.c_str(), compressedData.data(), compressedData.size());
    return totalCompSize;
}



int main(int argc, char *argv[])
{
    std::string inputFileName;
    std::string variableName;
    std::string variableType;
    size_t nDim = 0;
    std::vector<size_t> blockShape;
    size_t beginStepNum = 0;
    size_t endStepNum = 0;

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


    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);

    // 定义 flatBlockData 用于保存所有 step 的数据
    std::vector<double> flatBlockData;
    size_t step = 0;

    while (reader_engine.BeginStep() == adios2::StepStatus::OK)
    {
        if (step < beginStepNum) {
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

        }

        reader_engine.EndStep();
        step++;
        if (step > endStepNum) {
            break;
        }
    }

    reader_engine.Close();


    /*std::cout << "Testing block-level memory access efficiency..." << std::endl;

    double time_full = benchmark([&](){
        random_block_access(flatBlockData, blockSize, 1.0);
    });
    std::cout << "Full block sequential access time = " << time_full << " s" << std::endl;

    for (double ratio : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        double t = benchmark([&](){
            random_block_access(flatBlockData, blockSize, ratio);
        });
        std::cout << "Block ratio " << ratio
                << " -> " << t << " s, normalized = " << (t / time_full) << std::endl;
    }



    // 将数据保存为可读文本文件（打平的原始数据）
    //std::string outputFileName = "original_data_readable.txt";
    //saveFlatBlockDataAsText(flatBlockData, outputFileName);*/


    //Test
    size_t totalSize = stmCompress_SZ3(flatBlockData.data(), flatBlockData.size(), blockSize, beginStepNum, endStepNum);

    // 输出压缩结果的信息
    std::cout << "Compressed Data Size : " << totalSize << std::endl;

    return 0;
}







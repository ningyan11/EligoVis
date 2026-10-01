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
#include <vector>
#include <string>
#include <filesystem>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <zstd.h>
#include <scidx_BytesToolkit.h>
#include <unordered_map>
#include <array>
#include "../miniIsosurface/marchingCubes/util/util.h"
#include "../miniIsosurface/marchingCubes/util/MarchingCubesTables.h"
#include "../miniIsosurface/marchingCubes/util/TriangleMesh.h"
#include "../miniIsosurface/marchingCubes/util/SaveTriangleMesh.h"

std::vector<unsigned char> loadHuffmanTree(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open " << filename << std::endl;
        return {};
    }
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::vector<double> decompress_SZ3_default(
    const std::string& subDir, size_t totalBlocks, size_t blockSize, double error_bound) {

    using namespace SZ3;

    std::string unpredDataFileName = subDir + "/fixed_unpredData.bin";
    std::string compressedFileName = subDir + "/fixed_compressed_data.bin";
    std::string HuffmanFileName = subDir + "/huffman.bin";

    // 1. Load Huffman tree
    std::vector<unsigned char> huffmanOut = loadHuffmanTree(HuffmanFileName);
    int stateNum = 2 * 16384;
    scidx::HuffmanTree* decodeHuffmanTree = scidx::createHuffmanTree(stateNum);
    size_t nodeCount = scidx::bytesToInt_bigEndian(huffmanOut.data());
    scidx::node root = reconstruct_HuffTree_from_bytes_anyStates(
        decodeHuffmanTree, huffmanOut.data() + 8, nodeCount);

    std::cout << "[DEBUG] Huffman tree loaded." << std::endl;

    // 2. Load metadata
    size_t metaSize = 0;
    auto metaCombined = readfile<uchar>(unpredDataFileName.c_str(), metaSize);

    std::vector<size_t> metaSizes(totalBlocks);
    for (size_t i = 0; i < totalBlocks; ++i) {
        std::memcpy(&metaSizes[i], metaCombined.get() + i * sizeof(size_t), sizeof(size_t));
    }

    const uchar* metaDataStart = metaCombined.get() + totalBlocks * sizeof(size_t);

    for (size_t i = 0; i < std::min(totalBlocks, size_t(10)); ++i) {
        std::cout << "[DEBUG] metaSizes[" << i << "] = " << metaSizes[i] << std::endl;
    }

    // 3. Load compressed Huffman data
    size_t compSize = 0;
    auto compCombined = readfile<uchar>(compressedFileName.c_str(), compSize);

    std::vector<size_t> compSizes(totalBlocks);
    for (size_t i = 0; i < totalBlocks; ++i) {
        std::memcpy(&compSizes[i], compCombined.get() + i * sizeof(size_t), sizeof(size_t));
    }

    const uchar* compDataStart = compCombined.get() + totalBlocks * sizeof(size_t);

    std::vector<int> all_quant_inds;
    all_quant_inds.reserve(totalBlocks * blockSize); // 提前分配避免扩容
    size_t compOffset = 0;

    std::cout << "[DEBUG] before zstd " << std::endl;

    for (size_t b = 0; b < totalBlocks; ++b) {
        size_t encodedSize = compSizes[b];
        const uchar* encodedData = compDataStart + compOffset;

        std::vector<uchar> decodedBuffer(blockSize * sizeof(double) * 40);  // 宽裕分配
        uchar* decodedPtr = decodedBuffer.data();
        size_t decodedLen = decodedBuffer.size();

        SZ3::Lossless_zstd().decompress(encodedData, encodedSize, decodedPtr, decodedLen);

        std::vector<int> block_quant_inds(blockSize);
        scidx::decode(decodedPtr, blockSize, root, block_quant_inds.data());

        all_quant_inds.insert(all_quant_inds.end(), block_quant_inds.begin(), block_quant_inds.end());

        compOffset += encodedSize;
    }

    std::cout << "[DEBUG] after all zstd" << std::endl;
    std::cout << "[DEBUG] after huffman" << std::endl;
    std::cout << "[DEBUG] all_quant_inds.size() = " << all_quant_inds.size()
              << ", expected = " << totalBlocks * blockSize << std::endl;

    // 4. Dequantize each block
    std::vector<double> allData;
    allData.reserve(totalBlocks * blockSize);  // 减少内存扩容
    size_t metaOffset = 0;
    size_t qpos = 0;

    for (size_t b = 0; b < totalBlocks; ++b) {
        size_t mSize = metaSizes[b];
        size_t remaining = mSize; 
        for (size_t i = 0; i < std::min(totalBlocks, size_t(10)); ++i) {
            std::cout << "[DEBUG] metaSizes[" << i << "] = " << metaSizes[i] << std::endl;
        }
        const uchar* metaPtr = metaDataStart + metaOffset;

        Config conf(blockSize);
        conf.cmprAlgo = ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;

        auto quantizer = LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);
        decompose.load(metaPtr, remaining);

        std::vector<int> block_inds(all_quant_inds.begin() + qpos, all_quant_inds.begin() + qpos + blockSize);
        std::vector<double> blockData(blockSize);

        std::cout << "[DEBUG] Decompress block " << b
                  << " | qpos = " << qpos
                  << " | blockSize = " << blockSize
                  << " | metaOffset = " << metaOffset << std::endl;

        decompose.decompress(conf, block_inds, blockData.data());

        allData.insert(allData.end(), blockData.begin(), blockData.end());

        metaOffset += mSize;  
        qpos += blockSize;
    }

    return allData;
}


std::vector<double> decompress_SZ3_whole(
    const std::string& subDir, size_t totalBlocks, size_t blockSize, double error_bound)
{
    std::string compressedFileName = subDir + "/sz3_whole_compressed.bin";

    size_t cmpSize = 0;
    auto cmpData = SZ3::readfile<char>(compressedFileName.c_str(), cmpSize);

    SZ3::Config conf;
    double* decData = nullptr;
    SZ_decompress<double>(conf, cmpData.get(), cmpSize, decData);

    size_t numElements = totalBlocks * blockSize;
    std::vector<double> result(decData, decData + numElements);
    delete[] decData;
    return result;
}









double calculatePSNR(const std::vector<double>& flatOriginal,
    const std::vector<double>& flatDecompressed) {
    // 检查尺寸一致
    if (flatOriginal.size() != flatDecompressed.size()) {
    throw std::runtime_error("Size mismatch between original and decompressed data.");
    }

    // 计算 MSE（均方误差）
    double mse = 0.0;
    for (size_t i = 0; i < flatOriginal.size(); ++i) {
    double diff = flatOriginal[i] - flatDecompressed[i];
    mse += diff * diff;
    }
    mse /= flatOriginal.size();

    // 计算 MAX
    double max_val = *std::max_element(flatOriginal.begin(), flatOriginal.end());

    // 计算 PSNR
    if (mse == 0.0) {
    return std::numeric_limits<double>::infinity();  // 表示无损还原
    }
    return 10.0 * std::log10((max_val * max_val) / mse);
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


void RunAndSaveIsosurfaceFromFlatVolume(
    const std::vector<double>& volume,
    size_t nx, size_t ny, size_t nz,
    double isovalue,
    const std::string& outFile)
{
    auto val = [&](size_t x, size_t y, size_t z) -> double {
        return volume[x + y * nx + z * nx * ny];
    };

    std::vector<std::array<double, 3>> globalPoints;
    std::vector<std::array<double, 3>> globalNormals;
    std::vector<std::array<int, 3>>    globalTriangles;
    std::unordered_map<size_t, int>    globalPointMap;
    int ptIdx = 0;

    for (size_t z = 0; z < nz - 1; ++z) {
        for (size_t y = 0; y < ny - 1; ++y) {
            for (size_t x = 0; x < nx - 1; ++x) {
                std::array<double, 8> cubeValues = {{
                    val(x,   y,   z),   val(x+1, y,   z),
                    val(x+1, y+1, z),   val(x,   y+1, z),
                    val(x,   y,   z+1), val(x+1, y,   z+1),
                    val(x+1, y+1, z+1), val(x,   y+1, z+1)
                }};
                int cellCaseId = util::findCaseId(cubeValues, isovalue);
                if (cellCaseId == 0 || cellCaseId == 255) continue;

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

                std::array<std::array<double, 3>, 8> cubeGradients;
                for (int i = 0; i < 8; ++i) {
                    size_t px = x + (i & 1);
                    size_t py = y + ((i >> 1) & 1);
                    size_t pz = z + ((i >> 2) & 1);
                    std::array<double, 3> grad = {0.0, 0.0, 0.0};
                    grad[0] = (px == 0)    ? val(px,py,pz) - val(px+1,py,pz)
                            : (px == nx-1) ? val(px-1,py,pz) - val(px,py,pz)
                            : (val(px-1,py,pz) - val(px+1,py,pz)) / 2.0;
                    grad[1] = (py == 0)    ? val(px,py,pz) - val(px,py+1,pz)
                            : (py == ny-1) ? val(px,py-1,pz) - val(px,py,pz)
                            : (val(px,py-1,pz) - val(px,py+1,pz)) / 2.0;
                    grad[2] = (pz == 0)    ? val(px,py,pz) - val(px,py,pz+1)
                            : (pz == nz-1) ? val(px,py,pz-1) - val(px,py,pz)
                            : (val(px,py,pz-1) - val(px,py,pz+1)) / 2.0;
                    cubeGradients[i] = grad;
                }

                const int* triEdges = util::caseTrianglesEdges[cellCaseId];
                for (; *triEdges != -1; triEdges += 3) {
                    std::array<int, 3> tri;
                    bool valid = true;
                    for (int i = 0; i < 3; ++i) {
                        int edgeIdx = triEdges[i];
                        size_t globalEdgeIdx = ((z*(ny-1)+y)*(nx-1)+x)*12 + edgeIdx;
                        auto it = globalPointMap.find(globalEdgeIdx);
                        if (it != globalPointMap.end()) {
                            tri[i] = it->second;
                        } else {
                            const int* vs = util::edgeVertices[edgeIdx];
                            double denom = cubeValues[vs[1]] - cubeValues[vs[0]];
                            if (std::abs(denom) < 1e-12) { valid = false; break; }
                            double w = (isovalue - cubeValues[vs[0]]) / denom;
                            globalPoints.push_back(util::interpolate(cubePositions[vs[0]], cubePositions[vs[1]], w));
                            globalNormals.push_back(util::interpolate(cubeGradients[vs[0]], cubeGradients[vs[1]], w));
                            globalPointMap[globalEdgeIdx] = ptIdx;
                            tri[i] = ptIdx++;
                        }
                    }
                    if (valid && tri[0]!=tri[1] && tri[1]!=tri[2] && tri[2]!=tri[0])
                        globalTriangles.push_back(tri);
                }
            }
        }
    }

    util::TriangleMesh<double> mesh(globalPoints, globalNormals, globalTriangles);
    std::cout << "[ISO] vertices=" << mesh.numberOfVertices()
              << " triangles=" << mesh.numberOfTriangles() << std::endl;
    util::saveTriangleMesh(mesh, outFile.c_str());
    std::cout << "[ISO] Written: " << outFile << std::endl;
}

std::vector<double> decompress_SZ3_blockwise(
    const std::string& subDir, size_t totalBlocks, size_t blockSize, double error_bound)
{
    std::string compressedFileName = subDir + "/sz3_blockwise_compressed.bin";

    size_t cmpSize = 0;
    auto cmpData = SZ3::readfile<char>(compressedFileName.c_str(), cmpSize);

    SZ3::Config conf;
    double* decData = nullptr;
    SZ_decompress<double>(conf, cmpData.get(), cmpSize, decData);

    size_t numElements = totalBlocks * blockSize;
    std::vector<double> result(decData, decData + numElements);
    delete[] decData;
    return result;
}



int main(int argc, char* argv[]) {


    std::string inputFileName, variableName, indexDir;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0, bigBlocksPerStep = 1, smallBlocksPerBig = 1 ,  smallBlockSize = 1 , totalBlocksNumber = 1, smallBlocksPerStep = 1; 
    std::vector<double> queryRange;
    double relative_error_bound = 1E-3;
    size_t extraValue =0;
    std::string variableType;
    


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

    //index所在文件夹的根路径
    indexDir = "/home/nyan/scidx/scidx/" + std::filesystem::path(inputFileName).filename().string() + "_" + 
                std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) + "_index/";


    /*std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string mySubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_SZ3_bp_originalDataCompression/rank_0/";*/

    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string mySubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" +safeVarName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_blocksz3_originalDataCompression/";

    
    
    std::cout << "Looking for decompression files in: " << mySubDir << std::endl;

            

    std::vector<ScidxInterval<double>> allBigBlockIndices = loadBigBlockIndexFile(indexDir + "big_block_minmax");

    double globalMin = std::numeric_limits<double>::max();
    double globalMax = std::numeric_limits<double>::lowest();

    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }

    double error_bound = relative_error_bound * (globalMax - globalMin);
    std::cout << "Computed error_bound: " << error_bound << std::endl;
    


    std::vector<size_t> allGlobalSmallBlockIds(totalBlocksNumber);  
    for (size_t i = 0; i < totalBlocksNumber; ++i) {
        allGlobalSmallBlockIds[i] = i;
    }

   
    size_t myLocalBlockCount = allGlobalSmallBlockIds.size();

    std::cout << "[DEBUG] mySubDir = " << mySubDir << std::endl;
    std::cout << "[DEBUG] myLocalBlockCount = " << myLocalBlockCount << std::endl;
    std::cout << "[DEBUG] smallBlockSize = " << smallBlockSize << std::endl;
    std::cout << "[DEBUG] error_bound = " << error_bound << std::endl;



    auto data = decompress_SZ3_blockwise(mySubDir, myLocalBlockCount, smallBlockSize, error_bound);



    std::cout << "decompress_SZ3_default size: " << data.size() << std::endl;


    std::string isoFile =
    "/expanse/lustre/scratch/sdi/temp_project/isosurface_sz3block_step_" +
    std::to_string(beginStepNum) + "_iso_0.04.vtk";

RunAndSaveIsosurfaceFromFlatVolume(data, 1024, 1024, 1024, 0.04, isoFile);

    /*std::vector<size_t> blockShape = smallBlockShape;
    
    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);

    // 定义 flatBlockData 用于保存所有 step 的数据
    //全部step，按照一维小块的顺序排列
    std::vector<double> flatBlockData;
    size_t step = 0;

    double totalSplitTime = 0.0;

    std::set<size_t> targetSet;
    targetSet.insert(beginStepNum);




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

    double psnr = calculatePSNR(flatBlockData, data);
    std::cout << "PSNR: " << psnr << " dB" << std::endl;*/

    return 0;
}

#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <array>
#include <zfp.h>
#include <scidx_avl.h>
#include <scidx_block_min_max.h>
#include <scidx_avl_interval_tree.h>
#include "../miniIsosurface/marchingCubes/util/util.h"
#include "../miniIsosurface/marchingCubes/util/MarchingCubesTables.h"
#include "../miniIsosurface/marchingCubes/util/TriangleMesh.h"
#include "../miniIsosurface/marchingCubes/util/SaveTriangleMesh.h"

// ============================================================
// 从 index 目录读取 big_block_minmax，用于还原 error_bound
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
// ZFP 解压函数
//   读取 meta.bin + compressed_data.bin，还原所有 step 数据
// ============================================================
std::vector<double> decompress_ZFP(
    const std::string& subDir,
    double error_bound)
{
    std::string dataFile = subDir + "/compressed_data.bin";
    std::string metaFile = subDir + "/meta.bin";

    // ----------------------------------------------------------
    // 1. 读取元数据
    //    格式：[stepCount][compressedSizes x stepCount]
    //          [nDim][shapes x stepCount*nDim]
    // ----------------------------------------------------------
    std::ifstream metaIn(metaFile, std::ios::binary);
    if (!metaIn)
        throw std::runtime_error("[decompress_ZFP] Cannot open: " + metaFile);

    size_t stepCount = 0;
    metaIn.read(reinterpret_cast<char*>(&stepCount), sizeof(size_t));

    std::vector<size_t> compressedSizes(stepCount);
    metaIn.read(reinterpret_cast<char*>(compressedSizes.data()),
                stepCount * sizeof(size_t));

    size_t nDim = 0;
    metaIn.read(reinterpret_cast<char*>(&nDim), sizeof(size_t));

    std::vector<std::vector<size_t>> allShapes(stepCount, std::vector<size_t>(nDim));
    for (size_t s = 0; s < stepCount; ++s)
        metaIn.read(reinterpret_cast<char*>(allShapes[s].data()),
                    nDim * sizeof(size_t));
    metaIn.close();

    std::cout << "[decompress_ZFP] stepCount=" << stepCount
              << "  nDim=" << nDim
              << "  error_bound=" << error_bound << "\n";
    for (size_t s = 0; s < stepCount; ++s) {
        std::cout << "[decompress_ZFP] Step " << s << " shape=(";
        for (size_t d = 0; d < nDim; ++d)
            std::cout << allShapes[s][d] << (d+1<nDim ? "," : "");
        std::cout << ")  compressedSize=" << compressedSizes[s] << " bytes\n";
    }

    // ----------------------------------------------------------
    // 2. 逐 step 读取压缩数据并解压
    // ----------------------------------------------------------
    std::ifstream dataIn(dataFile, std::ios::binary);
    if (!dataIn)
        throw std::runtime_error("[decompress_ZFP] Cannot open: " + dataFile);

    std::vector<double> allData;

    for (size_t sIdx = 0; sIdx < stepCount; ++sIdx)
    {
        const auto& shape = allShapes[sIdx];
        size_t nElem = 1;
        for (auto s : shape) nElem *= s;

        // 读入该 step 的压缩字节
        size_t cmpSize = compressedSizes[sIdx];
        std::vector<char> cmpBuf(cmpSize);
        dataIn.read(cmpBuf.data(), cmpSize);
        if (!dataIn)
            throw std::runtime_error(
                "[decompress_ZFP] Unexpected EOF at step " + std::to_string(sIdx));

        std::vector<double> stepData(nElem);

        // 维度顺序反转：C行优先 shape[nDim-1] 最快 → ZFP nx=最快
        zfp_field* field = nullptr;
        if (nDim == 1)
            field = zfp_field_1d(stepData.data(), zfp_type_double,
                                 shape[0]);
        else if (nDim == 2)
            field = zfp_field_2d(stepData.data(), zfp_type_double,
                                 shape[1],   // nx = 最快
                                 shape[0]);  // ny
        else if (nDim == 3)
            field = zfp_field_3d(stepData.data(), zfp_type_double,
                                 shape[2],   // nx = 最快
                                 shape[1],   // ny
                                 shape[0]);  // nz = 最慢
        else
            throw std::runtime_error("[decompress_ZFP] Unsupported nDim=" +
                                     std::to_string(nDim));

        if (!field)
            throw std::runtime_error(
                "[decompress_ZFP] zfp_field_Nd failed at step " + std::to_string(sIdx));

        // 误差界必须与压缩时完全一致
        zfp_stream* zfp = zfp_stream_open(nullptr);
        zfp_stream_set_accuracy(zfp, error_bound);

        bitstream* bs = stream_open(cmpBuf.data(), cmpSize);
        zfp_stream_set_bit_stream(zfp, bs);
        zfp_stream_rewind(zfp);

        size_t ret = zfp_decompress(zfp, field);
        if (ret == 0)
            throw std::runtime_error(
                "[decompress_ZFP] zfp_decompress returned 0 at step " +
                std::to_string(sIdx));

        stream_close(bs);
        zfp_field_free(field);
        zfp_stream_close(zfp);

        allData.insert(allData.end(), stepData.begin(), stepData.end());

        std::cout << "[decompress_ZFP] Step " << sIdx
                  << "  nElem=" << nElem
                  << "  cmpSize=" << cmpSize << " bytes  done\n";
    }

    dataIn.close();
    std::cout << "[decompress_ZFP] Total elements restored: " << allData.size() << "\n";
    return allData;
}

// ============================================================
// 等值面提取（与原代码保持一致）
// ============================================================
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

    for (size_t z = 0; z < nz - 1; ++z)
    for (size_t y = 0; y < ny - 1; ++y)
    for (size_t x = 0; x < nx - 1; ++x)
    {
        std::array<double, 8> cubeValues = {{
            val(x,y,z),       val(x+1,y,z),
            val(x+1,y+1,z),   val(x,y+1,z),
            val(x,y,z+1),     val(x+1,y,z+1),
            val(x+1,y+1,z+1), val(x,y+1,z+1)
        }};
        int cellCaseId = util::findCaseId(cubeValues, isovalue);
        if (cellCaseId == 0 || cellCaseId == 255) continue;

        std::array<std::array<double,3>,8> cubePositions = {{
            {double(x),  double(y),  double(z)},
            {double(x+1),double(y),  double(z)},
            {double(x+1),double(y+1),double(z)},
            {double(x),  double(y+1),double(z)},
            {double(x),  double(y),  double(z+1)},
            {double(x+1),double(y),  double(z+1)},
            {double(x+1),double(y+1),double(z+1)},
            {double(x),  double(y+1),double(z+1)}
        }};

        std::array<std::array<double,3>,8> cubeGradients;
        for (int i = 0; i < 8; ++i) {
            size_t px = x+(i&1), py = y+((i>>1)&1), pz = z+((i>>2)&1);
            std::array<double,3> grad = {0.0, 0.0, 0.0};
            grad[0] = (px==0)    ? val(px,py,pz)-val(px+1,py,pz)
                    : (px==nx-1) ? val(px-1,py,pz)-val(px,py,pz)
                    : (val(px-1,py,pz)-val(px+1,py,pz))/2.0;
            grad[1] = (py==0)    ? val(px,py,pz)-val(px,py+1,pz)
                    : (py==ny-1) ? val(px,py-1,pz)-val(px,py,pz)
                    : (val(px,py-1,pz)-val(px,py+1,pz))/2.0;
            grad[2] = (pz==0)    ? val(px,py,pz)-val(px,py,pz+1)
                    : (pz==nz-1) ? val(px,py,pz-1)-val(px,py,pz)
                    : (val(px,py,pz-1)-val(px,py,pz+1))/2.0;
            cubeGradients[i] = grad;
        }

        const int* triEdges = util::caseTrianglesEdges[cellCaseId];
        for (; *triEdges != -1; triEdges += 3) {
            std::array<int,3> tri;
            bool valid = true;
            for (int i = 0; i < 3; ++i) {
                int edgeIdx = triEdges[i];
                size_t globalEdgeIdx = ((z*(ny-1)+y)*(nx-1)+x)*12+edgeIdx;
                auto it = globalPointMap.find(globalEdgeIdx);
                if (it != globalPointMap.end()) {
                    tri[i] = it->second;
                } else {
                    const int* vs = util::edgeVertices[edgeIdx];
                    double denom = cubeValues[vs[1]]-cubeValues[vs[0]];
                    if (std::abs(denom)<1e-12) { valid=false; break; }
                    double w = (isovalue-cubeValues[vs[0]])/denom;
                    globalPoints.push_back(util::interpolate(cubePositions[vs[0]],cubePositions[vs[1]],w));
                    globalNormals.push_back(util::interpolate(cubeGradients[vs[0]],cubeGradients[vs[1]],w));
                    globalPointMap[globalEdgeIdx] = ptIdx;
                    tri[i] = ptIdx++;
                }
            }
            if (valid && tri[0]!=tri[1] && tri[1]!=tri[2] && tri[2]!=tri[0])
                globalTriangles.push_back(tri);
        }
    }

    util::TriangleMesh<double> mesh(globalPoints, globalNormals, globalTriangles);
    std::cout << "[ISO] vertices=" << mesh.numberOfVertices()
              << " triangles=" << mesh.numberOfTriangles() << std::endl;
    util::saveTriangleMesh(mesh, outFile.c_str());
    std::cout << "[ISO] Written: " << outFile << std::endl;
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[])
{
    // ----------------------------------------------------------
    // 参数解析（与 file 8 完全一致）
    // ----------------------------------------------------------
    std::string inputFileName, variableName, indexDir;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0;
    size_t smallBlockSize = 1, smallBlocksPerStep = 1, totalBlocksNumber = 1;
    double relative_error_bound = 1E-3;
    size_t extraValue = 0;
    std::string variableType;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc) {
            inputFileName = argv[++i];
        } else if (arg == "--variable_name" && i + 1 < argc) {
            variableName = argv[++i];
        } else if (arg == "--dimensions" && i + 1 < argc) {
            nDim = std::stoul(argv[++i]);
        } else if (arg == "--begin_step" && i + 1 < argc) {
            beginStepNum = std::stoul(argv[++i]);
        } else if (arg == "--end_step" && i + 1 < argc) {
            endStepNum = std::stoul(argv[++i]);
        } else if (arg == "--stepData_shape" && i + nDim < argc) {
            for (size_t j = 0; j < nDim; j++)
                stepDataShape.push_back(std::stoul(argv[++i]));
        } else if (arg == "--small_block_shape") {
            if (nDim) {
                if ((int)(i + nDim) < argc) {
                    for (size_t j = i + 1; j < i + 1 + nDim; j++)
                        smallBlockShape.push_back(atoi(argv[j]));
                } else {
                    std::cerr << "--small_block_shape option requires [# of dimensions] argument.\n";
                    return 1;
                }
            }
        } else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        }
    }

    for (size_t d = 0; d < nDim; d++) {
        smallBlocksPerStep *= stepDataShape[d] / smallBlockShape[d];
        smallBlockSize     *= smallBlockShape[d];
    }
    totalBlocksNumber = smallBlocksPerStep;  // nSteps = 1

    // ----------------------------------------------------------
    // index 目录（用于读取 global min/max → error_bound）
    // ----------------------------------------------------------
    indexDir = "/home/nyan/scidx/scidx/"
             + std::filesystem::path(inputFileName).filename().string() + "_"
             + std::to_string(beginStepNum) + "_"
             + std::to_string(endStepNum)   + "_"
             + std::to_string(extraValue)   + "_index/";

    // ----------------------------------------------------------
    // ZFP 压缩文件目录（与压缩程序保持完全一致）
    // ----------------------------------------------------------
    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName =
        std::filesystem::path(inputFileName).filename().string();
    std::string mySubDir = "/home/nyan/scidx/scidx/"
                         + inputFileBaseName + "_" + safeVarName + "_"
                         + std::to_string(beginStepNum) + "_"
                         + std::to_string(endStepNum)   + "_"
                         + std::to_string(extraValue)
                         + "_ZFP_autoBlock_2/";

    std::cout << "Looking for decompression files in: " << mySubDir << "\n";

    // ----------------------------------------------------------
    // 计算 error_bound（与压缩时逻辑保持一致）
    // ----------------------------------------------------------
    std::vector<ScidxInterval<double>> allBigBlockIndices =
        loadBigBlockIndexFile(indexDir + "big_block_minmax");

    double globalMin = std::numeric_limits<double>::max();
    double globalMax = std::numeric_limits<double>::lowest();
    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }

    double error_bound = relative_error_bound * (globalMax - globalMin);
    std::cout << "Computed error_bound: " << error_bound << "\n";

    std::cout << "[DEBUG] mySubDir      = " << mySubDir << "\n"
              << "[DEBUG] totalBlocks   = " << totalBlocksNumber << "\n"
              << "[DEBUG] smallBlockSize= " << smallBlockSize << "\n"
              << "[DEBUG] error_bound   = " << error_bound << "\n";

    // ----------------------------------------------------------
    // ZFP 解压
    // ----------------------------------------------------------
    auto data = decompress_ZFP(mySubDir, error_bound);

    std::cout << "decompress_ZFP size: " << data.size() << "\n";

    // ----------------------------------------------------------
    // 输出解压后数据的总体范围
    // ----------------------------------------------------------
    if (!data.empty()) {
        double decMin = *std::min_element(data.begin(), data.end());
        double decMax = *std::max_element(data.begin(), data.end());

        // 计算均值和标准差
        double sum = 0.0;
        for (double v : data) sum += v;
        double mean = sum / data.size();

        double varAcc = 0.0;
        for (double v : data) {
            double d = v - mean;
            varAcc += d * d;
        }
        double stddev = std::sqrt(varAcc / data.size());

        std::cout << "\n===== Decompressed Data Statistics =====\n"
                  << "  Total elements : " << data.size() << "\n"
                  << "  Min value      : " << decMin << "\n"
                  << "  Max value      : " << decMax << "\n"
                  << "  Value range    : " << (decMax - decMin) << "\n"
                  << "  Mean           : " << mean << "\n"
                  << "  Std dev        : " << stddev << "\n"
                  << "========================================\n";
    }

    // ----------------------------------------------------------
    // 等值面提取
    // ----------------------------------------------------------
    std::string isoFile =
        "/expanse/lustre/scratch/sdi/temp_project/isosurface_zfp_step_"
        + std::to_string(beginStepNum) + "_iso_0.04_2.vtk";

    RunAndSaveIsosurfaceFromFlatVolume(data, 1024, 1024, 1024, 0.04, isoFile);

    return 0;
}
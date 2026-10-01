#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <random>
#include <set>
#include <adios2.h> 
#include <scidx_avl.h> 
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <utility> 
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem> 
#include <mpi.h>
#include <cstring>
#include <unordered_map>
#include <array>
#include <unordered_map>
#include <array>
#include "../miniIsosurface/marchingCubes/util/util.h"
#include "../miniIsosurface/marchingCubes/util/MarchingCubesTables.h"
#include "../miniIsosurface/marchingCubes/util/TriangleMesh.h"
#include "../miniIsosurface/marchingCubes/util/SaveTriangleMesh.h"


void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
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

size_t indicesToPosition(const std::vector<size_t>& shape, const std::vector<size_t>& indices) {
    size_t position = indices[0];
    size_t multiplier = 1;

    for (size_t i = 1; i < shape.size(); ++i) {
        multiplier *= shape[i - 1];
        position += indices[i] * multiplier;
    }

    return position;
}

void saveBigBlockMinMax(const std::string& filename, 
                        const std::vector<std::vector<std::vector<ScidxInterval<double>>>>& all_steps_block_interval_second) 
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    size_t numSteps = all_steps_block_interval_second.size();
    for (size_t step = 0; step < numSteps; ++step) {
        size_t numBigBlocks = all_steps_block_interval_second[step].size();
        for (size_t big = 0; big < numBigBlocks; ++big) {
            double minVal = std::numeric_limits<double>::max();
            double maxVal = std::numeric_limits<double>::lowest();

            for (const auto& interval : all_steps_block_interval_second[step][big]) {
                minVal = std::min(minVal, interval.low);
                maxVal = std::max(maxVal, interval.high);
            }

            outFile.write(reinterpret_cast<const char*>(&minVal), sizeof(double));
            outFile.write(reinterpret_cast<const char*>(&maxVal), sizeof(double));
        }
    }

    outFile.close();
}

void saveTreeStructureToByte(const std::vector<int>& fullTreeStructure, const std::string& treeID) {
    std::string filename = treeID + "-treeStructure";
    std::ofstream outFile(filename, std::ios::binary);
    
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    size_t structureSize = fullTreeStructure.size();
    outFile.write(reinterpret_cast<const char*>(&structureSize), sizeof(size_t));

    uint8_t currentByte = 0;
    int bitIndex = 0;

    for (int bit : fullTreeStructure) {
        currentByte |= (bit & 1) << bitIndex;
        bitIndex++;

        if (bitIndex == 8) {
            outFile.put(static_cast<char>(currentByte));
            currentByte = 0;
            bitIndex = 0;
        }
    }

    if (bitIndex > 0) {
        outFile.put(static_cast<char>(currentByte));
    }

    outFile.close();
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



int main(int argc, char *argv[])
{
    std::cout << "[DEBUG] Program started. " << std::endl;

    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::string inputFileName;
    std::string variableName;
    std::string variableType;
    size_t nDim = 0;

    size_t beginStepNum = 0;
    size_t endStepNum = 0;
    std::vector<size_t> targetSteps; 

    std::vector<size_t> blockShape(nDim), smallBlockShape(nDim);

    double relative_error_bound = 1E-3;
    size_t extraValue = 0;

    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc)
            inputFileName = argv[++i];
        else if (arg == "--variable_name" && i + 1 < argc)
            variableName = argv[++i];
        else if (arg == "--dimensions" && i + 1 < argc)
            nDim = std::stoul(argv[++i]);
        else if (arg == "--block_shape" && i + nDim < argc)
        {
            for (size_t j = 0; j < nDim; j++)
                blockShape.push_back(std::stoul(argv[++i]));
        }
        else if (arg == "--begin_step")
        {
            if (i + 1 < argc) beginStepNum = atoi(argv[i + 1]);
            else { std::cerr << "--begin_step option requires one argument." << std::endl; return 1; }
        }
        else if (arg == "--end_step")
        {
            if (i + 1 < argc) endStepNum = atoi(argv[i + 1]);
            else { std::cerr << "--end_step option requires one argument." << std::endl; return 1; }
        }
        else if (arg == "--target_steps")
        {
            while (i + 1 < argc && std::isdigit(argv[i + 1][0]))
                targetSteps.push_back(std::stoul(argv[++i]));
        }
        else if (arg == "--small_block_shape")
        {
            if (nDim)
            {
                if ((int)(i + nDim) < argc)
                    for (size_t j = i + 1; j < i + 1 + nDim; j++)
                        smallBlockShape.push_back(atoi(argv[j]));
                else { std::cerr << "--small_block_shape option requires [# of dimensions] argument." << std::endl; return 1; }
            }
        }
        else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        }
    }

    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);

    // 每个step，每个大块，所有小块的interval
    std::vector<std::vector<std::vector<ScidxInterval<double>>>> all_steps_block_interval_second;

    std::vector<std::vector<double>> all_steps_block_mins;
    std::vector<std::vector<double>> all_steps_block_maxs;

    // 小块间隙interval存储结构
    // all_steps_gap_intervals[step][big_block_idx]
    //   = vector of (core_small_block_id, gap_interval)
    std::vector<std::vector<std::vector<std::pair<size_t, ScidxInterval<double>>>>> all_steps_gap_intervals;

    size_t step = 0;
    std::set<size_t> targetSet(targetSteps.begin(), targetSteps.end());

    // numDirs = 2^nDim - 1，3D情况下为7（3面 + 3棱 + 1角）
    size_t numDirs = (size_t(1) << nDim) - 1;

    while (reader_engine.BeginStep() == adios2::StepStatus::OK)
    {
        std::cout << "[DEBUG] BeginStep ok, step=" << step << std::endl;
        size_t total_blocks = 1;

        if (!targetSet.empty() && targetSet.find(step) == targetSet.end()) 
        {
            reader_engine.EndStep();
            step++;
            continue;
        }

        auto var = reader_io.InquireVariable<double>(variableName);
        variableType = reader_io.VariableType(variableName);
       
        size_t varElements = 1;
        for (size_t i = 0; i < nDim; i++)
            varElements *= var.Shape()[i];

        std::vector<size_t> blockCountOnEachDim;
        for (size_t i = 0; i < nDim; i++)
        {
            blockCountOnEachDim.push_back(var.Shape()[i] / blockShape[i]);
            total_blocks *= var.Shape()[i] / blockShape[i];
        }

        if (variableType == "double")
        {
            std::vector<double> varData(varElements);
            reader_engine.Get(var, varData.data(), adios2::Mode::Sync);

            /*std::string safeVarName = variableName;
            std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');

            // 保存为 .vti
            {
                size_t nx = 1024, ny = 1024, nz = 1024;
                size_t totalPoints = nx * ny * nz;

                std::string vtkFile = "/expanse/lustre/scratch/sdi/temp_project/original_volume_step_" 
                    + std::to_string(step) + ".vtk";

                std::ofstream ofs(vtkFile, std::ios::binary);
                if (!ofs) { std::cerr << "Cannot open " << vtkFile << std::endl; exit(1); }

                ofs << "# vtk DataFile Version 3.0\n";
                ofs << "Volume Data\n";
                ofs << "BINARY\n";
                ofs << "DATASET STRUCTURED_POINTS\n";
                ofs << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
                ofs << "ORIGIN 0 0 0\n";
                ofs << "SPACING 1 1 1\n";
                ofs << "POINT_DATA " << totalPoints << "\n";
                ofs << "SCALARS B_y double 1\n";
                ofs << "LOOKUP_TABLE default\n";

                // VTK legacy binary 要求 big-endian，逐个字节翻转
                for (size_t i = 0; i < totalPoints; ++i) {
                    double val = varData[i];
                    uint64_t tmp;
                    std::memcpy(&tmp, &val, sizeof(double));
                    tmp = __builtin_bswap64(tmp);  // 比手写翻转更简洁，效果一样
                    ofs.write(reinterpret_cast<char*>(&tmp), sizeof(double));
                }

                ofs.close();
                std::cout << "[Done] Written " << vtkFile << std::endl;
            }*/





            std::string isoFile =
                "/expanse/lustre/scratch/sdi/temp_project/isosurface_original_step_" +
                std::to_string(step) + "_iso_0.04.vtk";

            RunAndSaveIsosurfaceFromFlatVolume(varData, 1024, 1024, 1024, 0.04, isoFile);



   
            
            /*std::vector<std::vector<double>> blocks(total_blocks);

            // 遍历所有点，放入对应大块
            for (size_t p = 0; p < varElements; p++)
            {
                std::vector<size_t> elem_global_id = positionToIndices(p, var.Shape());
                std::vector<size_t> block_global_id(nDim);
                for (size_t i = 0; i < nDim; i++)
                    block_global_id[i] = elem_global_id[i] / blockShape[i];
                             
                size_t block_position = indicesToPosition(blockCountOnEachDim, block_global_id);

                if (block_position >= total_blocks) {
                    std::cerr << "Error: block_position out of bounds." << std::endl;
                    reader_engine.EndStep();
                    return 1;
                }
                blocks[block_position].push_back(varData[p]);
            }

            std::vector<double> block_mins(total_blocks, std::numeric_limits<double>::max());
            std::vector<double> block_maxs(total_blocks, std::numeric_limits<double>::min());

            for (size_t b = 0; b < total_blocks; b++)
            {
                if (!blocks[b].empty())
                {
                    auto minmax = std::minmax_element(blocks[b].begin(), blocks[b].end());
                    block_mins[b] = *minmax.first;
                    block_maxs[b] = *minmax.second;
                }
            }

            all_steps_block_mins.push_back(block_mins);
            all_steps_block_maxs.push_back(block_maxs);

            // 一个step内所有大块的小块interval
            std::vector<std::vector<ScidxInterval<double>>> one_steps_block_interval_second;

            // 一个step内所有大块的小块间隙interval
            std::vector<std::vector<std::pair<size_t, ScidxInterval<double>>>> one_steps_gap_intervals;

            // 对每个大块进行处理
            for (size_t b = 0; b < total_blocks; b++)
            {
                std::vector<size_t> block_idx_on_each_dim = positionToIndices(b, blockCountOnEachDim);
                std::vector<size_t> blockCount(nDim);

                for (size_t i = 0; i < nDim; i++)
                {
                    if (block_idx_on_each_dim[i] == blockCountOnEachDim[i]-1)
                        blockCount[i] = var.Shape()[i] - block_idx_on_each_dim[i] * blockShape[i];
                    else
                        blockCount[i] = blockShape[i];
                }

                size_t blockSize = blocks[b].size();
                std::vector<double> blockData = blocks[b];

                // 计算该大块内小块的数量
                std::vector<size_t> smallBlockCountOnEachDim(nDim);
                size_t total_small_blocks = 1;
                for (size_t i = 0; i < nDim; i++)
                {
                    smallBlockCountOnEachDim[i] = (blockCount[i] + smallBlockShape[i] - 1) / smallBlockShape[i];
                    total_small_blocks *= smallBlockCountOnEachDim[i];
                }

                // 小块 min/max
                std::vector<double> small_block_mins(total_small_blocks, std::numeric_limits<double>::max());
                std::vector<double> small_block_maxs(total_small_blocks, std::numeric_limits<double>::min());

                // 小块间隙 min/max
                // gap_mins_sb[sb][dir-1]：小块 sb 负责的 dir 方向间隙最小值
                // gap_maxs_sb[sb][dir-1]：对应最大值
                // dir 用 bitmask 表示：
                //   dir=1(001)=+x面, dir=2(010)=+y面, dir=3(011)=+xy棱,
                //   dir=4(100)=+z面, dir=5(101)=+xz棱, dir=6(110)=+yz棱,
                //   dir=7(111)=+xyz角
                std::vector<std::vector<double>> gap_mins_sb(
                    total_small_blocks, std::vector<double>(numDirs, std::numeric_limits<double>::max()));
                std::vector<std::vector<double>> gap_maxs_sb(
                    total_small_blocks, std::vector<double>(numDirs, std::numeric_limits<double>::lowest()));

                // 遍历大块内所有点
                for (size_t p = 0; p < blockSize; p++)
                {
                    // 点在大块内的局部坐标
                    std::vector<size_t> elem_local_id = positionToIndices(p, blockCount);

                    // 点所属小块的坐标 & 点在小块内的局部坐标
                    std::vector<size_t> small_block_local_id(nDim);
                    std::vector<size_t> elem_in_small_local_id(nDim);

                    for (size_t i = 0; i < nDim; i++)
                    {
                        small_block_local_id[i]    = elem_local_id[i] / smallBlockShape[i];
                        elem_in_small_local_id[i]  = elem_local_id[i] % smallBlockShape[i];
                    }

                    size_t small_block_position = indicesToPosition(smallBlockCountOnEachDim, small_block_local_id);

                    // 更新小块 min/max
                    small_block_mins[small_block_position] = std::min(small_block_mins[small_block_position], blockData[p]);
                    small_block_maxs[small_block_position] = std::max(small_block_maxs[small_block_position], blockData[p]);

                    // ----------------------------------------------------------------
                    // 间隙 min/max 更新
                    //
                    // 对每个方向 dir，分两种情况：
                    //
                    // 情况1：当前点是小块 small_block_position 的正边界点
                    //   → dir 涉及的每个维度 i：
                    //       (a) 点在小块正边界：elem_in_small_local_id[i] == smallBlockShape[i]-1
                    //       (b) 正方向有邻居小块：small_block_local_id[i]+1 < smallBlockCountOnEachDim[i]
                    //   → 非 dir 维度：不作限制（整个面/棱/角都计入）
                    //   → 贡献到 gap_mins_sb[small_block_position][dir-1]
                    //
                    // 情况2：当前点是某核心小块邻居的负边界点
                    //   → dir 涉及的每个维度 i：
                    //       (a) 点在小块负边界：elem_in_small_local_id[i] == 0
                    //       (b) 有前驱核心小块：small_block_local_id[i] > 0
                    //   → 非 dir 维度：不作限制
                    //   → 反推核心小块坐标 core_small_idx（该维度 -1）
                    //   → 贡献到 gap_mins_sb[core_sb][dir-1]
                    //
                    // 情况1 和情况2 共同贡献到同一个 gap_mins_sb[core_sb][dir-1]，
                    // 最终合并为该方向间隙区间 [min, max]。
                    // ----------------------------------------------------------------
                    for (size_t dir = 1; dir <= numDirs; dir++)
                    {
                        // ---- 情况1：当前点在本小块正边界，正方向有邻居 ----
                        {
                            bool ok = true;
                            for (size_t i = 0; i < nDim; i++)
                            {
                                if (dir & (size_t(1) << i))
                                {
                                    // dir 维度：必须在正边界
                                    if (elem_in_small_local_id[i] != smallBlockShape[i] - 1)
                                    { ok = false; break; }
                                    // 正方向必须有邻居小块
                                    if (small_block_local_id[i] + 1 >= smallBlockCountOnEachDim[i])
                                    { ok = false; break; }
                                }
                                // 非 dir 维度：不作限制
                            }
                            if (ok)
                            {
                                gap_mins_sb[small_block_position][dir-1] =
                                    std::min(gap_mins_sb[small_block_position][dir-1], blockData[p]);
                                gap_maxs_sb[small_block_position][dir-1] =
                                    std::max(gap_maxs_sb[small_block_position][dir-1], blockData[p]);
                            }
                        }

                        // ---- 情况2：当前点在邻居小块负边界，反推核心小块 ----
                        {
                            bool ok = true;
                            std::vector<size_t> core_small_idx = small_block_local_id;
                            for (size_t i = 0; i < nDim; i++)
                            {
                                if (dir & (size_t(1) << i))
                                {
                                    // dir 维度：必须在负边界
                                    if (elem_in_small_local_id[i] != 0)
                                    { ok = false; break; }
                                    // 必须有前驱核心小块
                                    if (small_block_local_id[i] == 0)
                                    { ok = false; break; }
                                    // 反推核心小块坐标
                                    core_small_idx[i]--;
                                }
                                // 非 dir 维度：不作限制
                            }
                            if (ok)
                            {
                                size_t core_sb = indicesToPosition(smallBlockCountOnEachDim, core_small_idx);
                                gap_mins_sb[core_sb][dir-1] =
                                    std::min(gap_mins_sb[core_sb][dir-1], blockData[p]);
                                gap_maxs_sb[core_sb][dir-1] =
                                    std::max(gap_maxs_sb[core_sb][dir-1], blockData[p]);
                            }
                        }
                    }
                    // ---------------------------------------------------------------- 单点处理结束
                }

                // 构建小块 interval 列表
                std::vector<ScidxInterval<double>> blockIntervals;   
                for (size_t sb = 0; sb < total_small_blocks; sb++)
                {
                    ScidxInterval<double> interval;
                    interval.low  = small_block_mins[sb];
                    interval.high = small_block_maxs[sb];
                    blockIntervals.push_back(interval);
                }
                one_steps_block_interval_second.push_back(blockIntervals); 

                small_block_mins.clear(); small_block_mins.shrink_to_fit();
                small_block_maxs.clear(); small_block_maxs.shrink_to_fit();

                // 将该大块的小块间隙 min/max 转为 interval 列表
                // 遍历所有小块 sb 和所有正方向 dir：
                //   gap_mins_sb[sb][dir-1] > gap_maxs_sb[sb][dir-1] 说明无有效边界点，跳过
                //   否则构建间隙 interval，以核心小块 id sb 为节点 id
                std::vector<std::pair<size_t, ScidxInterval<double>>> big_block_gap_intervals;
                for (size_t sb = 0; sb < total_small_blocks; sb++)
                {
                    for (size_t dir = 1; dir <= numDirs; dir++)
                    {
                        double gMin = gap_mins_sb[sb][dir-1];
                        double gMax = gap_maxs_sb[sb][dir-1];
                        // 无有效边界点则跳过
                        if (gMin > gMax) continue;

                        ScidxInterval<double> gapInterval;
                        gapInterval.low  = gMin;
                        gapInterval.high = gMax;
                        big_block_gap_intervals.emplace_back(sb, gapInterval);
                    }
                }
                one_steps_gap_intervals.push_back(big_block_gap_intervals);

                gap_mins_sb.clear(); gap_mins_sb.shrink_to_fit();
                gap_maxs_sb.clear(); gap_maxs_sb.shrink_to_fit();
            }

            all_steps_block_interval_second.push_back(one_steps_block_interval_second);
            all_steps_gap_intervals.push_back(one_steps_gap_intervals);

            for (auto& block : blocks) { block.clear(); block.shrink_to_fit(); }
            varData.clear(); varData.shrink_to_fit();*/
        }

        reader_engine.EndStep();
        step++;
    }
    reader_engine.Close();

    /*std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string subDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + safeVarName + "_" 
                        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" 
                        + std::to_string(extraValue) + "_newour_index/";

    if (mpi_rank == 0) createDirectory(subDir);
    MPI_Barrier(MPI_COMM_WORLD);

    double global_min = std::numeric_limits<double>::max();
    double global_max = std::numeric_limits<double>::lowest();

    for (const auto& step_blocks : all_steps_block_interval_second)
        for (const auto& big_block : step_blocks)
            for (const auto& interval : big_block) {
                global_min = std::min(global_min, interval.low);
                global_max = std::max(global_max, interval.high);
            }

    double error_bound = relative_error_bound * (global_max - global_min);
    std::cout << "error_bound = " << error_bound << std::endl;

    std::string filename = subDir + "big_block_minmax";
    saveBigBlockMinMax(filename, all_steps_block_interval_second);

    size_t numSteps = all_steps_block_interval_second.size();

    MPI_Barrier(MPI_COMM_WORLD);

    // ===== 原有小块树计时 =====
    double orig_total_wall   = 0.0;
    double orig_sum_build    = 0.0;
    double orig_sum_compress = 0.0;

    double orig_start = MPI_Wtime();

    // 原有小块树构建循环
    for (size_t step = 0; step < numSteps; ++step) {
        size_t actualStep = beginStepNum + step;
        size_t numBigBlocks = all_steps_block_interval_second[step].size();

        for (size_t big = 0; big < numBigBlocks; ++big) {
            size_t task_id = step * numBigBlocks + big;
            if (task_id % mpi_size != mpi_rank) continue; 
    
            auto start_time1 = std::chrono::high_resolution_clock::now();

            ScidxAVLIntervalTree<double> avlIntervalTree;
            std::vector<ScidxInterval<double>> intervals = all_steps_block_interval_second[step][big];
            std::string treeID = subDir + std::to_string(actualStep) + "-" + std::to_string(big);

            std::vector<SkippedNode<double>> skippedSameLowIntervals;
            std::vector<SkippedNode<double>> skippedZeroLowIntervals;

            for (size_t i = 0; i < intervals.size(); i++) {
                if (intervals[i].low == 0) {
                    skippedZeroLowIntervals.push_back(SkippedNode<double>(intervals[i], i));
                    continue;
                }
                avlIntervalTree.insertNode(i, intervals[i], skippedSameLowIntervals);
            }

            if (avlIntervalTree.getRoot() == nullptr) {
                ScidxInterval<double> dummyInterval;
                dummyInterval.low = 0; dummyInterval.high = 0;
                std::vector<SkippedNode<double>> dummySkipped;
                avlIntervalTree.insertNode(static_cast<size_t>(-1), dummyInterval, dummySkipped);
            }
            
            std::cout << "skippedSameLowIntervals size: " << skippedSameLowIntervals.size() << std::endl;
            std::cout << "skippedZeroLowIntervals size: " << skippedZeroLowIntervals.size() << std::endl;

            std::vector<SkippedNode<double>> skippedAllIntervals;
            skippedAllIntervals.insert(skippedAllIntervals.end(), skippedZeroLowIntervals.begin(), skippedZeroLowIntervals.end());
            skippedAllIntervals.insert(skippedAllIntervals.end(), skippedSameLowIntervals.begin(), skippedSameLowIntervals.end());
            skippedSameLowIntervals.clear();
            skippedZeroLowIntervals.clear();

            if (skippedAllIntervals.empty()) {
                ScidxInterval<double> dummyInterval;
                dummyInterval.low = 0; dummyInterval.high = 0;
                skippedAllIntervals.emplace_back(dummyInterval, static_cast<size_t>(-1));
            }

            auto start_time2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = start_time2 - start_time1;
            orig_sum_build += elapsed2.count();
            std::cout << "[Rank " << mpi_rank << "] [Orig Time1 build]: " << elapsed2.count() << "s" << std::endl;

            std::vector<int> fullTreeStructure; 
            std::vector<std::vector<ScidxAVLNode<double>*>> allLevels;
            levelOrderTraversal(avlIntervalTree.getRoot(), avlIntervalTree.getTreeHeight() + 1, allLevels, fullTreeStructure);
            saveTreeStructureToByte(fullTreeStructure, treeID);
            computeOptimizedAVL(allLevels, error_bound, treeID, skippedAllIntervals);

            auto start_time3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed3 = start_time3 - start_time2;
            orig_sum_compress += elapsed3.count();
            std::cout << "[Rank " << mpi_rank << "] [Orig Time2 compress+io]: " << elapsed3.count() << "s" << std::endl;
        }
    }

    double orig_end = MPI_Wtime();
    orig_total_wall = orig_end - orig_start;

    // ===== 间隙树构建循环 =====
    MPI_Barrier(MPI_COMM_WORLD);

    double gap_total_wall   = 0.0;
    double gap_sum_build    = 0.0;
    double gap_sum_compress = 0.0;

    double gap_start = MPI_Wtime();

    for (size_t step = 0; step < numSteps; ++step)
    {
        size_t actualStep = beginStepNum + step;
        size_t numBigBlocks = all_steps_gap_intervals[step].size();

        for (size_t big = 0; big < numBigBlocks; ++big)
        {
            size_t task_id = step * numBigBlocks + big;
            if (task_id % mpi_size != mpi_rank) continue;

            std::string gapTreeID = subDir + std::to_string(actualStep) + "-" + std::to_string(big) + "-gap";

            auto gap_time1 = std::chrono::high_resolution_clock::now();

            ScidxAVLIntervalTree<double> gapTree;
            std::vector<SkippedNode<double>> gapSkippedSameLow;
            std::vector<SkippedNode<double>> gapSkippedZeroLow;

            for (const auto& [sb_id, gapInterval] : all_steps_gap_intervals[step][big])
            {
                if (gapInterval.low == 0) {
                    gapSkippedZeroLow.push_back(SkippedNode<double>(gapInterval, sb_id));
                    continue;
                }
                gapTree.insertNode(sb_id, gapInterval, gapSkippedSameLow);
            }

            if (gapTree.getRoot() == nullptr) {
                ScidxInterval<double> dummyInterval;
                dummyInterval.low = 0; dummyInterval.high = 0;
                std::vector<SkippedNode<double>> dummySkipped;
                gapTree.insertNode(static_cast<size_t>(-1), dummyInterval, dummySkipped);
            }

            std::vector<SkippedNode<double>> gapSkippedAll;
            gapSkippedAll.insert(gapSkippedAll.end(), gapSkippedZeroLow.begin(), gapSkippedZeroLow.end());
            gapSkippedAll.insert(gapSkippedAll.end(), gapSkippedSameLow.begin(), gapSkippedSameLow.end());
            gapSkippedSameLow.clear();
            gapSkippedZeroLow.clear();

            if (gapSkippedAll.empty()) {
                ScidxInterval<double> dummyInterval;
                dummyInterval.low = 0; dummyInterval.high = 0;
                gapSkippedAll.emplace_back(dummyInterval, static_cast<size_t>(-1));
            }

            auto gap_time2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> gap_elapsed_build = gap_time2 - gap_time1;
            gap_sum_build += gap_elapsed_build.count();
            std::cout << "[Rank " << mpi_rank << "] [Gap Time1 build] step=" << actualStep
                      << " big=" << big << ": " << gap_elapsed_build.count() << "s"
                      << "  total_gap_intervals=" << all_steps_gap_intervals[step][big].size() << std::endl;

            std::vector<int> gapTreeStructure;
            std::vector<std::vector<ScidxAVLNode<double>*>> gapAllLevels;
            levelOrderTraversal(gapTree.getRoot(), gapTree.getTreeHeight() + 1, gapAllLevels, gapTreeStructure);
            saveTreeStructureToByte(gapTreeStructure, gapTreeID);
            computeOptimizedAVL(gapAllLevels, error_bound, gapTreeID, gapSkippedAll);

            auto gap_time3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> gap_elapsed_compress = gap_time3 - gap_time2;
            gap_sum_compress += gap_elapsed_compress.count();
            std::cout << "[Rank " << mpi_rank << "] [Gap Time2 compress+io] step=" << actualStep
                      << " big=" << big << ": " << gap_elapsed_compress.count() << "s" << std::endl;
        }
    }

    double gap_end = MPI_Wtime();
    gap_total_wall = gap_end - gap_start;

    // ===== 分别统计两套树的时间（按阶段） =====
    double orig_wall_max, orig_build_max, orig_compress_max;
    double orig_wall_sum, orig_build_sum, orig_compress_sum;
    double gap_wall_max,  gap_build_max,  gap_compress_max;
    double gap_wall_sum,  gap_build_sum,  gap_compress_sum;

    // 原树 Reduce
    MPI_Reduce(&orig_total_wall,   &orig_wall_max,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&orig_total_wall,   &orig_wall_sum,     1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&orig_sum_build,    &orig_build_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&orig_sum_build,    &orig_build_sum,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&orig_sum_compress, &orig_compress_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&orig_sum_compress, &orig_compress_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // 间隙树 Reduce
    MPI_Reduce(&gap_total_wall,   &gap_wall_max,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gap_total_wall,   &gap_wall_sum,     1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gap_sum_build,    &gap_build_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gap_sum_build,    &gap_build_sum,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gap_sum_compress, &gap_compress_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gap_sum_compress, &gap_compress_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::cout << "\n========== Timing Summary ==========" << std::endl;

        std::cout << "[Original tree]" << std::endl;
        std::cout << "  Wall time   : max=" << orig_wall_max
                  << "s  avg=" << orig_wall_sum / mpi_size << "s" << std::endl;
        std::cout << "  Build time  : max=" << orig_build_max
                  << "s  avg=" << orig_build_sum / mpi_size << "s" << std::endl;
        std::cout << "  Compress+IO : max=" << orig_compress_max
                  << "s  avg=" << orig_compress_sum / mpi_size << "s" << std::endl;

        std::cout << "[Gap tree]" << std::endl;
        std::cout << "  Wall time   : max=" << gap_wall_max
                  << "s  avg=" << gap_wall_sum / mpi_size << "s" << std::endl;
        std::cout << "  Build time  : max=" << gap_build_max
                  << "s  avg=" << gap_build_sum / mpi_size << "s" << std::endl;
        std::cout << "  Compress+IO : max=" << gap_compress_max
                  << "s  avg=" << gap_compress_sum / mpi_size << "s" << std::endl;

        std::cout << "[Total]" << std::endl;
        std::cout << "  Wall time   : " << orig_wall_max + gap_wall_max << "s" << std::endl;
        std::cout << "====================================" << std::endl;
    }*/
     
    MPI_Finalize(); 
    return 0;
}
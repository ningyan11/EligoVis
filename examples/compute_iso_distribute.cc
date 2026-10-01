#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <algorithm>
#include <scidx_avl.h>
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <utility>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <iomanip>
#include <limits>
#include <cmath>

// ============================================================
// 预加载树的数据结构：每棵树只解压一次，存在内存里
// ============================================================
struct PreloadedTree {
    ScidxAVLNode<double>* root = nullptr;
    std::vector<double>   skippedMin;
    std::vector<double>   skippedMax;
    std::vector<size_t>   skippedIds;
    bool valid = false;
};

// ============================================================
// loadBigBlockIndexFile（与原码相同）
// ============================================================
std::vector<ScidxInterval<double>> loadBigBlockIndexFile(const std::string& filename)
{
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
// safeQueryOverlapIds（与原码相同）
// ============================================================
std::vector<size_t> safeQueryOverlapIds(ScidxAVLNode<double>* root,
                                        const ScidxInterval<double>& query)
{
    if (root && root->id == static_cast<size_t>(-1)) return {};
    return queryOverlapIds(root, query);
}

// ============================================================
// 使用预加载树查询：只做query，不再解压树
// adjustAndUpdateMaxHigh 已在预加载阶段完成
// ============================================================
std::vector<size_t> queryPreloadedTree(
    const PreloadedTree& tree,
    double isovalue,
    double error_bound)
{
    if (!tree.valid) return {};

    ScidxInterval<double> query;
    query.low  = isovalue;
    query.high = isovalue;

    auto ids = safeQueryOverlapIds(tree.root, query);

    for (size_t i = 0; i < tree.skippedMin.size(); ++i) {
        if (tree.skippedIds[i] == static_cast<size_t>(-1)) continue;
        double minWithError = tree.skippedMin[i] - error_bound;
        double maxWithError = tree.skippedMax[i] + error_bound;
        if (!(minWithError > query.high || maxWithError < query.low))
            ids.push_back(tree.skippedIds[i]);
    }

    return ids;
}

// ============================================================
// convertStaggerToUniformIds（与原码相同）
// ============================================================
std::vector<size_t> convertStaggerToUniformIds(
    size_t staggerFlatId,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& smallBlockShape,
    const std::vector<size_t>& halfBlockShape,
    const std::vector<size_t>& staggerBlockCount,
    const std::vector<size_t>& uniformBlockCount)
{
    size_t sk0 = staggerFlatId % staggerBlockCount[0];
    size_t sk1 = (staggerFlatId / staggerBlockCount[0]) % staggerBlockCount[1];
    size_t sk2 = staggerFlatId / (staggerBlockCount[0] * staggerBlockCount[1]);
    std::array<size_t, 3> sk = {sk0, sk1, sk2};

    std::array<std::pair<size_t, size_t>, 3> uRange;
    for (int d = 0; d < 3; d++) {
        size_t k = sk[d];
        size_t start, end_excl;
        if (k == 0) {
            start    = 0;
            end_excl = halfBlockShape[d];
        } else {
            start    = halfBlockShape[d] + (k - 1) * smallBlockShape[d];
            end_excl = std::min(start + smallBlockShape[d], shape[d]);
        }
        if (start >= shape[d] || end_excl == 0 || start >= end_excl) return {};

        size_t u_start = start / smallBlockShape[d];
        size_t u_end   = (end_excl - 1) / smallBlockShape[d];
        u_end = std::min(u_end, uniformBlockCount[d] - 1);
        uRange[d] = {u_start, u_end};
    }

    std::vector<size_t> result;
    for (size_t ux = uRange[0].first; ux <= uRange[0].second; ux++)
        for (size_t uy = uRange[1].first; uy <= uRange[1].second; uy++)
            for (size_t uz = uRange[2].first; uz <= uRange[2].second; uz++)
                result.push_back(ux
                    + uy * uniformBlockCount[0]
                    + uz * uniformBlockCount[0] * uniformBlockCount[1]);
    return result;
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;

    // ===== 参数解析（与原码相同）=====
    std::string inputFileName, variableName;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, blockShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0;
    size_t smallBlockSize = 1;
    double relative_error_bound = 1E-3;
    size_t extraValue = 0;

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
        } else if (arg == "--bigBlock_shape" && i + nDim < argc) {
            for (size_t j = 0; j < nDim; j++)
                blockShape.push_back(std::stoul(argv[++i]));
        } else if (arg == "--small_block_shape") {
            if (nDim && (int)(i + nDim) < argc)
                for (size_t j = i + 1; j < i + 1 + nDim; j++)
                    smallBlockShape.push_back(atoi(argv[j]));
        } else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        }
    }

    // ===== 计算布局参数（与原码相同）=====
    size_t smallBlocksPerStep = 1;
    for (size_t d = 0; d < nDim; d++) {
        smallBlocksPerStep *= stepDataShape[d] / smallBlockShape[d];
        smallBlockSize     *= smallBlockShape[d];
    }
    size_t nSteps = endStepNum - beginStepNum + 1;

    // ===== 构造目录路径（与原码相同）=====
    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    std::string indexDir = "/home/nyan/scidx/scidx/"
        + inputFileBaseName + "_" + safeVarName + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_newour_stagger1_index/";

    std::string staggerIndexDir = "/home/nyan/scidx/scidx/"
        + inputFileBaseName + "_" + safeVarName + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_newour_stagger2_index/";

    // ===== 错位块布局参数（与原码相同）=====
    std::vector<size_t> halfBlockShape(nDim);
    std::vector<size_t> staggerBlockCount(nDim);
    std::vector<size_t> uniformBlockCount(nDim);
    for (size_t d = 0; d < nDim; d++) {
        halfBlockShape[d]    = smallBlockShape[d] / 2;
        uniformBlockCount[d] = stepDataShape[d] / smallBlockShape[d];
        staggerBlockCount[d] = uniformBlockCount[d] + 1;
    }

    // ===== 块网格布局（用于邻居扩展）=====
    std::vector<size_t> blockCountOnEachDim(3);
    for (size_t i = 0; i < 3; i++)
        blockCountOnEachDim[i] = (stepDataShape[i] + smallBlockShape[i] - 1) / smallBlockShape[i];

    // ===== 加载 big_block_minmax =====
    auto allBigBlockIndices        = loadBigBlockIndexFile(indexDir + "big_block_minmax");
    auto allStaggerBigBlockIndices = loadBigBlockIndexFile(staggerIndexDir + "big_block_minmax");

    // ===== 计算 globalMin/Max 和 error_bound =====
    double globalMin = std::numeric_limits<double>::max();
    double globalMax = std::numeric_limits<double>::lowest();
    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }
    std::cout << "Global Min:        " << globalMin << std::endl;
    std::cout << "Global Max:        " << globalMax << std::endl;

    double error_bound = relative_error_bound * (globalMax - globalMin);
    std::cout << "error_bound:       " << error_bound << std::endl;
    std::cout << "Blocks per step:   " << smallBlocksPerStep << std::endl;
    std::cout << "nSteps:            " << nSteps << std::endl;

    // ============================================================
    // 预加载所有 step 的树（只做一次 decompressOptimizedAVL）
    // adjustAndUpdateMaxHigh 也只做一次（error_bound 固定不变）
    // ============================================================
    std::cout << "\n[Preload] Loading all index trees..." << std::endl;
    auto preloadStart = std::chrono::high_resolution_clock::now();

    std::vector<PreloadedTree> uniformTrees(nSteps);
    std::vector<PreloadedTree> staggerTrees(nSteps);

    for (size_t stepRelIdx = 0; stepRelIdx < nSteps; ++stepRelIdx) {
        size_t actualStepNum = beginStepNum + stepRelIdx;

        // --- 均匀 index 树 ---
        {
            std::string treeID = indexDir + std::to_string(actualStepNum) + "-0";
            auto [root, skippedMin, skippedMax, skippedIds] =
                decompressOptimizedAVL(treeID, error_bound);
            adjustAndUpdateMaxHigh(root, error_bound);
            uniformTrees[stepRelIdx].root       = root;
            uniformTrees[stepRelIdx].skippedMin = skippedMin;
            uniformTrees[stepRelIdx].skippedMax = skippedMax;
            uniformTrees[stepRelIdx].skippedIds = skippedIds;
            uniformTrees[stepRelIdx].valid      = true;
            std::cout << "  [Uniform] Step " << actualStepNum << " loaded" << std::endl;
        }

        // --- 错位 index 树 ---
        {
            std::string treeID = staggerIndexDir + std::to_string(actualStepNum) + "-0";
            auto [root, skippedMin, skippedMax, skippedIds] =
                decompressOptimizedAVL(treeID, error_bound);
            adjustAndUpdateMaxHigh(root, error_bound);
            staggerTrees[stepRelIdx].root       = root;
            staggerTrees[stepRelIdx].skippedMin = skippedMin;
            staggerTrees[stepRelIdx].skippedMax = skippedMax;
            staggerTrees[stepRelIdx].skippedIds = skippedIds;
            staggerTrees[stepRelIdx].valid      = true;
            std::cout << "  [Stagger] Step " << actualStepNum << " loaded" << std::endl;
        }
    }

    auto preloadEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> preloadTime = preloadEnd - preloadStart;
    std::cout << "[Preload] Done in " << preloadTime.count() << " seconds\n" << std::endl;

    // ============================================================
    // 26 方向邻居定义（与原码相同）
    // ============================================================
    struct NeighborDir { int dx, dy, dz; };
    std::vector<NeighborDir> neighborDirections;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (dx != 0 || dy != 0 || dz != 0)
                    neighborDirections.push_back({dx, dy, dz});

    // ============================================================
    // 对数均匀采样 1000 个 isovalue
    // 正负各 500 个，|isovalue| 在 [epsilon, globalMax] 对数均匀分布
    // 使数据稀疏区（极值附近）和稠密区（接近0）都有充分采样
    // ============================================================
    const int    N_SAMPLES = 1000;
    const int    half      = N_SAMPLES / 2;

    // epsilon：取数据范围的 1e-6，避免 isovalue 正好等于 0
    double absMax   = std::max(std::abs(globalMin), std::abs(globalMax));
    double epsilon  = 1e-6 * absMax;

    double logMin = std::log10(epsilon);
    double logMax = std::log10(absMax);

    std::vector<double> isovalues;
    isovalues.reserve(N_SAMPLES);

    // 正方向：对数均匀 [epsilon, absMax]
    for (int i = 0; i < half; ++i) {
        double t      = static_cast<double>(i) / (half - 1);
        double logVal = logMin + t * (logMax - logMin);
        isovalues.push_back(std::pow(10.0, logVal));
    }

    // 负方向：对数均匀 [-absMax, -epsilon]
    for (int i = 0; i < half; ++i) {
        double t      = static_cast<double>(i) / (half - 1);
        double logVal = logMin + t * (logMax - logMin);
        isovalues.push_back(-std::pow(10.0, logVal));
    }

    // 打乱顺序，避免正负交替规律输出
    std::mt19937 gen(42);
    std::shuffle(isovalues.begin(), isovalues.end(), gen);

    // ============================================================
    // 打印表头
    // ============================================================
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::setw(5)  << "Idx"
              << std::setw(18) << "Isovalue"
              << std::setw(14) << "CoreBlocks"
              << std::setw(16) << "ExpandedBlocks"
              << std::setw(14) << "Ratio(%)"
              << std::endl;
    std::cout << std::string(67, '-') << std::endl;

    // ============================================================
    // 对每个 isovalue：
    //   1. big_block_minmax 粗筛 step
    //   2. 精细树查询（使用预加载树）
    //   3. stagger → uniform 转换 + 去重
    //   4. 26邻居扩展
    //   5. 打印比例
    // ============================================================
    for (int isoIdx = 0; isoIdx < N_SAMPLES; ++isoIdx) {
        double isovalue = isovalues[isoIdx];

        // --- Step 1: big_block_minmax 粗筛 ---
        std::unordered_set<size_t> selectedStepsSet;
        for (size_t i = 0; i < allBigBlockIndices.size(); i++)
            if (isovalue >= allBigBlockIndices[i].low &&
                isovalue <= allBigBlockIndices[i].high)
                selectedStepsSet.insert(i);
        for (size_t i = 0; i < allStaggerBigBlockIndices.size(); i++)
            if (isovalue >= allStaggerBigBlockIndices[i].low &&
                isovalue <= allStaggerBigBlockIndices[i].high)
                selectedStepsSet.insert(i);

        // --- Step 2 & 3: 精细查询 + stagger→uniform + 去重 ---
        std::unordered_set<size_t> hitUniformBlocksSet;

        for (size_t stepRelIdx : selectedStepsSet) {
            if (stepRelIdx >= nSteps) continue;
            size_t stepOffset = stepRelIdx * smallBlocksPerStep;

            // 均匀 index 查询
            auto uniformRawIds = queryPreloadedTree(
                uniformTrees[stepRelIdx], isovalue, error_bound);
            for (size_t rawId : uniformRawIds)
                if (rawId < smallBlocksPerStep)
                    hitUniformBlocksSet.insert(stepOffset + rawId);

            // 错位 index 查询 → 转换为均匀块 ID
            auto staggerRawIds = queryPreloadedTree(
                staggerTrees[stepRelIdx], isovalue, error_bound);
            for (size_t staggerId : staggerRawIds) {
                auto uniformIds = convertStaggerToUniformIds(
                    staggerId, stepDataShape, smallBlockShape,
                    halfBlockShape, staggerBlockCount, uniformBlockCount);
                for (size_t uid : uniformIds)
                    if (uid < smallBlocksPerStep)
                        hitUniformBlocksSet.insert(stepOffset + uid);
            }
        }

        // core blocks
        std::vector<size_t> coreBlockIds(
            hitUniformBlocksSet.begin(), hitUniformBlocksSet.end());
        size_t coreCount = coreBlockIds.size();

        // --- Step 4: 26邻居扩展（与原码完全相同）---
        std::unordered_set<size_t> allBlocksSet(
            coreBlockIds.begin(), coreBlockIds.end());

        for (size_t coreBlockId : coreBlockIds) {
            size_t block_z   = coreBlockId % blockCountOnEachDim[2];
            size_t remaining = coreBlockId / blockCountOnEachDim[2];
            size_t block_y   = remaining % blockCountOnEachDim[1];
            size_t block_x   = remaining / blockCountOnEachDim[1];

            for (const auto& dir : neighborDirections) {
                int nbx = static_cast<int>(block_x) + dir.dx;
                int nby = static_cast<int>(block_y) + dir.dy;
                int nbz = static_cast<int>(block_z) + dir.dz;

                if (nbx < 0 || nbx >= static_cast<int>(blockCountOnEachDim[0]) ||
                    nby < 0 || nby >= static_cast<int>(blockCountOnEachDim[1]) ||
                    nbz < 0 || nbz >= static_cast<int>(blockCountOnEachDim[2]))
                    continue;

                size_t neighborId =
                    static_cast<size_t>(nbx) * blockCountOnEachDim[1] * blockCountOnEachDim[2] +
                    static_cast<size_t>(nby) * blockCountOnEachDim[2] +
                    static_cast<size_t>(nbz);

                allBlocksSet.insert(neighborId);
            }
        }

        size_t expandedCount = allBlocksSet.size();
        double ratio = (smallBlocksPerStep > 0)
            ? (100.0 * static_cast<double>(expandedCount) /
               static_cast<double>(smallBlocksPerStep))
            : 0.0;

        // --- Step 5: 打印结果 ---
        std::cout << std::setw(5)  << isoIdx
                  << std::setw(18) << isovalue
                  << std::setw(14) << coreCount
                  << std::setw(16) << expandedCount
                  << std::setw(13) << ratio << "%"
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
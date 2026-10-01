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
// 预加载树的数据结构（float版）：只解压一次，存在内存里
// 对接 doc17 / doc16 的 uniform-only 索引（无 stagger）
// ============================================================
struct PreloadedTree {
    ScidxAVLNode<float>* root = nullptr;
    std::vector<float>   skippedMin;
    std::vector<float>   skippedMax;
    std::vector<size_t>  skippedIds;
    bool valid = false;
};

// ============================================================
// loadBigBlockIndexFile（float版，对接 doc16 索引程序生成的格式）
// ============================================================
std::vector<ScidxInterval<float>> loadBigBlockIndexFile(const std::string& filename)
{
    std::vector<ScidxInterval<float>> bigBlockIndices;
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return bigBlockIndices;
    }
    float minVal, maxVal;
    while (inFile.read(reinterpret_cast<char*>(&minVal), sizeof(float)) &&
           inFile.read(reinterpret_cast<char*>(&maxVal), sizeof(float))) {
        bigBlockIndices.push_back({minVal, maxVal});
    }
    inFile.close();
    return bigBlockIndices;
}

// ============================================================
// safeQueryOverlapIds（float版）
// ============================================================
std::vector<size_t> safeQueryOverlapIds(ScidxAVLNode<float>* root,
                                        const ScidxInterval<float>& query)
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
    float isovalue,
    float error_bound)
{
    if (!tree.valid) return {};

    ScidxInterval<float> query;
    query.low  = isovalue;
    query.high = isovalue;

    auto ids = safeQueryOverlapIds(tree.root, query);

    for (size_t i = 0; i < tree.skippedMin.size(); ++i) {
        if (tree.skippedIds[i] == static_cast<size_t>(-1)) continue;
        float minWithError = tree.skippedMin[i] - error_bound;
        float maxWithError = tree.skippedMax[i] + error_bound;
        if (!(minWithError > query.high || maxWithError < query.low))
            ids.push_back(tree.skippedIds[i]);
    }

    return ids;
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;

    // ===== 参数解析（对齐 doc17，无 --query_range，因为isovalue由程序内部生成）=====
    std::string inputFileName, variableName;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0;
    size_t smallBlockSize = 1;
    float relative_error_bound = 1E-3;
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
        } else if (arg == "--small_block_shape") {
            if (nDim && (int)(i + nDim) < argc)
                for (size_t j = i + 1; j < i + 1 + nDim; j++)
                    smallBlockShape.push_back(atoi(argv[j]));
        } else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        }
    }

    // ===== 计算布局参数 =====
    size_t smallBlocksPerStep = 1;
    for (size_t d = 0; d < nDim; d++) {
        smallBlocksPerStep *= stepDataShape[d] / smallBlockShape[d];
        smallBlockSize     *= smallBlockShape[d];
    }
    size_t nSteps = endStepNum - beginStepNum + 1;

    // ===== 构造索引目录路径（对接 doc16）=====
    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();

    std::string indexDir = "/expanse/lustre/scratch/sdi/temp_project/nyx_index_compress/"
        + inputFileBaseName + "_" + safeVarName + "_"
        + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
        + std::to_string(extraValue) + "_hdf5_uniform_index/";

    // ===== 加载 big_block_minmax =====
    auto allBigBlockIndices = loadBigBlockIndexFile(indexDir + "big_block_minmax");

    // ===== 计算 globalMin/Max 和 error_bound =====
    float globalMin = std::numeric_limits<float>::max();
    float globalMax = std::numeric_limits<float>::lowest();
    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }
    std::cout << "Global Min:        " << globalMin << std::endl;
    std::cout << "Global Max:        " << globalMax << std::endl;

    float error_bound = relative_error_bound * (globalMax - globalMin);
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

    for (size_t stepRelIdx = 0; stepRelIdx < nSteps; ++stepRelIdx) {
        size_t actualStepNum = beginStepNum + stepRelIdx;

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

    auto preloadEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> preloadTime = preloadEnd - preloadStart;
    std::cout << "[Preload] Done in " << preloadTime.count() << " seconds\n" << std::endl;

    // ============================================================
    // 对数均匀采样 100 个 isovalue
    // 正负各 50 个，|isovalue| 在 [epsilon, absMax] 对数均匀分布
    // 使数据稀疏区（极值附近）和稠密区（接近0）都有充分采样
    // ============================================================
    const int    N_SAMPLES = 100;
    const int    half      = N_SAMPLES / 2;

    float absMax   = std::max(std::abs(globalMin), std::abs(globalMax));
    float epsilon  = 1e-6f * absMax;

    double logMin = std::log10(static_cast<double>(epsilon));
    double logMax = std::log10(static_cast<double>(absMax));

    std::vector<float> isovalues;
    isovalues.reserve(N_SAMPLES);

    // 正方向：对数均匀 [epsilon, absMax]
    for (int i = 0; i < half; ++i) {
        double t      = static_cast<double>(i) / (half - 1);
        double logVal = logMin + t * (logMax - logMin);
        isovalues.push_back(static_cast<float>(std::pow(10.0, logVal)));
    }

    // 负方向：对数均匀 [-absMax, -epsilon]
    for (int i = 0; i < half; ++i) {
        double t      = static_cast<double>(i) / (half - 1);
        double logVal = logMin + t * (logMax - logMin);
        isovalues.push_back(static_cast<float>(-std::pow(10.0, logVal)));
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
              << std::setw(14) << "Ratio(%)"
              << std::endl;
    std::cout << std::string(51, '-') << std::endl;

    // ============================================================
    // 对每个 isovalue：
    //   1. big_block_minmax 粗筛 step
    //   2. 精细树查询（使用预加载树，仅 uniform，无 stagger）
    //   3. 去重
    //   4. 打印命中比例（不做26邻居扩展，直接统计核心命中块）
    // ============================================================
    for (int isoIdx = 0; isoIdx < N_SAMPLES; ++isoIdx) {
        float isovalue = isovalues[isoIdx];

        // --- Step 1: big_block_minmax 粗筛 ---
        std::unordered_set<size_t> selectedStepsSet;
        for (size_t i = 0; i < allBigBlockIndices.size(); i++)
            if (isovalue >= allBigBlockIndices[i].low &&
                isovalue <= allBigBlockIndices[i].high)
                selectedStepsSet.insert(i);

        // --- Step 2 & 3: 精细查询 + 去重 ---
        std::unordered_set<size_t> hitBlocksSet;

        for (size_t stepRelIdx : selectedStepsSet) {
            if (stepRelIdx >= nSteps) continue;
            size_t stepOffset = stepRelIdx * smallBlocksPerStep;

            auto rawIds = queryPreloadedTree(
                uniformTrees[stepRelIdx], isovalue, error_bound);
            for (size_t rawId : rawIds)
                if (rawId < smallBlocksPerStep)
                    hitBlocksSet.insert(stepOffset + rawId);
        }

        size_t coreCount = hitBlocksSet.size();

        double ratio = (smallBlocksPerStep > 0)
            ? (100.0 * static_cast<double>(coreCount) /
               static_cast<double>(smallBlocksPerStep))
            : 0.0;

        // --- Step 4: 打印结果 ---
        std::cout << std::setw(5)  << isoIdx
                  << std::setw(18) << isovalue
                  << std::setw(14) << coreCount
                  << std::setw(13) << ratio << "%"
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}

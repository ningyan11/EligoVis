#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <set>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <limits>

#include <mpi.h>
#include <adios2.h>
#include <zfp.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // =========================================================
    // 1. 解析命令行参数
    // =========================================================
    std::string inputFileName;
    std::string variableName;
    size_t nDim            = 0;
    size_t beginStepNum    = 0;
    size_t endStepNum      = 0;
    size_t extraValue      = 0;
    double relative_error_bound = 1E-3;
    std::vector<size_t> targetSteps;

    for (int i = 1; i < argc; i++)
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
        else if (arg == "--begin_step" && i + 1 < argc)
        {
            beginStepNum = std::stoul(argv[++i]);
        }
        else if (arg == "--end_step" && i + 1 < argc)
        {
            endStepNum = std::stoul(argv[++i]);
        }
        else if (arg == "--target_steps")
        {
            while (i + 1 < argc && std::isdigit(argv[i + 1][0]))
                targetSteps.push_back(std::stoul(argv[++i]));
        }
        else if (arg == "--relative_error" && i + 2 < argc)
        {
            relative_error_bound = std::stod(argv[++i]);
            extraValue           = std::stoul(argv[++i]);
        }
    }

    if (inputFileName.empty() || variableName.empty() || nDim == 0)
    {
        if (world_rank == 0)
            std::cerr << "Usage: --input_file <f> --variable_name <v> "
                         "--dimensions <n> [--target_steps s1 s2 ...] "
                         "[--begin_step b] [--end_step e] "
                         "[--relative_error eb extra]\n";
        MPI_Finalize();
        return 1;
    }

    if (nDim < 1 || nDim > 3)
    {
        if (world_rank == 0)
            std::cerr << "[ERROR] Only 1D/2D/3D variables are supported.\n";
        MPI_Finalize();
        return 1;
    }

    // =========================================================
    // 2. 用 ADIOS2 读取数据（所有进程各自读，或只 rank0 读后广播）
    //    这里选择所有进程各自独立读取，简单可靠
    // =========================================================
    std::vector<double>              flatData;       // 所有 step 原始数据拼接
    std::vector<std::vector<size_t>> allShapes;      // 每个 step 的形状

    {
        adios2::ADIOS adios;        // 每个进程独立 IO
        adios2::IO    io     = adios.DeclareIO("ReaderIO");
        adios2::Engine engine = io.Open(inputFileName, adios2::Mode::Read);

        std::set<size_t> targetSet(targetSteps.begin(), targetSteps.end());
        size_t stepIdx = 0;

        while (engine.BeginStep() == adios2::StepStatus::OK)
        {
            // 跳过不需要的 step
            if (!targetSet.empty() && targetSet.find(stepIdx) == targetSet.end())
            {
                engine.EndStep();
                ++stepIdx;
                continue;
            }

            // 只读 double 类型变量
            std::string vtype = io.VariableType(variableName);
            if (vtype != "double")
            {
                std::cerr << "[WARN] Step " << stepIdx
                          << ": variable type is '" << vtype
                          << "', expected 'double'. Skipping.\n";
                engine.EndStep();
                ++stepIdx;
                continue;
            }

            auto var = io.InquireVariable<double>(variableName);
            if (!var)
            {
                std::cerr << "[WARN] Step " << stepIdx
                          << ": variable '" << variableName
                          << "' not found. Skipping.\n";
                engine.EndStep();
                ++stepIdx;
                continue;
            }

            // 读取形状
            auto adiosShape = var.Shape();
            if (adiosShape.size() != nDim)
            {
                std::cerr << "[ERROR] Step " << stepIdx
                          << ": variable has " << adiosShape.size()
                          << " dims, expected " << nDim << ".\n";
                engine.EndStep();
                ++stepIdx;
                continue;
            }

            std::vector<size_t> shape(adiosShape.begin(), adiosShape.end());
            size_t nElem = 1;
            for (auto s : shape) nElem *= s;

            // 读取数据
            std::vector<double> buf(nElem);
            engine.Get(var, buf.data(), adios2::Mode::Sync);

            flatData.insert(flatData.end(), buf.begin(), buf.end());
            allShapes.push_back(shape);

            engine.EndStep();
            ++stepIdx;
        }

        engine.Close();
    }

    if (allShapes.empty())
    {
        std::cerr << "[Rank " << world_rank << "] No valid steps found.\n";
        MPI_Finalize();
        return 1;
    }

    // =========================================================
    // 3. 计算全局误差界（基于所有读取数据的值域）
    // =========================================================
    double local_min = *std::min_element(flatData.begin(), flatData.end());
    double local_max = *std::max_element(flatData.begin(), flatData.end());

    double global_min, global_max;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double error_bound = relative_error_bound * (global_max - global_min);

    if (world_rank == 0)
        std::cout << "[Info] Value range: [" << global_min << ", " << global_max << "]\n"
                  << "[Info] Absolute error bound: " << error_bound << "\n";

    // =========================================================
    // 4. 准备输出目录
    // =========================================================
    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');

    std::string baseName = std::filesystem::path(inputFileName).filename().string();
    std::string subDir   = "/home/nyan/scidx/scidx/"
                         + baseName + "_" + safeVarName + "_"
                         + std::to_string(beginStepNum) + "_"
                         + std::to_string(endStepNum)   + "_"
                         + std::to_string(extraValue)
                         + "_ZFP_autoBlock_2/";

    std::filesystem::create_directories(subDir);

    std::string dataFile = subDir + "compressed_data.bin";
    std::string sizeFile = subDir + "compressed_sizes.bin";
    std::string metaFile = subDir + "meta.bin";           // 记录每个 step 的形状

    // =========================================================
    // 5. 逐 step 调用 ZFP 压缩
    //    - 数据保持 C 行优先原始顺序（不手动分块）
    //    - zfp_field_Nd 的维度参数按 ZFP 约定反转
    //      ZFP 约定：第一个参数 nx 对应内存中变化最快的维度
    //      C 行优先：shape[nDim-1] 是最快维度
    //      → 传入 shape[nDim-1], shape[nDim-2], ..., shape[0]
    // =========================================================
    std::ofstream outData(dataFile, std::ios::binary);
    if (!outData)
    {
        std::cerr << "[ERROR] Cannot open " << dataFile << " for writing.\n";
        MPI_Finalize();
        return 1;
    }

    std::vector<size_t> compressedSizes;
    compressedSizes.reserve(allShapes.size());

    double compressTime = 0.0;
    double writeTime    = 0.0;

    size_t dataOffset = 0;

    for (size_t sIdx = 0; sIdx < allShapes.size(); ++sIdx)
    {
        const auto& shape = allShapes[sIdx];

        size_t nElem = 1;
        for (auto s : shape) nElem *= s;

        double* ptr = flatData.data() + dataOffset;

        // ---- 创建 ZFP field（维度顺序反转：ZFP nx = 最快变化维度）----
        auto t0 = std::chrono::high_resolution_clock::now();

        zfp_field* field = nullptr;
        if (nDim == 1)
        {
            // shape[0] 唯一维度，直接传入
            field = zfp_field_1d(ptr, zfp_type_double, shape[0]);
        }
        else if (nDim == 2)
        {
            // C 行优先：shape[1] 最快
            // ZFP: zfp_field_2d(data, type, nx, ny)  nx=最快
            field = zfp_field_2d(ptr, zfp_type_double,
                                 shape[1],   // nx = 最快变化维度
                                 shape[0]);  // ny
        }
        else // nDim == 3
        {
            // C 行优先：shape[2] 最快，shape[0] 最慢
            // ZFP: zfp_field_3d(data, type, nx, ny, nz)  nx=最快
            field = zfp_field_3d(ptr, zfp_type_double,
                                 shape[2],   // nx = 最快变化维度
                                 shape[1],   // ny
                                 shape[0]);  // nz = 最慢变化维度
        }

        if (!field)
        {
            std::cerr << "[ERROR] zfp_field creation failed at step " << sIdx << "\n";
            MPI_Finalize();
            return 1;
        }

        // ---- 创建 ZFP 流并设置误差界 ----
        zfp_stream* zfp = zfp_stream_open(nullptr);
        zfp_stream_set_accuracy(zfp, error_bound);   // 使用 accuracy 模式

        // ---- 分配压缩缓冲区 ----
        size_t bufsize = zfp_stream_maximum_size(zfp, field);
        void*  buf     = malloc(bufsize);
        if (!buf)
        {
            std::cerr << "[ERROR] malloc failed for step " << sIdx << "\n";
            zfp_field_free(field);
            zfp_stream_close(zfp);
            MPI_Finalize();
            return 1;
        }

        // ---- 关联比特流 ----
        bitstream* bs = stream_open(buf, bufsize);
        zfp_stream_set_bit_stream(zfp, bs);
        zfp_stream_rewind(zfp);

        // ---- 执行压缩 ----
        size_t zfpsize = zfp_compress(zfp, field);
        if (zfpsize == 0)
        {
            std::cerr << "[ERROR] zfp_compress returned 0 at step " << sIdx << "\n";
            free(buf);
            stream_close(bs);
            zfp_field_free(field);
            zfp_stream_close(zfp);
            MPI_Finalize();
            return 1;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        compressTime += std::chrono::duration<double>(t1 - t0).count();

        // ---- 写入压缩数据 ----
        auto tw0 = std::chrono::high_resolution_clock::now();
        outData.write(reinterpret_cast<const char*>(buf), zfpsize);
        auto tw1 = std::chrono::high_resolution_clock::now();
        writeTime += std::chrono::duration<double>(tw1 - tw0).count();

        compressedSizes.push_back(zfpsize);

        // ---- 清理 ----
        free(buf);
        stream_close(bs);
        zfp_field_free(field);
        zfp_stream_close(zfp);

        std::cout << "[Rank " << world_rank << "] Step " << sIdx
                  << " shape=(";
        for (size_t d = 0; d < nDim; ++d)
            std::cout << shape[d] << (d + 1 < nDim ? "," : "");
        std::cout << ") original=" << (nElem * sizeof(double))
                  << " bytes  compressed=" << zfpsize << " bytes"
                  << "  ratio=" << (100.0 * zfpsize / (nElem * sizeof(double))) << "%\n";

        dataOffset += nElem;
    }

    outData.close();

    // =========================================================
    // 6. 写入元数据文件
    //    格式：[stepCount(size_t)]
    //          [compressedSize_0, ..., compressedSize_N-1  (size_t[])]
    //          [nDim(size_t)]
    //          [shape_0_0, shape_0_1, ..., shape_0_{nDim-1},
    //           shape_1_0, ...,  (size_t[], stepCount * nDim 个元素)]
    // =========================================================
    {
        auto tw0 = std::chrono::high_resolution_clock::now();

        std::ofstream outMeta(metaFile, std::ios::binary);
        size_t stepCount = allShapes.size();

        outMeta.write(reinterpret_cast<const char*>(&stepCount),
                      sizeof(size_t));
        outMeta.write(reinterpret_cast<const char*>(compressedSizes.data()),
                      stepCount * sizeof(size_t));
        outMeta.write(reinterpret_cast<const char*>(&nDim),
                      sizeof(size_t));
        for (const auto& shape : allShapes)
            outMeta.write(reinterpret_cast<const char*>(shape.data()),
                          nDim * sizeof(size_t));
        outMeta.close();

        auto tw1 = std::chrono::high_resolution_clock::now();
        writeTime += std::chrono::duration<double>(tw1 - tw0).count();
    }

    // =========================================================
    // 7. 统计输出
    // =========================================================
    size_t totalCompressed = 0;
    for (auto s : compressedSizes) totalCompressed += s;
    size_t totalOriginal = flatData.size() * sizeof(double);

    std::cout << "\n[Rank " << world_rank << "] ===== Summary =====\n"
              << "[Rank " << world_rank << "] Steps processed   : " << allShapes.size() << "\n"
              << "[Rank " << world_rank << "] Original size     : " << totalOriginal << " bytes\n"
              << "[Rank " << world_rank << "] Compressed size   : " << totalCompressed << " bytes\n"
              << "[Rank " << world_rank << "] Compression ratio : "
              << (100.0 * totalCompressed / totalOriginal) << "%\n"
              << "[Rank " << world_rank << "] Compress time     : " << compressTime << " s\n"
              << "[Rank " << world_rank << "] Write time        : " << writeTime << " s\n";

    MPI_Finalize();
    return 0;
}
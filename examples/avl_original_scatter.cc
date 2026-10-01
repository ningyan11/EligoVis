#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <limits>

#include <mpi.h>
#include <adios2.h>

// ============================================================
// Save CSP image
// ============================================================
void saveCSPImage(const std::vector<std::vector<double>>& cspImage,
                  const std::string& outFile)
{
    std::ofstream ofs(outFile, std::ios::binary);
    if (!ofs) { std::cerr << "Error: cannot open " << outFile << std::endl; return; }
    size_t res_f = cspImage.size();
    size_t res_g = res_f > 0 ? cspImage[0].size() : 0;
    ofs.write(reinterpret_cast<const char*>(&res_f), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&res_g), sizeof(size_t));
    for (const auto& row : cspImage)
        ofs.write(reinterpret_cast<const char*>(row.data()), row.size()*sizeof(double));
    ofs.close();
    std::cout << "[CSP] Saved CSP image (" << res_f << " x " << res_g
              << ") to " << outFile << std::endl;
}

// ============================================================
// CSP computation - raw data version
// ============================================================
std::vector<std::vector<double>> computeCSP_raw(
    const std::vector<double>& dataF,
    const std::vector<double>& dataG,
    const std::vector<size_t>& dataShape,  // [Nx, Ny, Nz]
    double f_min, double f_max,
    double g_min, double g_max,
    size_t csp_res_f,
    size_t csp_res_g)
{
    size_t Nx = dataShape[0];
    size_t Ny = dataShape[1];
    size_t Nz = dataShape[2];

    // 全局坐标 → 线性索引，和原始数据存储顺序一致
    auto idx = [&](size_t gx, size_t gy, size_t gz) -> size_t {
        return gx + gy * Nx + gz * Nx * Ny;
    };

    std::vector<std::vector<double>> cspImage(csp_res_f,
                                              std::vector<double>(csp_res_g, 0.0));
    double df = (f_max - f_min) / csp_res_f;
    double dg = (g_max - g_min) / csp_res_g;

    for (size_t gz = 0; gz < Nz - 1; ++gz)
    for (size_t gy = 0; gy < Ny - 1; ++gy)
    for (size_t gx = 0; gx < Nx - 1; ++gx)
    {
        double fv[8] = {
            dataF[idx(gx,   gy,   gz)],
            dataF[idx(gx+1, gy,   gz)],
            dataF[idx(gx+1, gy+1, gz)],
            dataF[idx(gx,   gy+1, gz)],
            dataF[idx(gx,   gy,   gz+1)],
            dataF[idx(gx+1, gy,   gz+1)],
            dataF[idx(gx+1, gy+1, gz+1)],
            dataF[idx(gx,   gy+1, gz+1)]
        };
        double gv[8] = {
            dataG[idx(gx,   gy,   gz)],
            dataG[idx(gx+1, gy,   gz)],
            dataG[idx(gx+1, gy+1, gz)],
            dataG[idx(gx,   gy+1, gz)],
            dataG[idx(gx,   gy,   gz+1)],
            dataG[idx(gx+1, gy,   gz+1)],
            dataG[idx(gx+1, gy+1, gz+1)],
            dataG[idx(gx,   gy+1, gz+1)]
        };

        // Cell-level early rejection
        double cell_fmin = *std::min_element(fv, fv+8);
        double cell_fmax = *std::max_element(fv, fv+8);
        double cell_gmin = *std::min_element(gv, gv+8);
        double cell_gmax = *std::max_element(gv, gv+8);
        if (cell_fmax < f_min || cell_fmin > f_max) continue;
        if (cell_gmax < g_min || cell_gmin > g_max) continue;

        // Gradient
        double grad_fx = 0.25*((fv[1]-fv[0])+(fv[2]-fv[3])+(fv[5]-fv[4])+(fv[6]-fv[7]));
        double grad_fy = 0.25*((fv[3]-fv[0])+(fv[2]-fv[1])+(fv[7]-fv[4])+(fv[6]-fv[5]));
        double grad_fz = 0.25*((fv[4]-fv[0])+(fv[5]-fv[1])+(fv[6]-fv[2])+(fv[7]-fv[3]));
        double grad_gx = 0.25*((gv[1]-gv[0])+(gv[2]-gv[3])+(gv[5]-gv[4])+(gv[6]-gv[7]));
        double grad_gy = 0.25*((gv[3]-gv[0])+(gv[2]-gv[1])+(gv[7]-gv[4])+(gv[6]-gv[5]));
        double grad_gz = 0.25*((gv[4]-gv[0])+(gv[5]-gv[1])+(gv[6]-gv[2])+(gv[7]-gv[3]));

        // |∇f × ∇g|
        double cross_x = grad_fy*grad_gz - grad_fz*grad_gy;
        double cross_y = grad_fz*grad_gx - grad_fx*grad_gz;
        double cross_z = grad_fx*grad_gy - grad_fy*grad_gx;
        double cross_mag = std::sqrt(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z);
        if (cross_mag < 1e-12) continue;

        // Cell center
        double fc = 0.0, gc = 0.0;
        for (int i = 0; i < 8; i++) { fc += fv[i]; gc += gv[i]; }
        fc /= 8.0; gc /= 8.0;

        double fc_c = std::max(f_min, std::min(f_max, fc));
        double gc_c = std::max(g_min, std::min(g_max, gc));
        size_t bin_f = static_cast<size_t>((fc_c - f_min) / df);
        size_t bin_g = static_cast<size_t>((gc_c - g_min) / dg);
        if (bin_f >= csp_res_f) bin_f = csp_res_f - 1;
        if (bin_g >= csp_res_g) bin_g = csp_res_g - 1;

        cspImage[bin_f][bin_g] += 1.0 / cross_mag;
    }

    return cspImage;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[])
{
    auto totalStart = std::chrono::high_resolution_clock::now();

    MPI_Init(&argc, &argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // --------------------------------------------------------
    // Parameter parsing
    // --------------------------------------------------------
    std::string inputFileName, varNameF, varNameG;
    size_t nDim = 0;
    std::vector<size_t> dataShape;
    size_t beginStepNum = 0, endStepNum = 0;
    std::vector<double> queryRangeF, queryRangeG;
    size_t csp_res = 256;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--input_file"      && i+1 < argc) inputFileName = argv[++i];
        else if (arg == "--variable_name_f" && i+1 < argc) varNameF = argv[++i];
        else if (arg == "--variable_name_g" && i+1 < argc) varNameG = argv[++i];
        else if (arg == "--dimensions"      && i+1 < argc) nDim = std::stoul(argv[++i]);
        else if (arg == "--begin_step"      && i+1 < argc) beginStepNum = std::stoul(argv[++i]);
        else if (arg == "--end_step"        && i+1 < argc) endStepNum = std::stoul(argv[++i]);
        else if (arg == "--data_shape"      && i+nDim < argc)
            for (size_t j = 0; j < nDim; j++) dataShape.push_back(std::stoul(argv[++i]));
        else if (arg == "--query_range_f"   && i+2 < argc) {
            queryRangeF.push_back(std::stod(argv[++i]));
            queryRangeF.push_back(std::stod(argv[++i]));
        }
        else if (arg == "--query_range_g"   && i+2 < argc) {
            queryRangeG.push_back(std::stod(argv[++i]));
            queryRangeG.push_back(std::stod(argv[++i]));
        }
        else if (arg == "--csp_res"         && i+1 < argc) csp_res = std::stoul(argv[++i]);
    }

    if (inputFileName.empty() || varNameF.empty() || varNameG.empty() ||
        nDim == 0 || dataShape.size() != nDim ||
        queryRangeF.size() != 2 || queryRangeG.size() != 2)
    {
        if (world_rank == 0)
            std::cerr << "Usage: --input_file <f> --variable_name_f <vf> "
                         "--variable_name_g <vg> --dimensions <n> "
                         "--data_shape s1 s2 s3 "
                         "--begin_step <b> --end_step <e> "
                         "--query_range_f <lo> <hi> "
                         "--query_range_g <lo> <hi> "
                         "[--csp_res <r>]\n";
        MPI_Finalize();
        return 1;
    }

    auto afterParseTime = std::chrono::high_resolution_clock::now();
    std::cerr << "[Time1] Preprocessing: "
              << std::chrono::duration<double>(afterParseTime - totalStart).count()
              << " seconds" << std::endl;

    // --------------------------------------------------------
    // Read raw data via ADIOS2
    // --------------------------------------------------------
    std::vector<double> dataF, dataG;

    auto t_read_start = std::chrono::high_resolution_clock::now();
    {
        adios2::ADIOS adios;
        adios2::IO    io     = adios.DeclareIO("ReaderIO");
        adios2::Engine engine = io.Open(inputFileName, adios2::Mode::Read);

        size_t stepIdx = 0;
        while (engine.BeginStep() == adios2::StepStatus::OK)
        {
            if (stepIdx < beginStepNum || stepIdx > endStepNum)
            {
                engine.EndStep();
                ++stepIdx;
                continue;
            }

            // Read F
            auto varF = io.InquireVariable<double>(varNameF);
            if (varF)
            {
                size_t nElem = 1;
                for (auto s : varF.Shape()) nElem *= s;
                std::vector<double> buf(nElem);
                engine.Get(varF, buf.data(), adios2::Mode::Sync);
                dataF.insert(dataF.end(), buf.begin(), buf.end());
            }
            else
            {
                std::cerr << "[WARN] Step " << stepIdx
                          << ": variable '" << varNameF << "' not found.\n";
            }

            // Read G
            auto varG = io.InquireVariable<double>(varNameG);
            if (varG)
            {
                size_t nElem = 1;
                for (auto s : varG.Shape()) nElem *= s;
                std::vector<double> buf(nElem);
                engine.Get(varG, buf.data(), adios2::Mode::Sync);
                dataG.insert(dataG.end(), buf.begin(), buf.end());
            }
            else
            {
                std::cerr << "[WARN] Step " << stepIdx
                          << ": variable '" << varNameG << "' not found.\n";
            }

            engine.EndStep();
            ++stepIdx;
        }
        engine.Close();
    }
    auto t_read_end = std::chrono::high_resolution_clock::now();
    double readTime = std::chrono::duration<double>(t_read_end - t_read_start).count();
    std::cerr << "[Time2] Data Read (F+G): " << readTime << " seconds" << std::endl;

    if (dataF.empty() || dataG.empty())
    {
        std::cerr << "[ERROR] No data read.\n";
        MPI_Finalize();
        return 1;
    }

    // --------------------------------------------------------
    // CSP computation
    // --------------------------------------------------------
    auto t_csp_start = std::chrono::high_resolution_clock::now();

    auto cspImage = computeCSP_raw(
        dataF, dataG,
        dataShape,
        queryRangeF[0], queryRangeF[1],
        queryRangeG[0], queryRangeG[1],
        csp_res, csp_res);

    auto t_csp_end = std::chrono::high_resolution_clock::now();
    double cspTime = std::chrono::duration<double>(t_csp_end - t_csp_start).count();
    std::cerr << "[Time3] CSP Computation: " << cspTime << " seconds" << std::endl;

    // --------------------------------------------------------
    // Save result
    // --------------------------------------------------------
    std::string safeF = varNameF; std::replace(safeF.begin(), safeF.end(), '/', '_');
    std::string safeG = varNameG; std::replace(safeG.begin(), safeG.end(), '/', '_');
    std::string outFile = "csp_" + safeF + "_" + safeG + "_original.bin";
    saveCSPImage(cspImage, outFile);

    // --------------------------------------------------------
    // Time summary
    // --------------------------------------------------------
    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(totalEnd - totalStart).count();

    std::cerr << "\n========== TIME SUMMARY (RAW) ==========" << std::endl;
    std::cerr << "[Stage 1] Preprocessing:   "
              << std::chrono::duration<double>(afterParseTime - totalStart).count() << " s" << std::endl;
    std::cerr << "[Stage 2] Data Read (F+G): " << readTime  << " s" << std::endl;
    std::cerr << "[Stage 3] CSP Computation: " << cspTime   << " s" << std::endl;
    std::cerr << "[TOTAL]                    " << totalTime << " s" << std::endl;
    std::cerr << "=========================================" << std::endl;

    MPI_Finalize();
    return 0;
}
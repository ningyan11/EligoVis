#include <adios2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>

// 数据与块维度
const size_t Nx = 1024, Ny = 1024, Nz = 1024;
const size_t sx = 4, sy = 4, sz = 4;
const size_t bx = Nx / sx, by = Ny / sy, bz = Nz / sz;

// 计算 Pearson correlation
double computePearson(const std::vector<double> &a, const std::vector<double> &b) {
    size_t n = a.size();
    double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / n;
    double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / n;

    double num = 0, denom_a = 0, denom_b = 0;
    for (size_t i = 0; i < n; ++i) {
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    return num / std::sqrt(denom_a * denom_b + 1e-10);  // 防止除 0
}

int main() {
    const std::string file = "/expanse/lustre/scratch/sdi/temp_project/openpmd.bp";
    const std::string varname = "/data/fields/B/y";

    adios2::ADIOS ad;
    adios2::IO io = ad.DeclareIO("reader");
    adios2::Engine reader = io.Open(file, adios2::Mode::Read);

    size_t step = 0;
    while (reader.BeginStep() == adios2::StepStatus::OK) {
        if (step % 2 == 0 && step <= 100) {  // ✅ 严格每隔2个 step，0 到 100 共 51 个
            auto var = io.InquireVariable<double>(varname);
            var.SetShape({Nz, Ny, Nx});
            std::vector<double> data(Nx * Ny * Nz);
            reader.Get(var, data.data(), adios2::Mode::Sync);

            std::vector<double> mins, maxs;
            mins.reserve(bx * by * bz);
            maxs.reserve(bx * by * bz);

            for (size_t z = 0; z < bz; ++z)
            for (size_t y = 0; y < by; ++y)
            for (size_t x = 0; x < bx; ++x) {
                double minval = std::numeric_limits<double>::max();
                double maxval = -std::numeric_limits<double>::max();

                for (size_t dz = 0; dz < sz; ++dz)
                for (size_t dy = 0; dy < sy; ++dy)
                for (size_t dx = 0; dx < sx; ++dx) {
                    size_t gx = x * sx + dx;
                    size_t gy = y * sy + dy;
                    size_t gz = z * sz + dz;
                    size_t idx = gz * Ny * Nx + gy * Nx + gx;
                    double val = data[idx];
                    minval = std::min(minval, val);
                    maxval = std::max(maxval, val);
                }

                mins.push_back(minval);
                maxs.push_back(maxval);
            }

            double corr = computePearson(mins, maxs);
            std::cout << "Step " << step << ": Pearson correlation = " << corr << std::endl;
        }
        reader.EndStep();
        ++step;
    }

    reader.Close();
    return 0;
}

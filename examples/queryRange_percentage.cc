#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <adios2.h>
#include <filesystem>

int main(int argc, char *argv[]) {
    std::string inputFileName;
    std::string variableName;
    size_t nDim = 0;
    size_t beginStepNum = 0;
    size_t endStepNum = 0;
    std::vector<size_t> targetSteps;

    // 解析命令行参数
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc)
            inputFileName = argv[++i];
        else if (arg == "--variable_name" && i + 1 < argc)
            variableName = argv[++i];
        else if (arg == "--dimensions" && i + 1 < argc)
            nDim = std::stoul(argv[++i]);
        else if (arg == "--begin_step" && i + 1 < argc)
            beginStepNum = std::stoul(argv[++i]);
        else if (arg == "--end_step" && i + 1 < argc)
            endStepNum = std::stoul(argv[++i]);
        else if (arg == "--target_steps")
            {
                while (i + 1 < argc && std::isdigit(argv[i + 1][0])) {
                    targetSteps.push_back(std::stoul(argv[++i]));
                }
        }
    }

    if (inputFileName.empty() || variableName.empty() || nDim == 0) {
        std::cerr << "Missing required arguments." << std::endl;
        return 1;
    }

    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader = io.Open(inputFileName, adios2::Mode::Read);

    std::vector<double> flatData;  // 所有 step 的数据

    size_t step = 0;
    std::set<size_t> targetSet(targetSteps.begin(), targetSteps.end());

    while (reader.BeginStep() == adios2::StepStatus::OK) {
        //固定step才读取
        if (!targetSet.empty() && targetSet.find(step) == targetSet.end()) 
        {
            reader.EndStep();
            step++;
            continue;
        }

        auto var = io.InquireVariable<double>(variableName);
        if (!var) {
            std::cerr << "Variable not found." << std::endl;
            return 1;
        }

        std::vector<size_t> shape = var.Shape();
        size_t varElements = 1;
        for (size_t i = 0; i < nDim; i++)
            varElements *= shape[i];

        std::vector<double> stepData(varElements);
        reader.Get(var, stepData.data(), adios2::Mode::Sync);
        flatData.insert(flatData.end(), stepData.begin(), stepData.end());

        reader.EndStep();
        step++;
       
    }

    reader.Close();

    std::cout << "Total values loaded: " << flatData.size() << std::endl;

    std::sort(flatData.begin(), flatData.end(), std::greater<double>());

    std::vector<double> percentages = {0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1 };
    double max_val = flatData[0];
    for (double p : percentages) {
        size_t index = static_cast<size_t>(p * flatData.size());
        if (index >= flatData.size()) index = flatData.size() - 1;
        double threshold = flatData[index];
        std::cout << "Top " << p * 100 << "% value range: [" << threshold << ", " << max_val << "]" << std::endl;
    }


    return 0;
}

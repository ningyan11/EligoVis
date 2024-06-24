
#include <vector>
#include <queue>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>

#include <random>

#include <adios2.h>


#include <scidx.h>

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());

// Function to generate a random interval
template <typename T>
ScidxInterval<T> generateRandomInterval(T maxLow, T maxHigh) {
    std::uniform_real_distribution<T> distLow(0, maxLow);
    std::uniform_real_distribution<T> distHigh(distLow(gen), maxHigh);

    ScidxInterval<T> interval;
    interval.low = distLow(gen);
    interval.high = distHigh(gen);

    return interval;
}

template <typename T>
bool containsDuplicateValues(const std::vector<T>& vec) {
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        auto count = std::count(vec.begin(), vec.end(), *it);
        if (count > 1) {
            return true;
        }
    }
    return false;
}

int main(int argc, char *argv[]) {

    std::string inputFileName;
    std::string variableName;
    std::string variableType;

    std::vector<size_t> blockShape;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--input_file")
        {
            if (i+1 < argc)
            {
                inputFileName = argv[i+1];
            }
            else
            {
                std::cerr << "--input_file option requires one argument." << std::endl;
                return 1;
            }            
        }
        else if (arg == "--variable_name")
        {
            if (i+1 < argc)
            {
                variableName = argv[i+1];
            }
            else
            {
                std::cerr << "--variable_name option requires one argument." << std::endl;
                return 1;
            }             
        }
        
    }

    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine = reader_io.Open(inputFileName, adios2::Mode::Read);
    size_t steps = reader_engine.Steps();
    //std::cout << "total steps: " << steps << std::endl;
    
    auto var = reader_io.InquireVariable(variableName);
    variableType= reader_io.VariableType(variableName);

    //std::cout << "variable name: " << variableName << ", variable type: " << variableType << std::endl;

    if (variableType == "double")
    {
        size_t blockID = 0;
        std::vector<double> block_mins;
        std::vector<double> block_maxs;
        for (size_t step = 1; step < steps; step++)
        {
            auto blocksInfo = reader_engine.AllStepsBlocksInfo(var).at(step);
            for (const auto &info : blocksInfo)
            {
                // std::cout << "    block " << blockID << " offset = ";
                // for (size_t i = 0; i < info.Start.size(); i++)
                // {
                //     std::cout << info.Start[i] << " ";
                // }
                size_t blockSize = 1;
                // std::cout << " size = ";
                for (size_t i = 0; i < info.Count.size(); i++)
                {
                    // std::cout << info.Count[i] << " ";
                    blockSize *= info.Count[i];
                }  
                          
                std::vector<double> blockData(blockSize);
                var.SetSelection({info.Start, info.Count});
                var.SetStepSelection({step, 1});
                reader_engine.Get(var, blockData.data(), adios2::Mode::Sync); 
                auto minmax = minmax_element(blockData.begin(), blockData.end());
                block_mins.push_back(*minmax.first);
                block_maxs.push_back(*minmax.second);
                // std::cout << "min = " << *minmax.first << " max = " << *minmax.second;
                // std::cout << std::endl; 
                blockID++;     
            }

        }
        std::vector<ScidxInterval<double>> intervals;

        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::min();
        for (size_t i = 0; i < block_mins.size(); i++)
        {
            if (block_mins[i] < global_min)
            {
                global_min = block_mins[i];
            }
            if (block_maxs[i] > global_max)
            {
                global_max = block_maxs[i];
            }        
            
            ScidxInterval<double> interval;
            interval.low = block_mins[i];
            interval.high = block_maxs[i];
            intervals.push_back(interval);
        }
        ScidxAVLIntervalTree<double> avlIntervalTree;

        std::vector<ScidxInterval<double>> zeroLowIntervalList;

        for (size_t i = 0; i < intervals.size(); i++)
        {
            std::cout << i << " [" << intervals[i].low << " " << intervals[i].high << "]" << std::endl;
            if (intervals[i].low == 0)
            {
                zeroLowIntervalList.push_back(intervals[i]);
                continue;
            }
            
            avlIntervalTree.insertNode(i, intervals[i]);
        }
        std::cout << "AVL Interval Tree after insertions:" << std::endl;
        avlIntervalTree.display();

        for (size_t i = 0; i < zeroLowIntervalList.size(); i++)
        {
            std::cout << " [" << zeroLowIntervalList[i].low << " " << zeroLowIntervalList[i].high << "]" << std::endl;
        }
        
    }
    
    

    reader_engine.Close();

    

    return 0;
}

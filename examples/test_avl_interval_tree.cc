
#include <vector>
#include <queue>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>

#include <random>

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

    char *inputFileName = NULL;
    size_t nDim = 0;
    std::vector<size_t> dataShape;
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
                std::cerr << "--input option requires one argument." << std::endl;
                return 1;
            }            
        }
        else if (arg == "--dimensions")
        {
            if (i+1 < argc)
            {
                std::stringstream ss_dim(argv[i+1]);
                ss_dim >> nDim;
            }
            else
            {
                std::cerr << "--dimensions option requires one argument." << std::endl;
                return 1;
            } 
        }
        else if (arg == "--data_shape")
        {
            if (nDim)
            {
                if ((int)(i+nDim) < argc)
                {
                    for (size_t j = i+1; j < i+1+nDim; j++)
                    {
                        dataShape.push_back(atoi(argv[j]));
                    }
                    
                }
                else
                {
                    std::cerr << "--data_shape option requires [# of dimensions] argument." << std::endl;
                    return 1;
                }  
            }
            else
            {
                std::cerr << "# of dimensions must be greater than 0." << std::endl;
                return 1;                
            }
        }
        else if (arg == "--block_shape")
        {
            if (nDim)
            {
                if ((int)(i+nDim) < argc)
                {
                    for (size_t j = i+1; j < i+1+nDim; j++)
                    {
                        blockShape.push_back(atoi(argv[j]));
                    }
                    
                }
                else
                {
                    std::cerr << "--blockShape option requires [# of dimensions] argument." << std::endl;
                    return 1;
                }  
            }
        }
        
    }

    size_t nElem;
    int status;
    float *dataBuffer = scidx_readFloatData(inputFileName, &nElem, &status);

    std::cout << "read in " << nElem << " data elements." << std::endl;

    std::vector<float> data{dataBuffer, dataBuffer+nElem};

    std::vector<std::vector<float>> blockMinMax = obtainBlockMinMax(data, dataShape, blockShape);
    
	
    std::vector<ScidxInterval<float>> intervals;
    std::vector<float> all_lows;

    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::min();
    for (size_t i = 0; i < blockMinMax.size(); i++)
    {
        if (blockMinMax[i][0] < global_min)
        {
            global_min = blockMinMax[i][0];
        }
        if (blockMinMax[i][1] > global_max)
        {
            global_max = blockMinMax[i][1];
        }        
        
        ScidxInterval<float> interval;
        interval.low = blockMinMax[i][0];
        interval.high = blockMinMax[i][1];
        intervals.push_back(interval);
        //std::cout << interval.low << ", " << interval.high << std::endl;
        all_lows.push_back(interval.low);
    }

    if (containsDuplicateValues(all_lows))
    {
        std::cout << "multiple intervals have the same low end value" << std::endl;
    }
    else
    {
        std::cout << "all intervals have unique low end value" << std::endl;
    }
    
    
    ScidxAVLIntervalTree<float> avlIntervalTree;
     //因为min值相同而未被插入的点
     std::vector<SkippedNode<float>> skippedIntervals;

    for (size_t i = 0; i < intervals.size(); i++)
    {
        //std::cout << intervals[i].low << " " << intervals[i].high << std::endl;
        avlIntervalTree.insertNode(i, intervals[i], skippedIntervals);
    }
    std::cout << "AVL Interval Tree after insertions:" << std::endl;
    avlIntervalTree.display();

    const int numberQueryOfIntervals = 20;
    const float maxLow = global_min;
    const float maxHigh = global_max;

    for (size_t i = 0; i < numberQueryOfIntervals; i++)
    {
        ScidxInterval<float> queryInterval = generateRandomInterval(maxLow, maxHigh);
        std::cout << "query interval: [" << queryInterval.low << ", " << queryInterval.high << "]" << std::endl;
        std::vector<ScidxAVLNode<float>*> result = avlIntervalTree.query(queryInterval);
        std::cout << "overlapped intervals: " << std::endl;
        for (size_t j = 0; j < result.size(); j++)
        {
            std::cout << "    [" << result[j]->interval.low << ", " << result[j]->interval.high << "] (id: " << result[j]->id << ")" << std::endl;
        }
        
    }
    

    return 0;
}

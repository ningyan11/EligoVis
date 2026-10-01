
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <random>
#include <adios2.h> 
#include <scidx_avl.h> 
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <utility>  // for std::pair
#include <fstream> 

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

void listAllVariables(adios2::IO &io) {
    std::cout << "Listing all variables in the file:" << std::endl;
    for (const auto &variablePair : io.AvailableVariables()) {
        std::cout << "Variable Name: " << variablePair.first << std::endl;
        for (const auto &attributePair : variablePair.second) {
            std::cout << "  Attribute: " << attributePair.first << " = " << attributePair.second << std::endl;
        }
    }
}

template <typename T>
ScidxAVLNode<T>* attachSubTreesBFS(const std::vector<bool>& fullTreeVectorOfMap, const std::vector<ScidxAVLNode<T>*>& compressedSubTrees);

std::pair<int, int> getBinIndices(double low, double high, double lowMin, double lowMax, double highMin, double highMax, int lowBinCount, int highBinCount);

template <typename T>
void calculateConditionalProbability(const std::vector<ScidxInterval<T>>& intervals, int lowBinCount, int highBinCount, const std::string& csvFile);

void exportCountMatrixToCSV(const std::vector<std::vector<int>>& countMatrix, const std::string& filename);

std::vector<ScidxInterval<double>> readIntervalsFromCSV(const std::string& filename);


int main() {
    adios2::ADIOS adios;

    // 读取 CSV 文件并获取区间数据
    std::string filename = "51_60_intervals4.csv";
    std::vector<ScidxInterval<double>> intervals = readIntervalsFromCSV(filename);


    /*std::vector<ScidxInterval<double>> intervals;

    // List of files to read
    std::vector<std::string> filenames = {
        "1_10_intervals.csv", "11_20_intervals.csv", "21_30_intervals.csv",
        "31_40_intervals.csv", "41_50_intervals.csv", "51_60_intervals.csv",
        "61_70_intervals.csv", "71_80_intervals.csv", "81_90_intervals.csv", "91_100_intervals.csv"
    };

    // Read each file and append the intervals to the all_intervals vector
    for (const auto& filename : filenames) {
        std::vector<ScidxInterval<double>> one_intervals = readIntervalsFromCSV(filename);
        intervals.insert(intervals.end(), one_intervals.begin(), one_intervals.end());
    }*/

    std::cout << "Total intervals read: " << intervals.size() << std::endl;


    calculateConditionalProbability(intervals, 1000, 1000, "low_high_bin_mapping.csv");

    std::cout << "CSV file saved as low_high_bin_mapping.csv" << std::endl;

    std::cout << "begin to build tree" << std::endl;

    adios2::IO iO = adios.DeclareIO("WriteData");
    std::string filenameAdios = "avl_output.bp";
    adios2::Engine bpWriter = iO.Open(filenameAdios, adios2::Mode::Write);
    bpWriter.BeginStep();

    ScidxAVLIntervalTree<double> avlIntervalTree; 
    std::vector<ScidxInterval<double>> zeroLowIntervalList;
    std::vector<SkippedNode<double>> skippedIntervals;

        std::cout << "interval size:" << intervals.size() << std::endl;
        for (size_t i = 0; i < intervals.size(); i++) {
            if (intervals[i].low == 0) {
                zeroLowIntervalList.push_back(intervals[i]);
                continue;
            }

            avlIntervalTree.insertNode(i, intervals[i],skippedIntervals);
        }

        std::cout << "AVL Interval Tree after insertions:" << std::endl;
        avlIntervalTree.display();

        int treeHeight = avlIntervalTree.getTreeHeight();
        std::cout << "Tree height after insertions: " << treeHeight << std::endl;


        int levelsToTraverse = treeHeight+1;

        //整个tree的map
        std::pair<std::vector<uint8_t>, int> allSubTreeMapPair = getAllSubTreesMap(avlIntervalTree, levelsToTraverse);
        std::vector<uint8_t> allSubTreeMap = allSubTreeMapPair.first;
        int remainingBits = allSubTreeMapPair.second;
        std::cout << "remainingBits "<< remainingBits << std::endl;

        int k = 1;
        std::string varNameAllSubTreeMap = std::to_string(k) + "_AllMap";
        std::string varNameAllSubTreeBits = std::to_string(k) + "_Bits";
        adios2::Variable<uint8_t> allSubTreeMapVar = iO.DefineVariable<uint8_t>(varNameAllSubTreeMap, {allSubTreeMap.size()}, {0}, {allSubTreeMap.size()}, adios2::ConstantDims); 
        adios2::Variable<int> allSubTreeBitVar = iO.DefineVariable<int>(varNameAllSubTreeBits);
        

        bpWriter.Put(allSubTreeMapVar, allSubTreeMap.data(), adios2::Mode::Sync);
        bpWriter.Put(allSubTreeBitVar, &remainingBits, adios2::Mode::Sync);

        std::vector<std::vector<std::vector<ScidxAVLNode<double>*>>> allSubTrees;
        std::vector<std::vector<ScidxAVLNode<double>*>> firstSubTreeNodesInLevels;
        //每一个subTree的map的大小
        std::vector<size_t> sizesOfBigMap;
        //一个大的vector，连续放入每个子树的map
        std::vector<int> combinedVector;
        std::vector<int> firstVectorOfMap;

        levelOrderTraversal(avlIntervalTree.getRoot(), levelsToTraverse, firstSubTreeNodesInLevels, firstVectorOfMap);
        

        allSubTrees.push_back(firstSubTreeNodesInLevels);
        sizesOfBigMap.push_back(firstVectorOfMap.size());
        combinedVector.insert(combinedVector.end(), firstVectorOfMap.begin(), firstVectorOfMap.end());


        std::vector<ScidxAVLNode<double>*>& lastLevelOfFirstSubTree = firstSubTreeNodesInLevels.back();
        std::vector<ScidxAVLNode<double>*> rootsOfNewSubTrees;
        rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfFirstSubTree.begin(), lastLevelOfFirstSubTree.end());

        while (!rootsOfNewSubTrees.empty()) {
            
            ScidxAVLNode<double>* node = rootsOfNewSubTrees.front();
            rootsOfNewSubTrees.erase(rootsOfNewSubTrees.begin());  

            if (node->left == nullptr && node->right == nullptr) {
                continue;
            }
            
            std::vector<std::vector<ScidxAVLNode<double>*>> currentSubTreeNodesInLevels;
            std::vector<int> currentSubTreeVectorOfMap;

            //TODO: why do levelOrderTraversal again?
            levelOrderTraversal(node, levelsToTraverse, currentSubTreeNodesInLevels, currentSubTreeVectorOfMap);
            
            sizesOfBigMap.push_back(currentSubTreeVectorOfMap.size());
            combinedVector.insert(combinedVector.end(), currentSubTreeVectorOfMap.begin(), currentSubTreeVectorOfMap.end());
            
            allSubTrees.push_back(currentSubTreeNodesInLevels);


            //level of subtree is smaller than levelsToTraverse
            if (currentSubTreeNodesInLevels.size() < levelsToTraverse) {
                continue;
            }

            std::vector<ScidxAVLNode<double>*>& lastLevelOfCurrentSubTree = currentSubTreeNodesInLevels.back();
            rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfCurrentSubTree.begin(), lastLevelOfCurrentSubTree.end());
        }

        float error_bound = 1E-4;
        std::vector<int> mergedCompressedTypesLow;

        std::vector<std::vector<int>> eachMapOfSubTree; 
        size_t startIndex = 0; 
        for (size_t size : sizesOfBigMap) {
            if (startIndex + size <= combinedVector.size()) {
                std::vector<int> subVector(combinedVector.begin() + startIndex, combinedVector.begin() + startIndex + size);
                eachMapOfSubTree.push_back(subVector);
                startIndex += size; 
            } else {
                std::cerr << "Error: The sizes in sizesOfBigMap exceed the size of combinedVector." << std::endl;
                break;
            }
        }

        if (allSubTrees.size() != eachMapOfSubTree.size()) {
            std::cerr << "Error: Mismatch in sizes of the vectors." << std::endl;
            return 1;
        }

        //每个subTree的map是连续写到combinedVector中去的，然后sizesOfBigMap记录了每个子树的大小
        std::string varNamesubSizeOfMap = std::to_string(k) + "_SizeSubMap";
        std::string varNamesubFullMap = std::to_string(k) + "_SubMap";

        adios2::Variable<size_t> subSizeOfMap = iO.DefineVariable<size_t>(varNamesubSizeOfMap, {sizesOfBigMap.size()}, {0}, {sizesOfBigMap.size()}, adios2::ConstantDims);
        adios2::Variable<int> subFullMap = iO.DefineVariable<int>(varNamesubFullMap, {combinedVector.size()}, {0}, {combinedVector.size()}, adios2::ConstantDims);
                
        bpWriter.Put(subSizeOfMap, sizesOfBigMap.data(), adios2::Mode::Sync);
        bpWriter.Put(subFullMap, combinedVector.data(), adios2::Mode::Sync);


        std::vector<ScidxAVLNode<double>*> compressedSubTrees; 

        std::vector<std::vector<int>> allTreeTypesLow;
        std::vector<std::vector<int>> allTreeTypesHigh;
        std::vector<std::vector<size_t>> allTreeId;
        std::vector<std::vector<int>> allTreeTypesMaxHigh;


        //begin to write the compressed data into adios

        std::cout << "allSubTrees.size " << allSubTrees.size() << std::endl;

        // Store the number of subtrees in ADIOS2 (consider whether this is necessary as you mentioned)
        std::string varNamesubTreeize = std::to_string(k) + "_NumSub";
        adios2::Variable<size_t> subTreeize = iO.DefineVariable<size_t>(varNamesubTreeize);

        size_t numSubTrees = allSubTrees.size();
        bpWriter.Put(subTreeize, &numSubTrees, adios2::Mode::Sync);
            
        for (size_t i = 0; i < allSubTrees.size(); ++i) {
            std::vector<std::vector<ScidxAVLNode<double>*>> currentSubTree = allSubTrees[i];
            computeTypeBufferAVL<double>(k, currentSubTree, error_bound, i, bpWriter, iO, allTreeTypesLow, allTreeTypesHigh, allTreeId, allTreeTypesMaxHigh);
        }

        std::cout << "Finished computeTypeBuffer" << std::endl;


        std::vector<scidx::HuffmanTree*> huffmanTrees = fullHuffmanAVL(k, allTreeTypesLow, allTreeTypesHigh, allTreeTypesMaxHigh, bpWriter, iO);

        std::cout << "allTreeTypesLow.size "<< allTreeTypesLow.size() << std::endl;
        std::cout << "k number "<< k << std::endl;
        
        for (size_t i = 0; i < allTreeTypesLow.size(); ++i) {
            std::vector<int> currentTypesLow = allTreeTypesLow[i];
            std::vector<int> currentTypesHigh = allTreeTypesHigh[i];
            std::vector<size_t> currentIds = allTreeId[i];
            std::vector<int> currentTypesMaxHigh = allTreeTypesMaxHigh[i];
            std::cout << "subTree number "<< i << std::endl;
            compressTree<double>(k, currentTypesLow, currentTypesHigh, currentIds, currentTypesMaxHigh, error_bound, bpWriter, iO, i, huffmanTrees[0], huffmanTrees[1], huffmanTrees[2]);
        }

    bpWriter.EndStep();
    bpWriter.Close();
    std::cout << "finish encode" << std::endl;
    return 0;

}

// 计算每个 low bin 和 high bin 的索引，返回 std::pair
std::pair<int, int> getBinIndices(double low, double high, double lowMin, double lowMax, double highMin, double highMax, int lowBinCount, int highBinCount) {
    int lowBin = std::floor((low - lowMin) / (lowMax - lowMin) * lowBinCount);
    int highBin = std::floor((high - highMin) / (highMax - highMin) * highBinCount);

    // 防止索引越界
    if (lowBin >= lowBinCount) lowBin = lowBinCount - 1;
    if (lowBin < 0) lowBin = 0;

    if (highBin >= highBinCount) highBin = highBinCount - 1;
    if (highBin < 0) highBin = 0;

    return std::make_pair(lowBin, highBin);  // 返回 std::pair
}

// 计算条件概率和期望值并保存到 CSV 文件
template <typename T>
void calculateConditionalProbability(const std::vector<ScidxInterval<T>>& intervals, int lowBinCount, int highBinCount, const std::string& csvFile) {
    // 1. 计算 low 和 high 的最小值和最大值
    T lowMin = intervals[0].low, lowMax = intervals[0].low;
    T highMin = intervals[0].high, highMax = intervals[0].high;

    for (const auto& interval : intervals) {
        if (interval.low < lowMin) lowMin = interval.low;
        if (interval.low > lowMax) lowMax = interval.low;
        if (interval.high < highMin) highMin = interval.high;
        if (interval.high > highMax) highMax = interval.high;
    }

    // 2. 初始化 2D 条件概率矩阵，大小为 [lowBinCount][highBinCount]
    std::vector<std::vector<int>> countMatrix(lowBinCount, std::vector<int>(highBinCount, 0));
    std::vector<int> lowBinTotalCount(lowBinCount, 0);

    // 3. 遍历每个区间，计算其 low 和 high 所在的 bin
    for (const auto& interval : intervals) {
        std::pair<int, int> binIndices = getBinIndices(interval.low, interval.high, lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount);
        int lowBin = binIndices.first;
        int highBin = binIndices.second;
        countMatrix[lowBin][highBin]++;
        lowBinTotalCount[lowBin]++;
    }


    std::string filename = "count_matrix.csv";
    std::cout << "Exporting count matrix to file: " << filename << std::endl;
    exportCountMatrixToCSV(countMatrix, filename);    


    // 4. 计算条件概率和期望值
    std::vector<std::pair<int, double>> nonZeroBins;  // 用于存储有非零值的 bin

    for (int i = 0; i < lowBinCount; ++i) {
        if (lowBinTotalCount[i] == 0) continue;  // 如果该 low bin 没有元素，跳过

        double expectedValue = 0.0;
        for (int j = 0; j < highBinCount; ++j) {
            double conditionalProbability = static_cast<double>(countMatrix[i][j]) / lowBinTotalCount[i];
            expectedValue += conditionalProbability * j;
        }

        if (expectedValue > 0) {
            nonZeroBins.emplace_back(i, expectedValue);
        }
    }

    // 5. 保存 lowMin, lowMax, highMin, highMax, lowBinCount, highBinCount 以及映射结果到 CSV 文件
    std::ofstream file(csvFile);
    if (file.is_open()) {
        // 写入前几行的统计信息
        file << "lowMin,lowMax,highMin,highMax,lowBinCount,highBinCount\n";
        file << lowMin << "," << lowMax << "," << highMin << "," << highMax << "," << lowBinCount << "," << highBinCount << "\n";

        // 写入 low bin -> high bin 的映射
        file << "Low Bin,Expected High Bin\n";
        for (const auto& bin : nonZeroBins) {
            file << bin.first << "," << bin.second << "\n";
        }

        file.close();
        std::cout << "CSV file saved as " << csvFile << std::endl;
    } else {
        std::cerr << "Unable to open file: " << csvFile << std::endl;
    }

  
    

}

// 导出 countMatrix 到 CSV 文件
void exportCountMatrixToCSV(const std::vector<std::vector<int>>& countMatrix, const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        // 遍历 countMatrix，并将其写入 CSV 文件
        for (const auto& row : countMatrix) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) {
                    file << ",";  // 添加逗号分隔
                }
            }
            file << "\n";  // 每一行之后换行
        }
        file.close();
        std::cout << "Count matrix exported to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

//保留次正规数的处理，因为原始数据中有次正规数
std::vector<ScidxInterval<double>> readIntervalsFromCSV(const std::string& filename) {
    std::ifstream infile(filename);  // 打开 CSV 文件进行读取
    std::vector<ScidxInterval<double>> intervals;  // 用于存储从文件中读取的区间数据
    std::string line;  // 用于存储读取到的每一行

    if (infile.is_open()) {
        while (std::getline(infile, line)) {  // 逐行读取文件内容
            std::stringstream ss(line);  // 将每行数据转为字符串流
            std::string lowStr, highStr;

            // 使用逗号分隔符提取 low 和 high 值
            if (std::getline(ss, lowStr, ',') && std::getline(ss, highStr, ',')) {
                try {
                    // 使用 stringstream 将字符串转换为 double 类型，避免 std::stod 的异常
                    double low, high;
                    std::stringstream lowStream(lowStr);
                    std::stringstream highStream(highStr);

                    if (!(lowStream >> low)) {
                        throw std::out_of_range("Unable to convert low value.");
                    }
                    if (!(highStream >> high)) {
                        throw std::out_of_range("Unable to convert high value.");
                    }

                    // 检查是否是次正规数
                    if (std::abs(low) < std::numeric_limits<double>::denorm_min() && low != 0.0) {
                        std::cout << "Low value is denormalized: " << lowStr << std::endl;
                    }
                    if (std::abs(high) < std::numeric_limits<double>::denorm_min() && high != 0.0) {
                        std::cout << "High value is denormalized: " << highStr << std::endl;
                    }

                    // 将解析出的区间数据存入 intervals 向量中
                    intervals.push_back({low, high});
                } catch (const std::invalid_argument& e) {
                    // 捕获转换错误，并输出无效数据
                    std::cerr << "Invalid element in file " << filename << ": " << e.what()
                              << " for values lowStr = " << lowStr << ", highStr = " << highStr << std::endl;
                } catch (const std::out_of_range& e) {
                    // 捕获超出范围的异常，并输出无效数据
                    std::cerr << "Out of range error in file " << filename << ": " << e.what()
                              << " for values lowStr = " << lowStr << ", highStr = " << highStr << std::endl;
                }
            }
        }
        infile.close();  // 关闭文件
    } else {
        std::cerr << "Cannot open file: " << filename << std::endl;
    }

    return intervals;  // 返回读取的区间向量
}

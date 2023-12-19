
#include <vector>
#include <queue>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>

#include <scidx.h>

int main() {
	
    std::vector<ScidxInterval<float>> intervals;
    std::ifstream file("minmax_values.txt");
    std::string line;

    while (std::getline(file, line)) {
        
        if (line.find("Min:") != std::string::npos && line.find("Max:") != std::string::npos) {
            std::istringstream iss(line);
            std::string minLabel, minVal, maxLabel, maxVal;
            iss >> minLabel >> minVal >> maxLabel >> maxVal;


            if (minLabel == "Min:" && maxLabel == "Max:") {
                float min = std::stof(minVal);
                float max = std::stof(maxVal);
                //std::cout << "min: " << min << std::endl;

                ScidxInterval<float> interval;
                interval.low = min;
                interval.high = max;

                intervals.push_back(interval);
            }
        }
    }

    ScidxRedBlackIntervalTree<float> rbIntervalTree;

    for (size_t i = 0; i < intervals.size(); i++)
    {
        //std::cout << intervals[i].low << " " << intervals[i].high << std::endl;
        rbIntervalTree.insert(intervals[i]);
        if (i == 50)
        {
            break;
        }
        
        
    }

    std::cout << "Red-Black Interval Tree after insertions:" << std::endl;
    rbIntervalTree.display();

    int levelsToTraverse = 3;

    std::vector<std::vector<std::vector<ScidxNode<float>*>>> allSubTrees;
    std::vector<std::vector<ScidxNode<float>*>> firstSubTreeNodesInLevels;

    levelOrderTraversal(rbIntervalTree.getRoot(), levelsToTraverse, firstSubTreeNodesInLevels);
    allSubTrees.push_back(firstSubTreeNodesInLevels);

    std::vector<ScidxNode<float>*>& lastLevelOfFirstSubTree = firstSubTreeNodesInLevels.back();

    std::vector<ScidxNode<float>*> rootsOfNewSubTrees;

    rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfFirstSubTree.begin(), lastLevelOfFirstSubTree.end());

    while (!rootsOfNewSubTrees.empty())
    {
        
        ScidxNode<float>* node = rootsOfNewSubTrees.front();
        rootsOfNewSubTrees.erase(rootsOfNewSubTrees.begin());  

        if (node->left == nullptr && node->right == nullptr)
        {
            continue;
        }
        

        std::vector<std::vector<ScidxNode<float>*>> currentSubTreeNodesInLevels;
        levelOrderTraversal(node, levelsToTraverse, currentSubTreeNodesInLevels);   
        allSubTrees.push_back(currentSubTreeNodesInLevels);
        std::vector<ScidxNode<float>*>& lastLevelOfCurrentSubTree = currentSubTreeNodesInLevels.back();

        rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfCurrentSubTree.begin(), lastLevelOfCurrentSubTree.end());

    }

    for (size_t i = 0; i < allSubTrees.size(); i++)
    {
        std::cout << "Subtree #" << i << ":" << std::endl;
        for (size_t j = 0; j < allSubTrees[i].size(); j++)
        {
            std::cout << "    level #" << j << ":" << std::endl;
            for (size_t k = 0; k < allSubTrees[i][j].size(); k++)
            {
                std::cout << "        " << "[" << allSubTrees[i][j][k]->interval.low << ", " << allSubTrees[i][j][k]->interval.high << "]" << std::endl;
            }
            
        }
        
    }
    
    
    return 0;
}

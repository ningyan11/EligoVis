#ifndef _SCIDX_RB_INTERVAL_TREE_H
#define _SCIDX_RB_INTERVAL_TREE_H

#include <scidx_defines.h>
#include <map>
#include <utility>

typedef enum ScidxRBColor 
{ RED, BLACK } ScidxRBColor;

template <typename T>
struct ScidxRBNode {
    ScidxInterval<T> interval;
    size_t id;
    T max_high; // Maximum high value among current node's interval and its descendants
    ScidxRBColor color;
    ScidxRBNode* parent;
    ScidxRBNode* left;
    ScidxRBNode* right;

    // Constructor with default color as RED
    ScidxRBNode(const ScidxInterval<T>& _interval, size_t _id, ScidxRBColor _color = RED, ScidxRBNode* _parent = nullptr, ScidxRBNode* _left = nullptr, ScidxRBNode* _right = nullptr)
        : interval(_interval), id(_id), max_high(_interval.high), color(_color), parent(_parent), left(_left), right(_right) {}
};

template <typename T>
class ScidxRedBlackIntervalTree {
private:
    ScidxRBNode<T>* root;

    void rotateLeft(ScidxRBNode<T>*&);
    void rotateRight(ScidxRBNode<T>*&);
    void fixInsertion(ScidxRBNode<T>*&);
    void updateMaxHigh(ScidxRBNode<T>*);

public:
    ScidxRedBlackIntervalTree() : root(nullptr) {}
    ScidxRBNode<T>* getRoot() const;
    void setRoot(ScidxRBNode<T>* newRoot);  
    void insert(const ScidxInterval<T>&, const size_t&);
    void display();
    std::vector<ScidxRBNode<T>*> query(const ScidxInterval<T>&);
    std::pair<std::vector<ScidxRBNode<T>*>, std::map<int, std::vector<std::pair<ScidxRBNode<T>*, unsigned long long>>>> queryWithMap(const ScidxInterval<T>& queryInterval, int times, int levelsToTraverse, float error_bound);
};



template <typename T>
void ScidxRedBlackIntervalTree<T>::display() {
    displayHelper(root, 0);
    std::cout << std::endl;
}

template <typename T>
void ScidxRedBlackIntervalTree<T>::updateMaxHigh(ScidxRBNode<T>* node) {
    if (node != nullptr) {
        node->max_high = std::max(node->interval.high, std::max(node->left ? node->left->max_high : node->interval.high,
                                                               node->right ? node->right->max_high : node->interval.high));
    }
}

// Method to query intervals that intersect with the given interval
template <typename T>
std::vector<ScidxRBNode<T>*> ScidxRedBlackIntervalTree<T>::query(const ScidxInterval<T>& queryInterval) {
    std::vector<ScidxRBNode<T>*> result;
    queryHelper(root, queryInterval, result);
    return result;
}

template <typename T>
std::pair<std::vector<ScidxRBNode<T>*>, std::map<int, std::vector<std::pair<ScidxRBNode<T>*, unsigned long long>>>> ScidxRedBlackIntervalTree<T>::queryWithMap(const ScidxInterval<T>& queryInterval, int times, int levelsToTraverse, float error_bound) {
    std::vector<ScidxRBNode<T>*> result;
    std::map<int, std::vector<std::pair<ScidxRBNode<T>*, unsigned long long>>> depthNodesMap;
    queryHelperWithMap(root, queryInterval, result, 0, 1, &depthNodesMap, times, levelsToTraverse, error_bound);
    return {result, depthNodesMap};
}

template <typename T>
ScidxRBNode<T>* ScidxRedBlackIntervalTree<T>::getRoot() const {
    return root;
}

template <typename T>
void ScidxRedBlackIntervalTree<T>::setRoot(ScidxRBNode<T>* newRoot) {
    root = newRoot;
}


template <typename T>
void ScidxRedBlackIntervalTree<T>::rotateLeft(ScidxRBNode<T>*& node) {
    ScidxRBNode<T>* rightChild = node->right;
    node->right = rightChild->left;

    if (rightChild->left != nullptr) {
        rightChild->left->parent = node;
    }

    rightChild->parent = node->parent;

    if (node->parent == nullptr) {
        root = rightChild;
    } else if (node == node->parent->left) {
        node->parent->left = rightChild;
    } else {
        node->parent->right = rightChild;
    }

    rightChild->left = node;
    node->parent = rightChild;

    // Update max_high for the rotated nodes
    updateMaxHigh(node);
    updateMaxHigh(rightChild);
}

template <typename T>
void ScidxRedBlackIntervalTree<T>::rotateRight(ScidxRBNode<T>*& node) {
    ScidxRBNode<T>* leftChild = node->left;
    node->left = leftChild->right;

    if (leftChild->right != nullptr) {
        leftChild->right->parent = node;
    }

    leftChild->parent = node->parent;

    if (node->parent == nullptr) {
        root = leftChild;
    } else if (node == node->parent->left) {
        node->parent->left = leftChild;
    } else {
        node->parent->right = leftChild;
    }

    leftChild->right = node;
    node->parent = leftChild;

    // Update max_high for the rotated nodes
    updateMaxHigh(node);
    updateMaxHigh(leftChild);
}


template <typename T>
void ScidxRedBlackIntervalTree<T>::fixInsertion(ScidxRBNode<T>*& node) {
    while (node != nullptr && node != root && node->parent != nullptr && node->parent->color == RED) {
        ScidxRBNode<T>* parent = node->parent;
        ScidxRBNode<T>* grandparent = parent->parent;

        if (parent == grandparent->left) {
            ScidxRBNode<T>* uncle = grandparent->right;

            if (uncle != nullptr && uncle->color == RED) {
                parent->color = BLACK;
                uncle->color = BLACK;
                grandparent->color = RED;
                node = grandparent;
            } else {
                if (node == parent->right) {
                    rotateLeft(parent);
                    std::swap(node, parent);
                }

                rotateRight(grandparent);
                std::swap(parent->color, grandparent->color);
                node = parent;
            }
        } else {
            ScidxRBNode<T>* uncle = grandparent->left;

            if (uncle != nullptr && uncle->color == RED) {
                parent->color = BLACK;
                uncle->color = BLACK;
                grandparent->color = RED;
                node = grandparent;
            } else {
                if (node == parent->left) {
                    rotateRight(parent);
                    std::swap(node, parent);
                }

                rotateLeft(grandparent);
                std::swap(parent->color, grandparent->color);
                node = parent;
            }
        }
    }

    root->color = BLACK;
}

template <typename T>
void ScidxRedBlackIntervalTree<T>::insert(const ScidxInterval<T>& interval, const size_t& id) {
    ScidxRBNode<T>* newNode = new ScidxRBNode<T>(interval, id);
    ScidxRBNode<T>* parent = nullptr;
    ScidxRBNode<T>* current = root;

    while (current != nullptr) {
        parent = current;
        if (newNode->interval.low < current->interval.low) {
            current = current->left;
        } else {
            current = current->right;
        }
    }

    newNode->parent = parent;

    if (parent == nullptr) {
        root = newNode;
    } else if (newNode->interval.low < parent->interval.low) {
        parent->left = newNode;
    } else {
        parent->right = newNode;
    }

    fixInsertion(newNode);

    // Update max_high for all ancestors
    while (newNode != nullptr) {
        updateMaxHigh(newNode);
        newNode = newNode->parent;
    }
}

template <typename T>
void displayHelper(ScidxRBNode<T>* root, int space) {
    if (root == nullptr) {
        return;
    }

    space += 5;

    displayHelper(root->right, space);

    std::cout << std::endl;
    for (int i = 5; i < space; i++) {
        std::cout << " ";
    }
    std::cout << "[" << root->interval.low << ", " << root->interval.high << "]"
              << " (max_high: " << root->max_high << ")"
              << "(" << (root->color == RED ? "RED" : "BLACK") << ")";
    displayHelper(root->left, space);
}

// Recursive helper function for querying
template <typename T>
void queryHelper(ScidxRBNode<T>* currentNode, const ScidxInterval<T>& queryInterval, std::vector<ScidxRBNode<T>*>& result) {
    if (currentNode == nullptr) {
        return;
    }

    // If the interval intersects with the query interval, add it to the result
    if (doIntervalsIntersect(currentNode->interval, queryInterval)) {
        result.push_back(currentNode);
    }

    // If the left child's max high value is greater than or equal to the query interval's low value,
    // then there may be intersecting intervals in the left subtree
    if (currentNode->left != nullptr && currentNode->left->max_high >= queryInterval.low) {
        queryHelper(currentNode->left, queryInterval, result);
    }

    // If the right child exists and its low value is less than or equal to the query interval's high value,
    // then there may be intersecting intervals in the right subtree
    if (currentNode->right != nullptr && currentNode->right->interval.low <= queryInterval.high) {
        queryHelper(currentNode->right, queryInterval, result);
    }
}






template <typename T>
void queryHelperWithMap(
    ScidxRBNode<T>* currentNode,
    const ScidxInterval<T>& queryInterval,
    std::vector<ScidxRBNode<T>*>& result,
    int currentDepth, // 新增参数：当前深度，默认为0
    unsigned long long currentIndex, // 新增参数：当前节点的索引，默认为1（根节点）
    std::map<int, std::vector<std::pair<ScidxRBNode<T>*, unsigned long long>>>* depthNodesMap, // 新增参数：用于记录每个深度节点的索引
    int times,
    int levelsToTraverse,
    float error_bound
) {
    if (currentNode == nullptr) {
        return;
    }


    // If the interval intersects with the query interval, add it to the result
    if (doIntervalsIntersectWithError(currentNode->interval, queryInterval, error_bound)) {
        result.push_back(currentNode);
        if (currentDepth == times * (levelsToTraverse-1 ) ) {
            (*depthNodesMap)[currentDepth].emplace_back(currentNode, currentIndex); // 在这里记录
        }
    }

    // 继续递归探索子树，同时更新深度和索引
    if (currentNode->left != nullptr && currentNode->left->max_high >= queryInterval.low) {
        queryHelperWithMap(currentNode->left, queryInterval, result, currentDepth + 1, currentIndex * 2, depthNodesMap, times, levelsToTraverse, error_bound);
    }

    if (currentNode->right != nullptr && currentNode->right->interval.low <= queryInterval.high) {
        queryHelperWithMap(currentNode->right, queryInterval, result, currentDepth + 1, currentIndex * 2 + 1, depthNodesMap, times, levelsToTraverse, error_bound);
    }
}



template <typename T>
bool doIntervalsIntersectWithError(const ScidxInterval<T>& interval1, const ScidxInterval<T>& interval2, float error_bound) {
    return ((interval1.low - error_bound) <= interval2.high && (interval1.high+ error_bound) >= interval2.low);
}

template <typename T>
void levelOrderTraversal(ScidxRBNode<T>* root, int levelsToTraverse, std::vector<std::vector<ScidxRBNode<T>*>>& result, std::vector<int>& treeStructure) {
    if (root == nullptr || levelsToTraverse <= 0) {
        return;
    }

    std::queue<ScidxRBNode<T>*> nodeQueue;
    nodeQueue.push(root);

    int currentLevel = 0;
    treeStructure.push_back(1); // Node exists
    while (!nodeQueue.empty() && currentLevel < levelsToTraverse) {
        int levelSize = nodeQueue.size();
        std::vector<ScidxRBNode<T>*> currentLevelNodes;

        for (int i = 0; i < levelSize; ++i) {
            ScidxRBNode<T>* currentNode = nodeQueue.front();
            nodeQueue.pop();

            if (currentNode != nullptr) {
                // Process the current node
                currentLevelNodes.push_back(currentNode);
                
                // Only enqueue actual children, do not maintain placeholders for non-existing children
                if (currentNode->left != nullptr) {
                    nodeQueue.push(currentNode->left);
                    if (currentLevel < (levelsToTraverse-1)) {
                        treeStructure.push_back(1); // Node exists
                    }
                    
                }
                else {
                    if (currentLevel < (levelsToTraverse-1)){
                        treeStructure.push_back(0); // Node does not exist
                    }
                    
                }

                if (currentNode->right != nullptr) {
                    nodeQueue.push(currentNode->right);
                   
                    if (currentLevel < (levelsToTraverse-1)) {
                        treeStructure.push_back(1); // Node exists
                    }
                }
                else {
                    if (currentLevel < (levelsToTraverse-1)) {
                        treeStructure.push_back(0); // Node does not exist
                    }
                }
            }
        }

        ++currentLevel;
        result.push_back(currentLevelNodes);
    }
}

template <typename T>
void convertTreeToArray(ScidxRBNode<T>* root, std::vector<int>& result) {
    if (root == nullptr) {
        return;
    }

    std::queue<ScidxRBNode<T>*> nodeQueue;
    nodeQueue.push(root);

    while (!nodeQueue.empty()) {
        ScidxRBNode<T>* current = nodeQueue.front();
        nodeQueue.pop();

        // 检查当前节点是否为 nullptr
        if (current != nullptr) {
            result.push_back((current->left != nullptr) ? 1 : 0);
            result.push_back((current->right != nullptr) ? 1 : 0);

            // 将非空子节点添加到队列
            if (current->left != nullptr) {
                nodeQueue.push(current->left);
            }
            if (current->right != nullptr) {
                nodeQueue.push(current->right);
            }
        } else {
            // 当前节点为 nullptr 时插入两个空节点
            result.push_back(0);
            result.push_back(0);
        }
    }
}


template <typename T>
std::vector<int> getAllSubTreesMap(ScidxRedBlackIntervalTree<T>& tree, int levelsToTraverse) {
    std::vector<int> result;
    if (!tree.getRoot()) return result; 

    std::queue<ScidxRBNode<T>*> queue;
    queue.push(tree.getRoot());

    int currentLevel = 1; 

    while (!queue.empty()) {
        int levelSize = queue.size(); 
        std::vector<int> tempStates; 

        while (levelSize > 0) {
            ScidxRBNode<T>* node = queue.front();
            queue.pop();

            // level to cut tree
             bool hasChild = (node && (node->left || node->right));
            if ((currentLevel>1) && ((currentLevel - 1) % (levelsToTraverse-1 )== 0)) {
                tempStates.push_back(hasChild ? 1 : 0);     
            }

          
            if (node) {
                queue.push(node->left ? node->left : nullptr);
                queue.push(node->right ? node->right : nullptr);
            } else {
                
                queue.push(nullptr);
                queue.push(nullptr);
            }

            levelSize--;
        }

        if (!tempStates.empty() && std::all_of(tempStates.begin(), tempStates.end(), [](int i) { return i == 0; })) {
            break;
        }

    
        if ((currentLevel>1) && ((currentLevel - 1) % (levelsToTraverse-1 )== 0)) {


            result.insert(result.end(), tempStates.begin(), tempStates.end());
        }

        currentLevel++; 
    }

    return result;
}


//get the last level nodes as the tree is a balance binary tree
template <typename T>
std::vector<ScidxRBNode<T>*> getLastLevelNodesIncludingNull(ScidxRBNode<T>* root) {
    if (!root) return {}; // If the tree is empty, return an empty vector

    std::vector<ScidxRBNode<T>*> lastLevelNodes, tempLevelNodes;
    std::queue<ScidxRBNode<T>*> queue;
    queue.push(root);

    while (!queue.empty()) {
        size_t levelSize = queue.size();
        bool allNulls = true; // Flag to check if all nodes in this level are null

        tempLevelNodes.clear(); // Clear tempLevelNodes for the new level

        for (size_t i = 0; i < levelSize; ++i) {
            ScidxRBNode<T>* currentNode = queue.front();
            queue.pop();

            tempLevelNodes.push_back(currentNode); // Save the current node (including null nodes)

             if (currentNode) {
                allNulls = false; // There's at least one non-null node in this level

                queue.push(currentNode->left); // It's safe to push nullptrs here
                queue.push(currentNode->right);
            } else {
                // If the current node is null, push nullptrs for its children (maintains structure)
                queue.push(nullptr);
                queue.push(nullptr);
            }
        }

        if (!allNulls) { // If not all nodes in this level are null, update lastLevelNodes
            lastLevelNodes = tempLevelNodes;
        } else {
            // If all nodes in this level are null, exit the loop as we've found the last non-null level
            break;
        }
    }

    return lastLevelNodes; // Return the nodes of the last non-null level
}


#endif /* ----- #ifndef _SCIDX_RB_INTERVAL_TREE_H  ----- */

#ifndef _SCIDX_AVL_INTERVAL_TREE_H
#define _SCIDX_AVL_INTERVAL_TREE_H

#include <scidx_defines.h>
#include <limits> 

template <typename T>
struct ScidxAVLNode {
    ScidxInterval<T> interval;
    size_t id;
    T max_high; // Maximum high value among current node's interval and its descendants
    ScidxAVLNode* left;
    ScidxAVLNode* right;
    int height;

    // Constructor
    ScidxAVLNode(const ScidxInterval<T>& _interval, size_t _id, ScidxAVLNode* _left = nullptr, ScidxAVLNode* _right = nullptr)
        : interval(_interval), id(_id), max_high(std::numeric_limits<T>::min()), left(_left), right(_right), height(1) {}
};

template<typename T>
class ScidxAVLIntervalTree {
private:
    ScidxAVLNode<T>* root;

    int height(ScidxAVLNode<T>* node);

    int max(int a, int b) {
        return (a > b) ? a : b;
    }

    ScidxAVLNode<T>* insert(ScidxAVLNode<T>* node, size_t id, ScidxInterval<T> interval);

    ScidxAVLNode<T>* rotateLeft(ScidxAVLNode<T>* x);

    ScidxAVLNode<T>* rotateRight(ScidxAVLNode<T>* y);

    int getBalance(ScidxAVLNode<T>* node);

    int getMaxHigh(ScidxAVLNode<T>* node);

public:
    ScidxAVLIntervalTree() : root(nullptr) {}
    
    void insertNode(size_t, ScidxInterval<T>);

    void display();

    std::vector<ScidxAVLNode<T>*> query(const ScidxInterval<T>&);

    ScidxAVLNode<T>* getRoot() const;

    void setRoot(ScidxAVLNode<T>* newRoot);  

};

template <typename T>
ScidxAVLNode<T>* ScidxAVLIntervalTree<T>::getRoot() const {
    return root;
} 

template <typename T>
void ScidxAVLIntervalTree<T>::setRoot(ScidxAVLNode<T>* newRoot) {
    root = newRoot;
}

template<typename T>
int ScidxAVLIntervalTree<T>::height(ScidxAVLNode<T>* node) {
    if (node == nullptr)
        return 0;
    return node->height;
}

template<typename T>
ScidxAVLNode<T>* ScidxAVLIntervalTree<T>::insert(ScidxAVLNode<T>* node, size_t id, ScidxInterval<T> interval) {
    if (node == nullptr) {
        return new ScidxAVLNode<T>(interval, id);
    }
    //std::cout << "at line " << __LINE__ << std::endl;
    if (interval.low < node->interval.low)
    {
        //std::cout << "at line " << __LINE__ << std::endl;
        node->left = insert(node->left, id, interval);
    }
    else
    {
        //std::cout << "at line " << __LINE__ << std::endl;
        node->right = insert(node->right, id, interval);
    }

    node->height = max(height(node->left), height(node->right)) + 1;

    node->max_high = max(node->interval.high, max(getMaxHigh(node->left), getMaxHigh(node->right)));

    int balance = getBalance(node);
    //std::cout << "at line " << __LINE__ << ": balance="<< balance << std::endl;

    if (balance > 1 && interval.low < node->left->interval.low) {
        //std::cout << "at line " << __LINE__ << std::endl;
        return rotateRight(node);
    }
    
    if (balance < -1 && interval.low > node->right->interval.low) {
        //std::cout << "at line " << __LINE__ << std::endl;
        return rotateLeft(node);
    }

    if (balance > 1 && interval.low > node->left->interval.low) {
        //std::cout << "at line " << __LINE__ << std::endl;
        node->left = rotateLeft(node->left);
        //std::cout << "at line " << __LINE__ << std::endl;
        return rotateRight(node);
    }

    if (balance < -1 && interval.low < node->right->interval.low) {
        //std::cout << "at line " << __LINE__ << std::endl;
        // if (node->right == nullptr)
        // {
        //     std::cout << "node->right is nullptr" << std::endl;
        // }
        // if (node->left == nullptr)
        // {
        //     std::cout << "node->left is nullptr" << std::endl;
        // }
        // if (node->right->right == nullptr)
        // {
        //     std::cout << "node->right->right is nullptr" << std::endl;
        // }
        // if (node->right->left == nullptr)
        // {
        //     std::cout << "node->right->left is nullptr" << std::endl;
        //     //std::cout << "node->left->height: " << node->left->height << ", node->right->height: " << node->right->height << std::endl;
        // }
        
        //std::cout << "  node left: " << node->left->id << ", " << "[" << node->left->interval.low << " " << node->left->interval.high << "]" << std::endl;
        //std::cout << "  node: " << node->id << ", " << "[" << node->interval.low << " " << node->interval.high << "]" << std::endl;
        //std::cout << "  node right: " << node->right->id << ", " << "[" << node->right->interval.low << " " << node->right->interval.high << "]" << std::endl;
        node->right = rotateRight(node->right);
        //std::cout << "at line " << __LINE__ << std::endl;
        //std::cout << "  node left: " << node->left->id << ", " << "[" << node->left->interval.low << " " << node->left->interval.high << "]" << std::endl;
        //std::cout << "  node: " << node->id << ", " << "[" << node->interval.low << " " << node->interval.high << "]" << std::endl;
        //std::cout << "  node right: " << node->right->id << ", " << "[" << node->right->interval.low << " " << node->right->interval.high << "]" << std::endl;
        return rotateLeft(node);
        //std::cout << "at line " << __LINE__ << std::endl;
        //std::cout << "  node left: " << node->left->id << ", " << "[" << node->left->interval.low << " " << node->left->interval.high << "]" << std::endl;
        //std::cout << "  node: " << node->id << ", " << "[" << node->interval.low << " " << node->interval.high << "]" << std::endl;
        //std::cout << "  node right: " << node->right->id << ", " << "[" << node->right->interval.low << " " << node->right->interval.high << "]" << std::endl;
    }

    //node->max_high = max(node->interval.high, max((node->left ? node->left->max_high : std::numeric_limits<T>::min()), (node->right ? node->right->max_high : std::numeric_limits<T>::min())));
    return node;
}

template<typename T>
ScidxAVLNode<T>* ScidxAVLIntervalTree<T>::rotateLeft(ScidxAVLNode<T>* x) {
    //if (x == nullptr || x->right == nullptr)
    //    return x;

    ScidxAVLNode<T>* y = x->right;
    ScidxAVLNode<T>* T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = max(height(x->left), height(x->right)) + 1;
    y->height = max(height(y->left), height(y->right)) + 1;

    x->max_high = getMaxHigh(x);
    y->max_high = getMaxHigh(y);

    return y;
}

template<typename T>
ScidxAVLNode<T>* ScidxAVLIntervalTree<T>::rotateRight(ScidxAVLNode<T>* y) {
    //if (y == nullptr || y->left == nullptr)
    //    return y;

    ScidxAVLNode<T>* x = y->left;
    ScidxAVLNode<T>* T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = max(height(y->left), height(y->right)) + 1;
    x->height = max(height(x->left), height(x->right)) + 1;

    y->max_high = getMaxHigh(y);
    x->max_high = getMaxHigh(x);

    return x;
}

template<typename T>
int ScidxAVLIntervalTree<T>::getBalance(ScidxAVLNode<T>* node) {
    if (node == nullptr)
        return 0;
    return height(node->left) - height(node->right);
}

template<typename T>
int ScidxAVLIntervalTree<T>::getMaxHigh(ScidxAVLNode<T>* node) {
    if (node == nullptr)
        return 0;
    return max(node->max_high, max(getMaxHigh(node->left), getMaxHigh(node->right)));
}

template<typename T>
 void ScidxAVLIntervalTree<T>::insertNode(size_t id, ScidxInterval<T> interval) {
    root = insert(root, id, interval);
 }

 template <typename T>
void ScidxAVLIntervalTree<T>::display() {
    displayHelper(root, 0);
    std::cout << std::endl;
}

template <typename T>
void displayHelper(ScidxAVLNode<T>* root, int space) {
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
              << " (max_high: " << root->max_high << ")";
    displayHelper(root->left, space);
}

// Method to query intervals that intersect with the given interval
template <typename T>
std::vector<ScidxAVLNode<T>*> ScidxAVLIntervalTree<T>::query(const ScidxInterval<T>& queryInterval) {
    std::vector<ScidxAVLNode<T>*> result;
    queryHelper(root, queryInterval, result);
    return result;
}

// Recursive helper function for querying
template <typename T>
void queryHelper(ScidxAVLNode<T>* currentNode, const ScidxInterval<T>& queryInterval, std::vector<ScidxAVLNode<T>*>& result) {
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
int DFSrecur(ScidxAVLNode<T>* node) {
    if (!node) return 0;
    return std::max(DFSrecur(node->left)+1, DFSrecur(node->right)+1);
}

//get the structure map of the whole tree(only the level to cut)
template <typename T>
std::vector<bool> getAllSubTreesMap(ScidxAVLIntervalTree<T>& tree, int levelsToTraverse) {
    int maxLevel = DFSrecur(tree.getRoot());
    std::cout << "max level: " << maxLevel << std::endl;
    std::vector<bool> result;
    
    if (!tree.getRoot()) return result; 

    std::queue<ScidxAVLNode<T>*> queue;
    queue.push(tree.getRoot());

    int currentLevel = 1; 

    while (!queue.empty()) {
        int levelSize = queue.size(); 
        //std::cout << "Finished " << currentLevel << " currentLevel and handling " << levelSize << " level size" << std::endl;
        std::vector<bool> tempStates; 

        while (levelSize > 0) {
            ScidxAVLNode<T>* node = queue.front();
            queue.pop();

            // Check if the node has children
            bool hasChild = (node && (node->left || node->right));
            if ((currentLevel > 1) && ((currentLevel - 1) % (levelsToTraverse - 1) == 0)) {
                tempStates.push_back(hasChild ? true : false);     
            }

            // Add children to the queue
            if (node) {
                queue.push(node->left ? node->left : nullptr);
                queue.push(node->right ? node->right : nullptr);
            } else {
                queue.push(nullptr);
                queue.push(nullptr);
            }

            levelSize--;
        }

        // Check if the current level is entirely empty subtrees
        if (!tempStates.empty() && std::all_of(tempStates.begin(), tempStates.end(), [](bool b) { return b == false; })) {
            break;
        }

        // Append the current level subtree states to the result
        if ((currentLevel > 1) && ((currentLevel - 1) % (levelsToTraverse - 1) == 0)) {
            result.insert(result.end(), tempStates.begin(), tempStates.end());
        }

        currentLevel++; 
    }

    return result;
}


// Level-order traversal function for ScidxAVLIntervalTree
template <typename T>
void levelOrderTraversal(ScidxAVLNode<T>* root, int levelsToTraverse, std::vector<std::vector<ScidxAVLNode<T>*>>& result, std::vector<int>& treeStructure) {
    if (root == nullptr || levelsToTraverse <= 0) {
        return;
    }

    std::queue<ScidxAVLNode<T>*> nodeQueue;
    nodeQueue.push(root);

    int currentLevel = 0;
    treeStructure.push_back(1); // Node exists
    while (!nodeQueue.empty() && currentLevel < levelsToTraverse) {
        int levelSize = nodeQueue.size();
        std::vector<ScidxAVLNode<T>*> currentLevelNodes;

        for (int i = 0; i < levelSize; ++i) {
            ScidxAVLNode<T>* currentNode = nodeQueue.front();
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
                } else {
                    if (currentLevel < (levelsToTraverse-1)) {
                        treeStructure.push_back(0); // Node does not exist
                    }
                }

                if (currentNode->right != nullptr) {
                    nodeQueue.push(currentNode->right);
                    if (currentLevel < (levelsToTraverse-1)) {
                        treeStructure.push_back(1); // Node exists
                    }
                } else {
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
std::vector<ScidxAVLNode<T>*> getLastLevelNodesIncludingNull(ScidxAVLNode<T>* root) {
    if (!root) return {}; // 如果树为空，则返回空向量

    std::vector<ScidxAVLNode<T>*> lastLevelNodes, tempLevelNodes;
    std::queue<ScidxAVLNode<T>*> queue;
    queue.push(root);

    while (!queue.empty()) {
        size_t levelSize = queue.size();
        bool allNulls = true; // 检查这一层是否全是空节点的标志

        tempLevelNodes.clear(); // 为新的一层清空tempLevelNodes

        for (size_t i = 0; i < levelSize; ++i) {
            ScidxAVLNode<T>* currentNode = queue.front();
            queue.pop();

            tempLevelNodes.push_back(currentNode); // 保存当前节点（包括空节点）

            if (currentNode) {
                allNulls = false; // 这一层至少有一个非空节点

                queue.push(currentNode->left); // 这里可以安全地推送nullptr
                queue.push(currentNode->right);
            } else {
                // 如果当前节点为空，推送它的子节点（为空，保持结构）
                queue.push(nullptr);
                queue.push(nullptr);
            }
        }

        if (!allNulls) { // 如果这一层不是全为空节点，更新lastLevelNodes
            lastLevelNodes = tempLevelNodes;
        } else {
            // 如果这一层全是空节点，退出循环，因为我们找到了最后一个非空节点层
            break;
        }
    }

    return lastLevelNodes; // 返回最后一个非空层的节点
}






#endif /* ----- #ifndef _SCIDX_AVL_INTERVAL_TREE_H  ----- */
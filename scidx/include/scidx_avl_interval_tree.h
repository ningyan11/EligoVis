#ifndef _SCIDX_AVL_INTERVAL_TREE_H
#define _SCIDX_AVL_INTERVAL_TREE_H

#include <scidx_defines.h>
#include <limits> 
#include <algorithm> 
#include <stack>


template <typename T>
struct ScidxAVLNode {
    ScidxInterval<T> interval;
    size_t id;
    T max_high; // Maximum high value among current node's interval and its descendants
    ScidxAVLNode* left;
    ScidxAVLNode* right;
    int height;

// **修改点1：构造函数，max_high 初始化为 interval.high，而不是 std::numeric_limits<T>::min()**
    ScidxAVLNode(const ScidxInterval<T>& _interval, size_t _id, ScidxAVLNode* _left = nullptr, ScidxAVLNode* _right = nullptr)
        : interval(_interval), id(_id), max_high(_interval.high), left(_left), right(_right), height(1) {}
};

template<typename T>
struct SkippedNode {
    ScidxInterval<T> interval;
    size_t id;

    SkippedNode(const ScidxInterval<T>& _interval, size_t _id)
        : interval(_interval), id(_id) {}
};


template<typename T>
class ScidxAVLIntervalTree {
private:
    ScidxAVLNode<T>* root;
    int height(ScidxAVLNode<T>* node) const;

    ScidxAVLNode<T>* insert(ScidxAVLNode<T>* node, size_t id, ScidxInterval<T> interval, std::vector<SkippedNode<T> >& skippedIntervals);

    ScidxAVLNode<T>* rotateLeft(ScidxAVLNode<T>* x);

    ScidxAVLNode<T>* rotateRight(ScidxAVLNode<T>* y);

    int getBalance(ScidxAVLNode<T>* node);
    
    T getMaxHigh(ScidxAVLNode<T>* node);

public:
    ScidxAVLIntervalTree() : root(nullptr) {}
    
    void insertNode(size_t, ScidxInterval<T>, std::vector<SkippedNode<T>>& );

    int getTreeHeight() const; 

    void display();

    std::vector<ScidxAVLNode<T>*> query(const ScidxInterval<T>&);

    ScidxAVLNode<T>* getRoot() const;

    void setRoot(ScidxAVLNode<T>* newRoot);  

};

template <typename T>
ScidxAVLNode<T>* ScidxAVLIntervalTree<T>::getRoot() const {
    return root;
} 

template<typename T>
int ScidxAVLIntervalTree<T>::getTreeHeight() const {
    return height(root);  // 根节点的高度即为树的高度
}

template <typename T>
void ScidxAVLIntervalTree<T>::setRoot(ScidxAVLNode<T>* newRoot) {
    root = newRoot;
}

template<typename T>
int ScidxAVLIntervalTree<T>::height(ScidxAVLNode<T>* node) const {  // 添加 const 限定符
    if (node == nullptr)
        return 0;
    return node->height;
}


template<typename T>
ScidxAVLNode<T>* ScidxAVLIntervalTree<T>::insert(ScidxAVLNode<T>* node, size_t id, ScidxInterval<T> interval, std::vector<SkippedNode<T>>& skippedIntervals) {
    if (node == nullptr) {
        return new ScidxAVLNode<T>(interval, id);
    }
    //std::cout << "at line " << __LINE__ << std::endl;
    if (interval.low < node->interval.low)
    {
        //std::cout << "at line " << __LINE__ << std::endl;
        node->left = insert(node->left, id, interval, skippedIntervals);
    }else if (interval.low > node->interval.low) {
        node->right = insert(node->right, id, interval, skippedIntervals);
    }else {
        // 处理 interval.low 相同的情况：记录跳过的节点
        skippedIntervals.push_back(SkippedNode<T>(interval, id));  // 将 interval 和 id 一起保存
        return node;  // 直接返回当前节点，不再插入
    }

    node->height = std::max(height(node->left), height(node->right)) + 1;

    //std::cout << "Node ID: " << node->id << ", New Height: " << node->height << std::endl;

    //node->max_high = std::max(node->interval.high, std::max(getMaxHigh(node->left), getMaxHigh(node->right)));

    node->max_high = std::max(
        node->interval.high,
        std::max(
            node->left ? node->left->max_high : std::numeric_limits<T>::lowest(),
            node->right ? node->right->max_high : std::numeric_limits<T>::lowest()
        )
    );


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
        /*if (node->right == nullptr)
        {
            std::cout << "node->right is nullptr" << std::endl;
            std::cout << "Current node info: " << std::endl;
          
            std::cout << "  node ID: " << node->id << std::endl;
            std::cout << "  node Interval: [" << node->interval.low << ", " << node->interval.high << "]" << std::endl;
            std::cout << "  node Max High: " << node->max_high << std::endl;
            std::cout << "  node Height: " << node->height << std::endl;

    
        }
         if (node->left == nullptr)
         {
             std::cout << "node->left is nullptr" << std::endl;
         }
         if (node->right->right == nullptr)
         {
             std::cout << "node->right->right is nullptr" << std::endl;
        }
        if (node->right->left == nullptr)
         {
            std::cout << "node->right->left is nullptr" << std::endl;
            //std::cout << "node->left->height: " << node->left->height << ", node->right->height: " << node->right->height << std::endl;
         }
        
        std::cout << "  node left: " << node->left->id << ", " << "[" << node->left->interval.low << " " << node->left->interval.high << "]" << std::endl;
        std::cout << "  node: " << node->id << ", " << "[" << node->interval.low << " " << node->interval.high << "]" << std::endl;
        std::cout << "  node right: " << node->right->id << ", " << "[" << node->right->interval.low << " " << node->right->interval.high << "]" << std::endl;*/
        //std::cout << "at line " << __LINE__ << std::endl;
        //std::cout << "  node right: " << node->right->id << ", " << "[" << node->right->interval.low << " " << node->right->interval.high << "]" << std::endl;
       
        node->right = rotateRight(node->right);
        //std::cout << "at line " << __LINE__ << std::endl;
        /*std::cout << "  node left: " << node->left->id << ", " << "[" << node->left->interval.low << " " << node->left->interval.high << "]" << std::endl;
        //std::cout << "  node: " << node->id << ", " << "[" << node->interval.low << " " << node->interval.high << "]" << std::endl;
        //std::cout << "  node right: " << node->right->id << ", " << "[" << node->right->interval.low << " " << node->right->interval.high << "]" << std::endl;*/
        return rotateLeft(node);
        /*std::cout << "at line " << __LINE__ << std::endl;
        //std::cout << "  node left: " << node->left->id << ", " << "[" << node->left->interval.low << " " << node->left->interval.high << "]" << std::endl;
        //std::cout << "  node: " << node->id << ", " << "[" << node->interval.low << " " << node->interval.high << "]" << std::endl;
        //std::cout << "  node right: " << node->right->id << ", " << "[" << node->right->interval.low << " " << node->right->interval.high << "]" << std::endl;*/
    }

    //node->max_high = max(node->interval.high, max((node->left ? node->left->max_high : std::numeric_limits<T>::min()), (node->right ? node->right->max_high : std::numeric_limits<T>::min())));
    return node;
}

//迭代的方式
/*template<typename T>
void insertNodeIterative(ScidxAVLNode<T>*& root, size_t id, const ScidxInterval<T>& interval) {
    // Step 1: Find the appropriate position to insert the new node iteratively
    ScidxAVLNode<T>* node = root;
    ScidxAVLNode<T>* parent = nullptr;

    while (node != nullptr) {
        parent = node;
        if (interval.low < node->interval.low) {
            node = node->left;
        } else {
            node = node->right;
        }
    }

    // Step 2: Create a new node
    ScidxAVLNode<T>* newNode = new ScidxAVLNode<T>(interval, id);

    // Step 3: Insert the new node at the correct position
    if (parent == nullptr) {
        root = newNode;  // Tree was empty, set root to new node
    } else if (interval.low < parent->interval.low) {
        parent->left = newNode;
    } else {
        parent->right = newNode;
    }

    // Step 4: Rebalance the tree and update heights and max_high values
    ScidxAVLNode<T>* current = root;
    std::stack<ScidxAVLNode<T>*> path;  // Stack to keep track of the path

    // Step 5: Traverse the path back to the root to rebalance
    while (current != nullptr) {
        path.push(current);
        if (interval.low < current->interval.low) {
            current = current->left;
        } else {
            current = current->right;
        }
    }

    // Step 6: Go up the tree and rebalance where necessary
    while (!path.empty()) {
        ScidxAVLNode<T>* currentNode = path.top();
        path.pop();

        // Update the height and max_high for the current node
        currentNode->height = std::max(height(currentNode->left), height(currentNode->right)) + 1;
        currentNode->max_high = std::max(currentNode->interval.high,
                                          std::max(getMaxHigh(currentNode->left), getMaxHigh(currentNode->right)));

        // Check the balance factor and rebalance if needed
        int balance = getBalance(currentNode);

        // Left-Left (Right Rotation)
        if (balance > 1 && interval.low < currentNode->left->interval.low) {
            if (!path.empty()) {
                ScidxAVLNode<T>* parentOfCurrent = path.top();
                if (parentOfCurrent->left == currentNode) {
                    parentOfCurrent->left = rotateRight(currentNode);
                } else {
                    parentOfCurrent->right = rotateRight(currentNode);
                }
            } else {
                root = rotateRight(currentNode);
            }
        }

        // Right-Right (Left Rotation)
        else if (balance < -1 && interval.low > currentNode->right->interval.low) {
            if (!path.empty()) {
                ScidxAVLNode<T>* parentOfCurrent = path.top();
                if (parentOfCurrent->left == currentNode) {
                    parentOfCurrent->left = rotateLeft(currentNode);
                } else {
                    parentOfCurrent->right = rotateLeft(currentNode);
                }
            } else {
                root = rotateLeft(currentNode);
            }
        }

        // Left-Right (Left-Right Rotation)
        else if (balance > 1 && interval.low > currentNode->left->interval.low) {
            currentNode->left = rotateLeft(currentNode->left);
            if (!path.empty()) {
                ScidxAVLNode<T>* parentOfCurrent = path.top();
                if (parentOfCurrent->left == currentNode) {
                    parentOfCurrent->left = rotateRight(currentNode);
                } else {
                    parentOfCurrent->right = rotateRight(currentNode);
                }
            } else {
                root = rotateRight(currentNode);
            }
        }

        // Right-Left (Right-Left Rotation)
        else if (balance < -1 && interval.low < currentNode->right->interval.low) {
            currentNode->right = rotateRight(currentNode->right);
            if (!path.empty()) {
                ScidxAVLNode<T>* parentOfCurrent = path.top();
                if (parentOfCurrent->left == currentNode) {
                    parentOfCurrent->left = rotateLeft(currentNode);
                } else {
                    parentOfCurrent->right = rotateLeft(currentNode);
                }
            } else {
                root = rotateLeft(currentNode);
            }
        }
    }
}*/

// rotateLeft 函数的修改
template<typename T>
ScidxAVLNode<T>* ScidxAVLIntervalTree<T>::rotateLeft(ScidxAVLNode<T>* x) {
    ScidxAVLNode<T>* y = x->right;
    ScidxAVLNode<T>* T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = std::max(height(x->left), height(x->right)) + 1;
    y->height = std::max(height(y->left), height(y->right)) + 1;

    /*// 更新 max_high 的值，使用相同类型的 std::max 函数
    x->max_high = std::max(x->interval.high, std::max(getMaxHigh(x->left), getMaxHigh(x->right)));
    y->max_high = std::max(y->interval.high, std::max(getMaxHigh(y->left), getMaxHigh(y->right)));*/

    // 更新 max_high 值，使用封装的 getMaxHigh 方法
    /*x->max_high = getMaxHigh(x);
    y->max_high = getMaxHigh(y);*/

    x->max_high = std::max(
        x->interval.high,
        std::max(
            x->left ? x->left->max_high : std::numeric_limits<T>::lowest(),
            x->right ? x->right->max_high : std::numeric_limits<T>::lowest()
        )
    );
    y->max_high = std::max(
        y->interval.high,
        std::max(
            y->left ? y->left->max_high : std::numeric_limits<T>::lowest(),
            y->right ? y->right->max_high : std::numeric_limits<T>::lowest()
        )
    );


    return y;
}

// rotateRight 函数的修改
template<typename T>
ScidxAVLNode<T>* ScidxAVLIntervalTree<T>::rotateRight(ScidxAVLNode<T>* y) {

   
    ScidxAVLNode<T>* x = y->left;
    ScidxAVLNode<T>* T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = std::max(height(y->left), height(y->right)) + 1;
    x->height = std::max(height(x->left), height(x->right)) + 1;

    /*// 更新 max_high 的值，使用相同类型的 std::max 函数
    y->max_high = std::max(y->interval.high, std::max(getMaxHigh(y->left), getMaxHigh(y->right)));
    x->max_high = std::max(x->interval.high, std::max(getMaxHigh(x->left), getMaxHigh(x->right)));*/


    // 更新 max_high 值，使用封装的 getMaxHigh 方法
    /*y->max_high = getMaxHigh(y);
    x->max_high = getMaxHigh(x);*/


    y->max_high = std::max(
    y->interval.high,
        std::max(
            y->left ? y->left->max_high : std::numeric_limits<T>::lowest(),
            y->right ? y->right->max_high : std::numeric_limits<T>::lowest()
        )
    );
    x->max_high = std::max(
        x->interval.high,
        std::max(
            x->left ? x->left->max_high : std::numeric_limits<T>::lowest(),
            x->right ? x->right->max_high : std::numeric_limits<T>::lowest()
        )
    );


    // Logging for debugging
    /*std::cout << "Rotated right: " << std::endl;
    std::cout << "  y (old root): id=" << y->id << ", interval=[" << y->interval.low << ", " << y->interval.high << "], max_high=" << y->max_high << std::endl;
    std::cout << "  x (new root): id=" << x->id << ", interval=[" << x->interval.low << ", " << x->interval.high << "], max_high=" << x->max_high << std::endl;*/


    return x;
}


template<typename T>
int ScidxAVLIntervalTree<T>::getBalance(ScidxAVLNode<T>* node) {
    if (node == nullptr)
        return 0;
    return height(node->left) - height(node->right);
}


/*
// **修改点2：改进的 getMaxHigh 函数，确保获取正确的 max_high 值**
// 修改后的 getMaxHigh 函数
template<typename T>
T ScidxAVLIntervalTree<T>::getMaxHigh(ScidxAVLNode<T>* node) {
    if (node == nullptr)
        return 0;  // 对空节点返回 0，避免使用 std::numeric_limits<T>::min()

    std::cout << "getMaxHigh: Node ID: " << node->id << ", interval: [" << node->interval.low << ", " << node->interval.high << "]" << std::endl;

    
    // 返回当前节点与其左右子节点中的最大 high 值
    T leftMaxHigh = getMaxHigh(node->left);
    T rightMaxHigh = getMaxHigh(node->right);

    std::cout << "getMaxHigh: Node ID: " << node->id << " Left Max: " << leftMaxHigh << ", Right Max: " << rightMaxHigh << std::endl;


    // 当前节点 max_high 是它自己和左右子节点 max_high 的最大值
    return std::max(node->interval.high, std::max(leftMaxHigh, rightMaxHigh));
}*/

/*template<typename T>
T ScidxAVLIntervalTree<T>::getMaxHigh(ScidxAVLNode<T>* node) {
    if (node == nullptr)
        return 0;
    return std::max(node->max_high, std::max(getMaxHigh(node->left), getMaxHigh(node->right)));
}*/


//修改上面的方法，不用递归getMaxHigh
template<typename T>
T ScidxAVLIntervalTree<T>::getMaxHigh(ScidxAVLNode<T>* node) {
    return node ? node->max_high : std::numeric_limits<T>::lowest();
}


template<typename T>
 void ScidxAVLIntervalTree<T>::insertNode(size_t id, ScidxInterval<T> interval, std::vector<SkippedNode<T>>& skippedIntervals ) {
    /*std::cout << "Inserting node with id: " << id 
              << ", interval: [" << interval.low << ", " << interval.high << "]" << std::endl;*/

    root = insert(root, id, interval, skippedIntervals);
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
              << " (max_high: " << root->max_high << ")"<< ", " << " (id: " << root->id << ")";
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
//整个树的切分层，是否下挂subTree的map
//中间值用bool,在c++中只用一个bit,最终还记录8bit一个byte,remainingBits为不足8的bit
template <typename T>
std::pair<std::vector<uint8_t>, int> getAllSubTreesMap(ScidxAVLIntervalTree<T>& tree, int levelsToTraverse) {
    int maxLevel = DFSrecur(tree.getRoot());
    //std::cout << "max level: " << maxLevel << std::endl;
    std::vector<uint8_t> result;
    int remainingBits = 0; // Track the number of bits in the final byte that are used

    if (!tree.getRoot()) return {result, remainingBits};

    std::queue<ScidxAVLNode<T>*> queue;
    queue.push(tree.getRoot());

    int currentLevel = 1;
    std::vector<bool> allStates; // Store all boolean states here

    while (!queue.empty()) {
        int levelSize = queue.size();
        std::vector<bool> tempStates;  // Store current level states

        while (levelSize > 0) {
            ScidxAVLNode<T>* node = queue.front();
            queue.pop();

            // Check if the node has children
            bool hasChild = (node && (node->left || node->right));
            if ((currentLevel > 1) && ((currentLevel - 1) % (levelsToTraverse - 1) == 0)) {
                tempStates.push_back(hasChild);  // Collect boolean state for current level
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

        // Append the current level subtree states to allStates
        if ((currentLevel > 1) && ((currentLevel - 1) % (levelsToTraverse - 1) == 0)) {
            allStates.insert(allStates.end(), tempStates.begin(), tempStates.end());
        }

        currentLevel++;
    }

    /*std::cout << "allStates: ";
    for (bool state : allStates) {
        std::cout << state;
    }
    std::cout << std::endl;*/

    //Now process allStates to convert them into bytes
    uint8_t byte = 0;
    int bitCount = 0;
    for (bool state : allStates) {
        byte |= (state << bitCount);
        bitCount++;

        // If 8 bits have been collected, push the byte to the result
        if (bitCount == 8) {
            result.push_back(byte);
            byte = 0;
            bitCount = 0;
        }
    }

    // If there are remaining bits, push the final byte and record the number of bits used
    if (bitCount > 0) {
        result.push_back(byte);
        remainingBits = bitCount;
    }


    // Output the results for verification
    std::cout << "Remaining Bits: " << remainingBits << std::endl;



    return {result, remainingBits};
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


//a是node区间，b查询区间
template <typename T>
bool doAVLIntervalsIntersect(const ScidxInterval<T>& a, const ScidxInterval<T>& b) {
    return !(a.high < b.low || a.low > b.high);
}


template <typename T>
std::vector<size_t> queryOverlapIds(ScidxAVLNode<T>* decompressedTree, const ScidxInterval<T>& queryInterval) {
    std::vector<size_t> result;
    std::stack<ScidxAVLNode<T>*> nodeStack;
    
    if (decompressedTree) nodeStack.push(decompressedTree);

    while (!nodeStack.empty()) {
        ScidxAVLNode<T>* node = nodeStack.top();
        nodeStack.pop();

        if (!node) continue;

        // 强剪枝：如果 node->max_high < queryInterval.low，说明整棵子树都不可能相交，直接跳过
        if (node->max_high < queryInterval.low) continue;

        // 弱剪枝：如果 node->interval.low > queryInterval.high，跳过右子树，只搜索左子树
        if (node->interval.low > queryInterval.high) {
            nodeStack.push(node->left);
            continue;
        }

        // 检查当前节点是否相交
        if (doAVLIntervalsIntersect(node->interval, queryInterval)) {
            /*std::cout << "[DEBUG] Match Node ID: " << node->id 
              << ", Interval: [" << node->interval.low 
              << ", " << node->interval.high << "]" << std::endl;*/

            result.push_back(node->id);  // 记录匹配的节点 ID

        }

        // 左子树搜索（如果 max_high >= q_low）
        if (node->left && node->left->max_high >= queryInterval.low) {
            nodeStack.push(node->left);
        }

        // 右子树搜索（如果 node->interval.low <= q_high）
        if (node->right && node->interval.low <= queryInterval.high) {
            nodeStack.push(node->right);
        }
    }
    
    return result;
}



//根据errorbound调整查找区间和更新maxhigh
template<typename T>
T adjustAndUpdateMaxHigh(ScidxAVLNode<T>* node, T error_bound) {
    if (!node) return std::numeric_limits<T>::lowest();

    // 递归处理左右子树
    T leftMax = adjustAndUpdateMaxHigh(node->left, error_bound);
    T rightMax = adjustAndUpdateMaxHigh(node->right, error_bound);

    // 修正 min 和 max
    node->interval.low -= error_bound;
    node->interval.high += error_bound;

    // 更新 max_high
    node->max_high = std::max({node->interval.high, leftMax, rightMax});
    return node->max_high;
}


/*//查询overlap的id
template <typename T>
std::vector<size_t> queryOverlapIds(ScidxAVLNode<T>* node, const ScidxInterval<T>& query) {
    std::vector<size_t> result;
    std::stack<ScidxAVLNode<T>*> stack;

    if (node) stack.push(node);

    while (!stack.empty()) {
        ScidxAVLNode<T>* current = stack.top();
        stack.pop();

        if (!current) continue;

        // 1. max_high 剪枝（强剪枝）
        if (current->max_high < query.low) continue;

        // 2. interval.low > query.high 剪枝（弱剪枝）
        if (current->interval.low > query.high) {
            // interval.low 大于查询上界，右边不可能了，只搜左子树
            stack.push(current->left);
            continue;
        }

        // ⚠️ interval.high < query.low 不能直接剪枝（它的右子树仍可能匹配）

        // 3. 判断当前节点是否相交
        if (current->interval.high >= query.low && current->interval.low <= query.high) {
            result.push_back(current->id);  // 匹配上
        }

        // 4. 递归搜索（stack 模拟）
        if (current->right && current->interval.low <= query.high) {
            stack.push(current->right);  // 有可能包含符合的区间
        }

        if (current->left && current->left->max_high >= query.low) {
            stack.push(current->left);  // 可能含符合的 max
        }
    }

    return result;
}*/












#endif /* ----- #ifndef _SCIDX_AVL_INTERVAL_TREE_H  ----- */
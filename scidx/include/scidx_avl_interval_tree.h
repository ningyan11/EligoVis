#ifndef _SCIDX_AVL_INTERVAL_TREE_H
#define _SCIDX_AVL_INTERVAL_TREE_H

#include <scidx_defines.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <queue>

template <typename T>
struct ScidxavlInterval {
    T low;
    T high;
};

template <typename T>
struct ScidxavlNode {
    ScidxavlInterval<T> interval;
    size_t id;
    T max_high;
    int height;
    ScidxavlNode* parent;
    ScidxavlNode* left;
    ScidxavlNode* right;

    ScidxavlNode(const ScidxavlInterval<T>& _interval, size_t _id, ScidxavlNode* _parent = nullptr, ScidxavlNode* _left = nullptr, ScidxavlNode* _right = nullptr)
        : interval(_interval), id(_id), max_high(_interval.high), height(1), parent(_parent), left(_left), right(_right) {}
};

template <typename T>
class ScidxAVLIntervalTree {
private:
    ScidxavlNode<T>* root;

    void rotateLeft(ScidxavlNode<T>*&);
    void rotateRight(ScidxavlNode<T>*&);
    void fixInsertion(ScidxavlNode<T>*&);
    void updateNode(ScidxavlNode<T>*);

public:
    ScidxAVLIntervalTree() : root(nullptr) {}
    ScidxavlNode<T>* getRoot() const;
    void insert(const ScidxavlInterval<T>&, const size_t&);
    void display();
    std::vector<ScidxavlNode<T>*> query(const ScidxavlInterval<T>&);

};

template <typename T>
int getHeight(ScidxavlNode<T>* node) {
    if (node == nullptr) {
        return 0;
    }
    return node->height;
}

template <typename T>
void ScidxAVLIntervalTree<T>::display() {
    displayHelper(root, 0);
    std::cout << std::endl;
}

template <typename T>
void ScidxAVLIntervalTree<T>::updateNode(ScidxavlNode<T>* node) {
    if (node != nullptr) {
        node->height = 1 + std::max(getHeight(node->left), getHeight(node->right));
        node->max_high = std::max(node->interval.high, std::max(node->left ? node->left->max_high : node->interval.high,
                                                               node->right ? node->right->max_high : node->interval.high));

        // 添加更新父节点的 max_high
        if (node->parent != nullptr) {
            updateNode(node->parent);
        }
    }
}



template <typename T>
std::vector<ScidxavlNode<T>*> ScidxAVLIntervalTree<T>::query(const ScidxavlInterval<T>& queryInterval) {
    std::vector<ScidxavlNode<T>*> result;
    queryHelper(root, queryInterval, result);
    return result;
}

template <typename T>
ScidxavlNode<T>* ScidxAVLIntervalTree<T>::getRoot() const {
    return root;
}

template <typename T>
void ScidxAVLIntervalTree<T>::rotateLeft(ScidxavlNode<T>*& node) {
    ScidxavlNode<T>* rightChild = node->right;
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

    updateNode(node);
    updateNode(rightChild);
}

template <typename T>
void ScidxAVLIntervalTree<T>::rotateRight(ScidxavlNode<T>*& node) {
    ScidxavlNode<T>* leftChild = node->left;
    node->left = leftChild->right;

    if (leftChild->right != nullptr) {
        leftChild->right->parent = node;
    }

    leftChild->parent = node->parent;

    if (node->parent == nullptr) {
        root = leftChild;
    } else if (node == node->parent->right) {
        node->parent->right = leftChild;
    } else {
        node->parent->left = leftChild;
    }

    leftChild->right = node;
    node->parent = leftChild;

    updateNode(node);
    updateNode(leftChild);
}

template <typename T>
void ScidxAVLIntervalTree<T>::fixInsertion(ScidxavlNode<T>*& node) {
    while (node != nullptr && node->parent != nullptr) {
        updateNode(node);
        int balanceFactor = getHeight(node->left) - getHeight(node->right);

        if (balanceFactor > 1) {
            if (node->left != nullptr && getHeight(node->left->right) > getHeight(node->left->left)) {
                rotateLeft(node->left);
            }
            rotateRight(node->parent);  // 修正此处的调用
        } else if (balanceFactor < -1) {
            if (node->right != nullptr && getHeight(node->right->left) > getHeight(node->right->right)) {
                rotateRight(node->right);
            }
            rotateLeft(node->parent);  // 修正此处的调用
        }

        // 更新节点信息
        updateNode(node);

        // 输出调试信息
        std::cout << "Current node: [" << node->interval.low << ", " << node->interval.high << "] at height " << node->height << std::endl;

        // 移动到父节点，添加对 node 是否为 nullptr 的检查
        if (node != nullptr) {
            node = node->parent;
        }
    }
}



template <typename T>
void ScidxAVLIntervalTree<T>::insert(const ScidxavlInterval<T>& interval, const size_t& id) {
    ScidxavlNode<T>* newNode = new ScidxavlNode<T>(interval, id);
    ScidxavlNode<T>* parent = nullptr;
    ScidxavlNode<T>* current = root;

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

    // 调用修复插入的函数
    fixInsertion(newNode);

    // 添加调试输出
    std::cout << "Inserted: [" << newNode->interval.low << ", " << newNode->interval.high << "] at height " << newNode->height << std::endl;

    // 输出整个树的状态
    // display();
}


template <typename T>
void displayHelper(ScidxavlNode<T>* root, int space) {
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

template <typename T>
bool doIntervalsIntersect(const ScidxavlInterval<T>& interval1, const ScidxavlInterval<T>& interval2) {
    return (interval1.low <= interval2.high && interval1.high >= interval2.low);
}

template <typename T>
void queryHelper(ScidxavlNode<T>* currentNode, const ScidxavlInterval<T>& queryInterval, std::vector<ScidxavlNode<T>*>& result) {
    if (currentNode == nullptr) {
        return;
    }

    if (doIntervalsIntersect(currentNode->interval, queryInterval)) {
        result.push_back(currentNode);
    }

    if (currentNode->left != nullptr && currentNode->left->max_high >= queryInterval.low) {
        queryHelper(currentNode->left, queryInterval, result);
    }

    if (currentNode->right != nullptr && currentNode->right->interval.low <= queryInterval.high) {
        queryHelper(currentNode->right, queryInterval, result);
    }
}

template <typename T>
void levelOrderTraversal(ScidxavlNode<T>* root, int levelsToTraverse, std::vector<std::vector<ScidxavlNode<T>*>>& result) {
    if (root == nullptr || levelsToTraverse <= 0) {
        return;
    }

    std::queue<ScidxavlNode<T>*> nodeQueue;
    nodeQueue.push(root);

    int currentLevel = 0;

    while (!nodeQueue.empty() && currentLevel < levelsToTraverse) {
        int nodesInCurrentLevel = nodeQueue.size();
        std::vector<ScidxavlNode<T>*> currentLevelNodes;

        for (int i = 0; i < nodesInCurrentLevel; ++i) {
            ScidxavlNode<T>* current = nodeQueue.front();
            nodeQueue.pop();

            currentLevelNodes.push_back(current);

            if (current->left != nullptr) {
                nodeQueue.push(current->left);
            }
            if (current->right != nullptr) {
                nodeQueue.push(current->right);
            }
        }

        result.push_back(currentLevelNodes);
        ++currentLevel;
    }
}

#endif /* _SCIDX_AVL_INTERVAL_TREE_H */

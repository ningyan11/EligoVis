#ifndef _SCIDX_RB_INTERVAL_TREE_H
#define _SCIDX_RB_INTERVAL_TREE_H

#include <scidx_defines.h>


#include <iostream>
#include <vector>

typedef enum ScidxColor 
{ RED, BLACK } ScidxColor;

template <typename T>
struct ScidxrbInterval {
    T low;
    T high;
};

template <typename T>
struct ScidxrbNode {
    ScidxrbInterval<T> rbInterval;
    size_t id;
    T max_high; // Maximum high value among current node's interval and its descendants
    ScidxColor color;
    ScidxrbNode* parent;
    ScidxrbNode* left;
    ScidxrbNode* right;

    // Constructor with default color as RED
    ScidxrbNode(const ScidxrbInterval<T>& _interval, size_t _id, ScidxColor _color = RED, ScidxrbNode* _parent = nullptr, ScidxrbNode* _left = nullptr, ScidxrbNode* _right = nullptr)
        : rbInterval(_interval), id(_id), max_high(_interval.high), color(_color), parent(_parent), left(_left), right(_right) {}
};

template <typename T>
class ScidxRedBlackIntervalTree {
private:
    ScidxrbNode<T>* root;

    void rotateLeft(ScidxrbNode<T>*&);
    void rotateRight(ScidxrbNode<T>*&);
    void fixInsertion(ScidxrbNode<T>*&);
    void updateMaxHigh(ScidxrbNode<T>*);

public:
    ScidxRedBlackIntervalTree() : root(nullptr) {}
    ScidxrbNode<T>* getRoot() const;
    void insert(const ScidxrbInterval<T>&, const size_t&);
    void display();
    std::vector<ScidxrbNode<T>*> query(const ScidxrbInterval<T>&);
};



template <typename T>
void ScidxRedBlackIntervalTree<T>::display() {
    displayHelper(root, 0);
    std::cout << std::endl;
}

template <typename T>
void ScidxRedBlackIntervalTree<T>::updateMaxHigh(ScidxrbNode<T>* node) {
    if (node != nullptr) {
        node->max_high = std::max(node->rbInterval.high, std::max(node->left ? node->left->max_high : node->rbInterval.high,
                                                               node->right ? node->right->max_high : node->rbInterval.high));
    }
}

// Method to query intervals that intersect with the given interval
template <typename T>
std::vector<ScidxrbNode<T>*> ScidxRedBlackIntervalTree<T>::query(const ScidxrbInterval<T>& queryInterval) {
    std::vector<ScidxrbNode<T>*> result;
    queryHelper(root, queryInterval, result);
    return result;
}

template <typename T>
ScidxrbNode<T>* ScidxRedBlackIntervalTree<T>::getRoot() const {
    return root;
}

template <typename T>
void ScidxRedBlackIntervalTree<T>::rotateLeft(ScidxrbNode<T>*& node) {
    ScidxrbNode<T>* rightChild = node->right;
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
void ScidxRedBlackIntervalTree<T>::rotateRight(ScidxrbNode<T>*& node) {
    ScidxrbNode<T>* leftChild = node->left;
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
void ScidxRedBlackIntervalTree<T>::fixInsertion(ScidxrbNode<T>*& node) {
    while (node != nullptr && node != root && node->parent != nullptr && node->parent->color == RED) {
        ScidxrbNode<T>* parent = node->parent;
        ScidxrbNode<T>* grandparent = parent->parent;

        if (parent == grandparent->left) {
            ScidxrbNode<T>* uncle = grandparent->right;

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
            ScidxrbNode<T>* uncle = grandparent->left;

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
void ScidxRedBlackIntervalTree<T>::insert(const ScidxrbInterval<T>& interval, const size_t& id) {
    ScidxrbNode<T>* newNode = new ScidxrbNode<T>(interval, id);
    ScidxrbNode<T>* parent = nullptr;
    ScidxrbNode<T>* current = root;

    while (current != nullptr) {
        parent = current;
        if (newNode->rbInterval.low < current->rbInterval.low) {
            current = current->left;
        } else {
            current = current->right;
        }
    }

    newNode->parent = parent;

    if (parent == nullptr) {
        root = newNode;
    } else if (newNode->rbInterval.low < parent->rbInterval.low) {
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
void displayHelper(ScidxrbNode<T>* root, int space) {
    if (root == nullptr) {
        return;
    }

    space += 5;

    displayHelper(root->right, space);

    std::cout << std::endl;
    for (int i = 5; i < space; i++) {
        std::cout << " ";
    }
    std::cout << "[" << root->rbInterval.low << ", " << root->rbInterval.high << "]"
              << " (max_high: " << root->max_high << ")"
              << "(" << (root->color == RED ? "RED" : "BLACK") << ")";
    displayHelper(root->left, space);
}

// Recursive helper function for querying
template <typename T>
void queryHelper(ScidxrbNode<T>* currentNode, const ScidxrbInterval<T>& queryInterval, std::vector<ScidxrbNode<T>*>& result) {
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

// Function to check if two intervals intersect
template <typename T>
bool doIntervalsIntersect(const ScidxrbInterval<T>& interval1, const ScidxrbInterval<T>& interval2) {
    return (interval1.low <= interval2.high && interval1.high >= interval2.low);
}

template <typename T>
void levelOrderTraversal(ScidxrbNode<T>* root, int levelsToTraverse, std::vector<std::vector<ScidxrbNode<T>*>>& result) {
    if (root == nullptr || levelsToTraverse <= 0) {
        return;
    }

    std::queue<ScidxrbNode<T>*> nodeQueue;
    nodeQueue.push(root);

    int currentLevel = 0;

    while (!nodeQueue.empty() && currentLevel < levelsToTraverse) {
        int nodesInCurrentLevel = nodeQueue.size();
        std::vector<ScidxrbNode<T>*> currentLevelNodes;

        for (int i = 0; i < nodesInCurrentLevel; ++i) {
            ScidxrbNode<T>* current = nodeQueue.front();
            nodeQueue.pop();

            // Process the current node
            currentLevelNodes.push_back(current);

            // Enqueue the left and right children, if they exist
            if (current->left != nullptr) {
                nodeQueue.push(current->left);
            }
            if (current->right != nullptr) {
                nodeQueue.push(current->right);
            }
        }

        // Move to the next level
        ++currentLevel;

        // Store the result of the current level
        result.push_back(currentLevelNodes);
    }
}


template <typename T>
void convertTreeToArray(ScidxrbNode<T>* root, std::vector<int>& result) {
    if (root == nullptr) {
        return;
    }

    std::queue<ScidxrbNode<T>*> nodeQueue;
    nodeQueue.push(root);

    while (!nodeQueue.empty()) {
        ScidxrbNode<T>* current = nodeQueue.front();
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

#endif /* ----- #ifndef _SCIDX_RB_INTERVAL_TREE_H  ----- */
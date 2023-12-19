#ifndef _SCIDX_RB_INTERVAL_TREE_H
#define _SCIDX_RB_INTERVAL_TREE_H

#include <scidx_defines.h>

typedef enum ScidxColor 
{ RED, BLACK } ScidxColor;

template <typename T>
struct ScidxInterval {
    T low;
    T high;
};

template <typename T>
struct ScidxNode {
    ScidxInterval<T> interval;
    T max_high; // Maximum high value among current node's interval and its descendants
    ScidxColor color;
    ScidxNode* parent;
    ScidxNode* left;
    ScidxNode* right;

    // Constructor with default color as RED
    ScidxNode(const ScidxInterval<T>& _interval, ScidxColor _color = RED, ScidxNode* _parent = nullptr, ScidxNode* _left = nullptr, ScidxNode* _right = nullptr)
        : interval(_interval), max_high(_interval.high), color(_color), parent(_parent), left(_left), right(_right) {}
};

template <typename T>
class ScidxRedBlackIntervalTree {
private:
    ScidxNode<T>* root;

    void rotateLeft(ScidxNode<T>*&);
    void rotateRight(ScidxNode<T>*&);
    void fixInsertion(ScidxNode<T>*&);
    void updateMaxHigh(ScidxNode<T>*);

public:
    ScidxRedBlackIntervalTree() : root(nullptr) {}
    ScidxNode<T>* getRoot() const;
    void insert(const ScidxInterval<T>&);
    void display();
};

template <typename T>
void displayHelper(ScidxNode<T>* root, int space) {
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

template <typename T>
void ScidxRedBlackIntervalTree<T>::display() {
    displayHelper(root, 0);
    std::cout << std::endl;
}

template <typename T>
void ScidxRedBlackIntervalTree<T>::updateMaxHigh(ScidxNode<T>* node) {
    if (node != nullptr) {
        node->max_high = std::max(node->interval.high, std::max(node->left ? node->left->max_high : node->interval.high,
                                                               node->right ? node->right->max_high : node->interval.high));
    }
}

template <typename T>
ScidxNode<T>* ScidxRedBlackIntervalTree<T>::getRoot() const {
    return root;
}

template <typename T>
void ScidxRedBlackIntervalTree<T>::rotateLeft(ScidxNode<T>*& node) {
    ScidxNode<T>* rightChild = node->right;
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
void ScidxRedBlackIntervalTree<T>::rotateRight(ScidxNode<T>*& node) {
    ScidxNode<T>* leftChild = node->left;
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
void ScidxRedBlackIntervalTree<T>::fixInsertion(ScidxNode<T>*& node) {
    while (node != nullptr && node != root && node->parent != nullptr && node->parent->color == RED) {
        ScidxNode<T>* parent = node->parent;
        ScidxNode<T>* grandparent = parent->parent;

        if (parent == grandparent->left) {
            ScidxNode<T>* uncle = grandparent->right;

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
            ScidxNode<T>* uncle = grandparent->left;

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
void ScidxRedBlackIntervalTree<T>::insert(const ScidxInterval<T>& interval) {
    ScidxNode<T>* newNode = new ScidxNode<T>(interval);
    ScidxNode<T>* parent = nullptr;
    ScidxNode<T>* current = root;

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
void levelOrderTraversal(ScidxNode<T>* root, int levelsToTraverse, std::vector<std::vector<ScidxNode<T>*>>& result) {
    if (root == nullptr || levelsToTraverse <= 0) {
        return;
    }

    std::queue<ScidxNode<T>*> nodeQueue;
    nodeQueue.push(root);

    int currentLevel = 0;

    while (!nodeQueue.empty() && currentLevel < levelsToTraverse) {
        int nodesInCurrentLevel = nodeQueue.size();
        std::vector<ScidxNode<T>*> currentLevelNodes;

        for (int i = 0; i < nodesInCurrentLevel; ++i) {
            ScidxNode<T>* current = nodeQueue.front();
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

#endif /* ----- #ifndef _SCIDX_RB_INTERVAL_TREE_H  ----- */

#ifndef _SCIDX_RB_INTERVAL_TREE_H
#define _SCIDX_RB_INTERVAL_TREE_H

#include <scidx_defines.h>

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
    void insert(const ScidxInterval<T>&, const size_t&);
    void display();
    std::vector<ScidxRBNode<T>*> query(const ScidxInterval<T>&);
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
ScidxRBNode<T>* ScidxRedBlackIntervalTree<T>::getRoot() const {
    return root;
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
void levelOrderTraversal(ScidxRBNode<T>* root, int levelsToTraverse, std::vector<std::vector<ScidxRBNode<T>*>>& result) {
    if (root == nullptr || levelsToTraverse <= 0) {
        return;
    }

    std::queue<ScidxRBNode<T>*> nodeQueue;
    nodeQueue.push(root);

    int currentLevel = 0;

    while (!nodeQueue.empty() && currentLevel < levelsToTraverse) {
        int nodesInCurrentLevel = nodeQueue.size();
        std::vector<ScidxRBNode<T>*> currentLevelNodes;

        for (int i = 0; i < nodesInCurrentLevel; ++i) {
            ScidxRBNode<T>* current = nodeQueue.front();
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

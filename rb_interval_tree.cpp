#include <vector>
#include <queue>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>

enum Color { RED, BLACK };

template <typename T>
struct Interval {
    T low;
    T high;
};

template <typename T>
struct Node {
    Interval<T> interval;
    T max_high; // Maximum high value among current node's interval and its descendants
    Color color;
    Node* parent;
    Node* left;
    Node* right;

    // Constructor with default color as RED
    Node(const Interval<T>& _interval, Color _color = RED, Node* _parent = nullptr, Node* _left = nullptr, Node* _right = nullptr)
        : interval(_interval), max_high(_interval.high), color(_color), parent(_parent), left(_left), right(_right) {}
};

template <typename T>
class RedBlackIntervalTree {
private:
    Node<T>* root;

    void rotateLeft(Node<T>*&);
    void rotateRight(Node<T>*&);
    void fixInsertion(Node<T>*&);
    void updateMaxHigh(Node<T>*);

public:
    RedBlackIntervalTree() : root(nullptr) {}
    Node<T>* getRoot() const;
    void insert(const Interval<T>&);
    void display();
};

template <typename T>
void displayHelper(Node<T>* root, int space) {
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
void RedBlackIntervalTree<T>::display() {
    displayHelper(root, 0);
    std::cout << std::endl;
}

template <typename T>
void RedBlackIntervalTree<T>::updateMaxHigh(Node<T>* node) {
    if (node != nullptr) {
        node->max_high = std::max(node->interval.high, std::max(node->left ? node->left->max_high : node->interval.high,
                                                               node->right ? node->right->max_high : node->interval.high));
    }
}

template <typename T>
Node<T>* RedBlackIntervalTree<T>::getRoot() const {
    return root;
}

template <typename T>
void RedBlackIntervalTree<T>::rotateLeft(Node<T>*& node) {
    Node<T>* rightChild = node->right;
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
void RedBlackIntervalTree<T>::rotateRight(Node<T>*& node) {
    Node<T>* leftChild = node->left;
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
void RedBlackIntervalTree<T>::fixInsertion(Node<T>*& node) {
    while (node != nullptr && node != root && node->parent != nullptr && node->parent->color == RED) {
        Node<T>* parent = node->parent;
        Node<T>* grandparent = parent->parent;

        if (parent == grandparent->left) {
            Node<T>* uncle = grandparent->right;

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
            Node<T>* uncle = grandparent->left;

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
void RedBlackIntervalTree<T>::insert(const Interval<T>& interval) {
    Node<T>* newNode = new Node<T>(interval);
    Node<T>* parent = nullptr;
    Node<T>* current = root;

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
void levelOrderTraversal(Node<T>* root, int levelsToTraverse, std::vector<std::vector<Node<T>*>>& result) {
    if (root == nullptr || levelsToTraverse <= 0) {
        return;
    }

    std::queue<Node<T>*> nodeQueue;
    nodeQueue.push(root);

    int currentLevel = 0;

    while (!nodeQueue.empty() && currentLevel < levelsToTraverse) {
        int nodesInCurrentLevel = nodeQueue.size();
        std::vector<Node<T>*> currentLevelNodes;

        for (int i = 0; i < nodesInCurrentLevel; ++i) {
            Node<T>* current = nodeQueue.front();
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


int main() {
    std::vector<Interval<float>> intervals;
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

                Interval<float> interval;
                interval.low = min;
                interval.high = max;

                intervals.push_back(interval);
            }
        }
    }

    RedBlackIntervalTree<float> rbIntervalTree;

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

    std::vector<std::vector<std::vector<Node<float>*>>> allSubTrees;
    std::vector<std::vector<Node<float>*>> firstSubTreeNodesInLevels;

    levelOrderTraversal(rbIntervalTree.getRoot(), levelsToTraverse, firstSubTreeNodesInLevels);
    allSubTrees.push_back(firstSubTreeNodesInLevels);

    std::vector<Node<float>*>& lastLevelOfFirstSubTree = firstSubTreeNodesInLevels.back();

    std::vector<Node<float>*> rootsOfNewSubTrees;

    rootsOfNewSubTrees.insert(rootsOfNewSubTrees.end(), lastLevelOfFirstSubTree.begin(), lastLevelOfFirstSubTree.end());

    while (!rootsOfNewSubTrees.empty())
    {
        
        Node<float>* node = rootsOfNewSubTrees.front();
        rootsOfNewSubTrees.erase(rootsOfNewSubTrees.begin());  

        if (node->left == nullptr && node->right == nullptr)
        {
            continue;
        }
        

        std::vector<std::vector<Node<float>*>> currentSubTreeNodesInLevels;
        levelOrderTraversal(node, levelsToTraverse, currentSubTreeNodesInLevels);   
        allSubTrees.push_back(currentSubTreeNodesInLevels);
        std::vector<Node<float>*>& lastLevelOfCurrentSubTree = currentSubTreeNodesInLevels.back();

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

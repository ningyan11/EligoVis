#ifndef _SCIDX_AVL_INTERVAL_TREE_H
#define _SCIDX_AVL_INTERVAL_TREE_H

#include <scidx_defines.h>

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

};

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



#endif /* ----- #ifndef _SCIDX_AVL_INTERVAL_TREE_H  ----- */
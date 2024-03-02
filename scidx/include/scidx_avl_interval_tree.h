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

    void insert(ScidxAVLNode<T>*& node, size_t id, ScidxInterval<T> interval);

    ScidxAVLNode<T>* rotateLeft(ScidxAVLNode<T>* x);

    ScidxAVLNode<T>* rotateRight(ScidxAVLNode<T>* y);

    int getBalance(ScidxAVLNode<T>* node);

public:
    ScidxAVLIntervalTree() : root(nullptr) {}

    void insertNode(size_t, ScidxInterval<T>);

    void display();

};

template<typename T>
int ScidxAVLIntervalTree<T>::height(ScidxAVLNode<T>* node) {
    if (node == nullptr)
        return 0;
    return node->height;
}

template<typename T>
void ScidxAVLIntervalTree<T>::insert(ScidxAVLNode<T>*& node, size_t id, ScidxInterval<T> interval) {
    if (node == nullptr) {
        node = new ScidxAVLNode<T>(interval, id);
        return;
    }

    if (interval.low < node->interval.low)
        insert(node->left, id, interval);
    else
        insert(node->right, id, interval);

    node->height = max(height(node->left), height(node->right)) + 1;

    int balance = getBalance(node);

    if (balance > 1 && interval.low < node->left->interval.low)
        node = rotateRight(node);

    if (balance < -1 && interval.low > node->right->interval.low)
        node = rotateLeft(node);

    if (balance > 1 && interval.low > node->left->interval.low) {
        node->left = rotateLeft(node->left);
        node = rotateRight(node);
    }

    if (balance < -1 && interval.low < node->right->interval.low) {
        node->right = rotateRight(node->right);
        node = rotateLeft(node);
    }

    node->max_high = max(node->interval.high, max((node->left ? node->left->max_high : std::numeric_limits<T>::min()), (node->right ? node->right->max_high : std::numeric_limits<T>::min())));
}

template<typename T>
ScidxAVLNode<T>* ScidxAVLIntervalTree<T>::rotateLeft(ScidxAVLNode<T>* x) {
    ScidxAVLNode<T>* y = x->right;
    ScidxAVLNode<T>* T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = max(height(x->left), height(x->right)) + 1;
    y->height = max(height(y->left), height(y->right)) + 1;

    x->max_high = max(x->interval.high, max((x->left ? x->left->max_high : std::numeric_limits<T>::min()), (x->right ? x->right->max_high : std::numeric_limits<T>::min())));
    y->max_high = max(y->interval.high, max((y->left ? y->left->max_high : std::numeric_limits<T>::min()), (y->right ? y->right->max_high : std::numeric_limits<T>::min())));

    return y;
}

template<typename T>
ScidxAVLNode<T>* ScidxAVLIntervalTree<T>::rotateRight(ScidxAVLNode<T>* y) {
    ScidxAVLNode<T>* x = y->left;
    ScidxAVLNode<T>* T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = max(height(y->left), height(y->right)) + 1;
    x->height = max(height(x->left), height(x->right)) + 1;

    y->max_high = max(y->interval.high, max((y->left ? y->left->max_high : std::numeric_limits<T>::min()), (y->right ? y->right->max_high : std::numeric_limits<T>::min())));
    x->max_high = max(x->interval.high, max((x->left ? x->left->max_high : std::numeric_limits<T>::min()), (x->right ? x->right->max_high : std::numeric_limits<T>::min())));

    return x;
}

template<typename T>
int ScidxAVLIntervalTree<T>::getBalance(ScidxAVLNode<T>* node) {
    if (node == nullptr)
        return 0;
    return height(node->left) - height(node->right);
}

template<typename T>
 void ScidxAVLIntervalTree<T>::insertNode(size_t id, ScidxInterval<T> interval) {
    insert(root, id, interval);
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

#endif /* ----- #ifndef _SCIDX_AVL_INTERVAL_TREE_H  ----- */
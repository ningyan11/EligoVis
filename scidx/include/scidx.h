/**
 * Scientific Compact Data Index (scidx)
 * Authors: Ning Yan, Lipeng Wan, Sheng Di
 *
**/


#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <scidx_rw.h>
#include <scidx_block_min_max.h>

#ifndef _SCIDX_H
#define _SCIDX_H

#ifdef _WIN32
#define PATH_SEPARATOR ';'
#else
#define PATH_SEPARATOR ':'
#endif

#ifdef __cplusplus
extern "C" {
#endif

void compressTree(std::vector<std::vector<ScidxRBNode<float> *>> singleSubTree, float error_bound, std::vector<int> firstVector);


#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SCIDX_H  ----- */

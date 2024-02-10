/**
 * Scientific Compact Data Index (scidx)
 * Authors: Ning Yan, Lipeng Wan, Sheng Di
 *
**/


#include <scidx_rb_interval_tree.h>
#include <scidx_rw.h>
#include <scidx_block_min_max.h>

std::vector<std::vector<int>> compress_index(std::vector<std::vector<ScidxrbNode<float>*>> subTreeNodesInLevels, float error_bound);

std::vector<int> compress_data_layered(std::vector<std::vector<ScidxrbNode<float>*>> subTreeNodesInLevels, float error_bound);


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


#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _SCIDX_H  ----- */

#ifndef _SCIDX_BLOCK_MIN_MAX_H
#define _SCIDX_BLOCK_MIN_MAX_H

#include <scidx_defines.h>


std::vector<size_t> positionToIndices(size_t position, const std::vector<size_t>& shape);
size_t indicesToPosition(const std::vector<size_t>& shape, const std::vector<size_t>& indices);
template <typename T> std::vector<std::vector<T>> obtainBlockMinMax(std::vector<T> data, std::vector<size_t> dataShape, std::vector<size_t> blockShape);


#endif
#include <vector>
#include "scidx_block_min_max.h"

std::vector<size_t> positionToIndices(size_t position, const std::vector<size_t>& shape) 
{
    std::vector<size_t> indices;
    size_t remainingPosition = position;

    for (auto dimensionSize = shape.rbegin(); dimensionSize != shape.rend(); ++dimensionSize) 
    {
        indices.insert(indices.begin(), remainingPosition % *dimensionSize);
        remainingPosition /= *dimensionSize;
    }

    return indices;
}

size_t indicesToPosition(const std::vector<size_t>& shape, const std::vector<size_t>& indices) {
    size_t position = indices[0];
    size_t multiplier = 1;

    for (size_t i = 1; i < shape.size(); ++i) {
        multiplier *= shape[i - 1];
        position += indices[i] * multiplier;
    }

    return position;
}

template <typename T>
std::vector<std::vector<T>> obtainBlockMinMax(std::vector<T> data, std::vector<size_t> dataShape, std::vector<size_t> blockShape) 
{
    size_t nElem = data.size();
    size_t nDim = dataShape.size();
    std::vector<size_t> blockCountOnEachDim;
    size_t total_blocks = 1;
    for (size_t i = 0; i < nDim; i++)
    {
        blockCountOnEachDim.push_back(dataShape[i]/blockShape[i]);
        total_blocks *= dataShape[i]/blockShape[i];
    }
    std::vector<T> block_mins(total_blocks, 0);
    std::vector<T> block_maxs(total_blocks, 0);

    for (size_t p = 0; p < nElem; p++)
    {
        std::vector<size_t> elem_global_id = positionToIndices(p, dataShape);
        std::vector<size_t> block_global_id(nDim);
        for (size_t i = 0; i < nDim; i++)
        {
            block_global_id[i] = elem_global_id[i]/blockShape[i];
        }
        
        size_t block_position = indicesToPosition(blockCountOnEachDim, block_global_id);

        if (data[p] < block_mins[block_position])
        {
            block_mins[block_position] = data[p];
        }
        if (data[p] > block_maxs[block_position])
        {
            block_maxs[block_position] = data[p];
        }
    }
    std::vector<std::vector<T>> results;

    for (size_t i = 0; i < total_blocks; i++)
    {
        std::vector<T> mim_max_pair;
        mim_max_pair.push_back(block_mins[i]);
        mim_max_pair.push_back(block_maxs[i]);
        results.push_back(mim_max_pair);
    }
    
    return results;
}
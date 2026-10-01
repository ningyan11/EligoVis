#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <scidx_avl.h> 
#include <SZ3/api/sz.hpp> 
#include <scidx_block_min_max.h>
#include <scidx_Huffman.h>
#include <scidx_rb_interval_tree.h>
#include <scidx_avl_interval_tree.h>
#include <utility> 
#include <scidx_avl_interval_tree.h>  
#include <scidx_block_min_max.h>      
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem> 
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include "../miniIsosurface/marchingCubes/util/Image3D.h"
#include "../miniIsosurface/marchingCubes/util/TriangleMesh.h"
#include "../miniIsosurface/marchingCubes/util/SaveTriangleMesh.h"
#include "../miniIsosurface/marchingCubes/util/util.h"
#include "../miniIsosurface/marchingCubes/util/MarchingCubesTables.h"
#include "../miniIsosurface/marchingCubes/util/Timer.h"
#include "../miniIsosurface/marchingCubes/util/LoadImage.h"
#include "../miniIsosurface/marchingCubes/mantevoCommon/YAML_Doc.hpp"
#include <unordered_map>
#include <unordered_set>



 void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
}


std::vector<ScidxInterval<double>> loadBigBlockIndexFile(const std::string& filename) {
    std::vector<ScidxInterval<double>> bigBlockIndices;

    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return bigBlockIndices;  // 返回空
    }

    double minVal, maxVal;
    while (inFile.read(reinterpret_cast<char*>(&minVal), sizeof(double)) &&
           inFile.read(reinterpret_cast<char*>(&maxVal), sizeof(double))) {
        bigBlockIndices.push_back({minVal, maxVal});
    }

    inFile.close();
    return bigBlockIndices;
}




/*size_t computeGlobalSmallId(size_t block_id, size_t local_small_block_idx,
                            size_t Nx, size_t Ny, size_t Nz,
                            size_t sx, size_t sy, size_t sz,
                            size_t small_x, size_t small_y, size_t small_z) {
    // 每个维度上大块数
    size_t bx = Nx / sx;
    size_t by = Ny / sy;
    size_t bz = Nz / sz;

    size_t big_blocks_per_step = bx * by * bz;

    // 根据block_id得到step索引和step内大块索引
    size_t step_idx = block_id / (bx * by * bz);
    size_t block_id_in_step = block_id % (bx * by * bz);

    // 大块的三维坐标
    size_t big_block_z = block_id_in_step % bz;
    size_t big_block_y = (block_id_in_step / bz) % by;
    size_t big_block_x = block_id_in_step / (by * bz);

    // 每个大块内小块数量
    size_t local_small_blocks_x = sx / small_x;
    size_t local_small_blocks_y = sy / small_y;
    size_t local_small_blocks_z = sz / small_z;

    // 小块在大块内的坐标
    size_t local_small_z = local_small_block_idx % local_small_blocks_z;
    size_t local_small_y = (local_small_block_idx / local_small_blocks_z) % local_small_blocks_y;
    size_t local_small_x = local_small_block_idx / (local_small_blocks_y * local_small_blocks_z);

    // 小块在step内的全局三维坐标
    size_t global_small_x = big_block_x * local_small_blocks_x + local_small_x;
    size_t global_small_y = big_block_y * local_small_blocks_y + local_small_y;
    size_t global_small_z = big_block_z * local_small_blocks_z + local_small_z;

    // step内小块个数
    size_t global_small_blocks_Ny = Ny / small_y;
    size_t global_small_blocks_Nz = Nz / small_z;

    // 最终globalsmallid
    size_t globalsmallid = global_small_x * global_small_blocks_Ny * global_small_blocks_Nz
                           + global_small_y * global_small_blocks_Nz
                           + global_small_z;

    globalsmallid += step_idx * (Nx / small_x)*(Ny / small_y)*(Nz / small_z);

    return globalsmallid;
}*/

size_t computeGlobalSmallId(size_t block_id, size_t local_small_block_idx,
    size_t Nx, size_t Ny, size_t Nz,
    size_t sx, size_t sy, size_t sz,
    size_t small_x, size_t small_y, size_t small_z) {
// === Step 1: 每个 step 的大块数量 ===
size_t bx = Nx / sx;
size_t by = Ny / sy;
size_t bz = Nz / sz;

size_t big_blocks_per_step = bx * by * bz;

// === Step 2: 当前 block 属于哪个 step ===
size_t step_idx = block_id / big_blocks_per_step;
size_t block_id_in_step = block_id % big_blocks_per_step;

// === Step 3: 大块的三维坐标（按 row-major）===
size_t big_block_x = block_id_in_step % bx;
size_t big_block_y = (block_id_in_step / bx) % by;
size_t big_block_z = block_id_in_step / (bx * by);

// === Step 4: 每个大块内的小块数量 ===
size_t local_small_blocks_x = sx / small_x;
size_t local_small_blocks_y = sy / small_y;
size_t local_small_blocks_z = sz / small_z;

// === Step 5: 小块在大块内的坐标（row-major 展开）===
size_t local_small_x = local_small_block_idx % local_small_blocks_x;
size_t local_small_y = (local_small_block_idx / local_small_blocks_x) % local_small_blocks_y;
size_t local_small_z = local_small_block_idx / (local_small_blocks_x * local_small_blocks_y);

// === Step 6: 小块在整个 step 中的全局坐标 ===
size_t global_small_x = big_block_x * local_small_blocks_x + local_small_x;
size_t global_small_y = big_block_y * local_small_blocks_y + local_small_y;
size_t global_small_z = big_block_z * local_small_blocks_z + local_small_z;

// === Step 7: 计算全局小块总布局信息 ===
size_t global_blocks_x = Nx / small_x;
size_t global_blocks_y = Ny / small_y;
size_t global_blocks_z = Nz / small_z;

// === Step 8: 在当前 step 内的小块 ID（row-major flatten） ===
size_t global_id_in_step = global_small_x +
       global_small_y * global_blocks_x +
       global_small_z * global_blocks_x * global_blocks_y;

// === Step 9: 加上 step 偏移 ===
size_t globalsmallid = global_id_in_step +
   step_idx * global_blocks_x * global_blocks_y * global_blocks_z;

return globalsmallid;
}

std::vector<size_t> safeQueryOverlapIds(ScidxAVLNode<double>* root, const ScidxInterval<double>& query) {
    if (root && root->id == static_cast<size_t>(-1)) {
        return {};  // dummy 树，返回空
    }
    return queryOverlapIds(root, query);
}


// ============================================================
// 新增函数1：直接查询步级 index 树
// 去掉大块层后，每步只有一棵树，文件名格式为 {stepNum}-0
// 返回该步内的原始 flat 小块 ID（均匀或错位 index 通用）
// ============================================================
std::vector<size_t> queryIndexRawIds(
    size_t actualStepNum,
    const std::string& indexDir,
    const std::vector<double>& queryRange,
    double error_bound)
{
    // 无大块层，localBlockID 固定为 0
    std::string treeID = indexDir + std::to_string(actualStepNum) + "-0";

    std::cout << "[queryIndexRawIds] treeID: " << treeID << std::endl;

    auto [decompressedTree, decodedSkippedMin, decodedSkippedMax, skippedIdOut] =
        decompressOptimizedAVL(treeID, error_bound);
    
    auto time_before_query = std::chrono::high_resolution_clock::now();

    adjustAndUpdateMaxHigh(decompressedTree, error_bound);

    ScidxInterval<double> query;
    query.low  = queryRange[0];
    query.high = queryRange[1];

    auto ids = safeQueryOverlapIds(decompressedTree, query);

    for (size_t i = 0; i < decodedSkippedMin.size(); ++i) {
        if (skippedIdOut[i] == static_cast<size_t>(-1)) continue;
        double minWithError = decodedSkippedMin[i] - error_bound;
        double maxWithError = decodedSkippedMax[i] + error_bound;
        if (!(minWithError > query.high || maxWithError < query.low)) {
            ids.push_back(skippedIdOut[i]);
        }
    }

    auto time_after_query = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> adjustAndQueryTime = time_after_query - time_before_query;
    std::cerr << "[Time5] adjustAndQueryInTree (" << treeID << "): "
              << adjustAndQueryTime.count() << " seconds" << std::endl;


    std::cout << "[queryIndexRawIds] Step " << actualStepNum
              << " -> " << ids.size() << " raw block IDs" << std::endl;
    return ids;
}

// ============================================================
// 新增函数2：错位块 flat ID -> 覆盖的均匀块 flat ID 集合
//
// 块 flat ID 编码（与压缩侧一致，dim0 最快变化）：
//   id = coord[0] + coord[1]*count[0] + coord[2]*count[0]*count[1]
//
// 错位块 k 在维度 d 的数据范围：
//   k=0  : [0, halfBlockShape[d])
//   k>=1 : [halfBlockShape[d] + (k-1)*smallBlockShape[d],
//            halfBlockShape[d] +  k   *smallBlockShape[d]) ∩ [0, shape[d])
//
// 找出该范围内覆盖的均匀块索引，每维最多 2 个，共最多 8 个。
// ============================================================
std::vector<size_t> convertStaggerToUniformIds(
    size_t staggerFlatId,
    const std::vector<size_t>& shape,           // stepDataShape
    const std::vector<size_t>& smallBlockShape,
    const std::vector<size_t>& halfBlockShape,
    const std::vector<size_t>& staggerBlockCount,
    const std::vector<size_t>& uniformBlockCount)
{
    // 解码错位 flat ID -> 3D 坐标（dim0 最快变化）
    size_t sk0 = staggerFlatId % staggerBlockCount[0];
    size_t sk1 = (staggerFlatId / staggerBlockCount[0]) % staggerBlockCount[1];
    size_t sk2 = staggerFlatId / (staggerBlockCount[0] * staggerBlockCount[1]);
    std::array<size_t, 3> sk = {sk0, sk1, sk2};

    std::array<std::pair<size_t, size_t>, 3> uRange;
    for (int d = 0; d < 3; d++) {
        size_t k = sk[d];
        size_t start, end_excl;

        if (k == 0) {
            start    = 0;
            end_excl = halfBlockShape[d];
        } else {
            start    = halfBlockShape[d] + (k - 1) * smallBlockShape[d];
            end_excl = std::min(start + smallBlockShape[d], shape[d]);
        }

        if (start >= shape[d] || end_excl == 0 || start >= end_excl) {
            return {};  // 超出数据范围
        }

        size_t u_start = start / smallBlockShape[d];
        size_t u_end   = (end_excl - 1) / smallBlockShape[d];
        u_end = std::min(u_end, uniformBlockCount[d] - 1);
        uRange[d] = {u_start, u_end};
    }

    // 枚举所有覆盖的均匀块（最多 2×2×2 = 8 个）
    std::vector<size_t> result;
    for (size_t ux = uRange[0].first; ux <= uRange[0].second; ux++) {
        for (size_t uy = uRange[1].first; uy <= uRange[1].second; uy++) {
            for (size_t uz = uRange[2].first; uz <= uRange[2].second; uz++) {
                size_t flatId = ux
                              + uy * uniformBlockCount[0]
                              + uz * uniformBlockCount[0] * uniformBlockCount[1];
                result.push_back(flatId);
            }
        }
    }
    return result;
}


//查找固定的大块中小块构建的tree，并查找samelow的interval，返回ID
std::vector<size_t> process_query_task(size_t globalBlockID, size_t beginStepNum, size_t total_bigBlocks_per_step, 
                        const std::vector<double>& queryRange, const std::string& indexDir, double error_bound, size_t Nx, 
                        size_t sx, 
                        size_t small_x) {

    int rank = 0;

   

    //auto start_time_processTree = std::chrono::high_resolution_clock::now();
    // **转换全局 blockID 为 stepID 和 localBlockID**

    std::cout << "[Debug] beginStepNum = " << beginStepNum
          << ", globalBlockID = " << globalBlockID
          << ", total_bigBlocks_per_step = "
          << total_bigBlocks_per_step << "\n";
    
    size_t stepID = beginStepNum + (globalBlockID / total_bigBlocks_per_step);
    //step中对应的第n个大块
    size_t localBlockID = globalBlockID % total_bigBlocks_per_step;

   

    // **构造文件名**
    std::string blockPrefix = std::to_string(stepID) + "-" + std::to_string(localBlockID);
 
    std::string treeID = indexDir + blockPrefix;
    
    std::cout << "treeID:" << treeID << std::endl;
    auto [decompressedTree, decodedSkippedMin, decodedSkippedMax, skippedIdOut] = decompressOptimizedAVL(treeID, error_bound);
    
   

    auto time_finish_decompressTree = std::chrono::high_resolution_clock::now();


    //std::cout << "[DEBUG] 2 = " << total_bigBlocks_per_step << std::endl;

    //display tree
    /*ScidxAVLIntervalTree<double> tree;
    tree.setRoot(decompressedTree);
    tree.display();*/

    //调整min,max,maxhigh查找区间（消除errorbound）
    adjustAndUpdateMaxHigh(decompressedTree, error_bound);
    //std::cout << "[DEBUG] 3 = " << total_bigBlocks_per_step << std::endl;

 


    //auto time_finish_adjustTree = std::chrono::high_resolution_clock::now();

    


    ScidxInterval<double> query;
    query.low = queryRange[0];
    query.high = queryRange[1];

    //查询overlap的id
    //std::vector<size_t> ids = queryOverlapIds(decompressedTree, query);

    auto ids = safeQueryOverlapIds(decompressedTree, query);


    std::vector<size_t> overlappingSkippedIds;

    for (size_t i = 0; i < decodedSkippedMin.size(); ++i) {
        if (skippedIdOut[i] == static_cast<size_t>(-1)) {
            continue;
        }
        
        // Construct the interval for the skipped node, considering the error bounds
        double minWithError = decodedSkippedMin[i] - error_bound;
        double maxWithError = decodedSkippedMax[i] + error_bound;

        // Check if the intervals overlap
        if (!(minWithError > query.high || maxWithError < query.low)) {
            // If there is an overlap, store the skipped ID
            overlappingSkippedIds.push_back(skippedIdOut[i]);
        }
    }

    ids.insert(ids.end(), overlappingSkippedIds.begin(), overlappingSkippedIds.end());

    std::vector<size_t> globalIDs;

    //for test
    /*std::cout << "[DEBUG] Overlap local_small_block_idx count = " << ids.size() << std::endl;
    std::cout << "[DEBUG] Overlap local_small_block_idx values: ";
    for (size_t id : ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;*/
    
    // 打印 computeGlobalSmallId 参数前缀
    /*std::cout << "[DEBUG] GlobalBlockID = " << globalBlockID 
              << ", Nx = " << Nx << ", sx = " << sx << ", small_x = " << small_x << std::endl;*/
      


    for (size_t id : ids) {

        size_t globalsmallid = computeGlobalSmallId(
            globalBlockID, id,
            Nx, Nx, Nx,
            sx, sx, sx,
            small_x, small_x, small_x
        );

        globalIDs.push_back(globalsmallid);
    
    }



    std::cout << "[DEBUG] process_query_task on block " << globalBlockID << " → small block count: " << globalIDs.size() << std::endl;


    auto time_finish_queryTree = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> adjustAndQueryTree = time_finish_queryTree - time_finish_decompressTree;
    std::cerr << "[Rank " << rank << "[Time5]: adjustAndQueryInTree: " << adjustAndQueryTree.count() << " seconds" << std::endl;
    
    /*// **读取小块数据**
    std::vector<double> finalResults;
    for (int smallBlockID : smallBlockIDs) {
        std::string blockFile = indexDir + blockPrefix + "-" + std::to_string(smallBlockID) + "-data.dat";
        std::ifstream file(blockFile, std::ios::binary);
        if (!file) {
            std::cerr << "[Rank " << MPI::COMM_WORLD.Get_rank() << "] Error opening small block file " << blockFile << std::endl;
            continue;
        }
        double val;
        while (file.read(reinterpret_cast<char*>(&val), sizeof(double))) {
            finalResults.push_back(val);
        }
    }

   

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);  // 获取进程 rank

    //std::cout << "[Rank " << mpi_rank << "] Read & decompress index: " << index_read_time.count();*/

    return globalIDs;

}


    //blockSize每小块数据个数大小，totalBlocks全部原数据小块大小
    std::vector<std::vector<double>> batchDecompressBlocks(const std::string& subDir,
        const std::vector<size_t>& blockIds,
        size_t blockSize,
        size_t totalBlocks,
        double error_bound) {

            int rank = 0;

            std::cout << "read: "  << std::endl;

            auto beginReadOriginalBlocks = std::chrono::high_resolution_clock::now(); 


            std::vector<std::vector<double>> decompressedOriginalresult;

            //double memory_access_time = 0.0;
            //double decompress_only_time = 0.0;

            std::string unpredDataFileName = subDir + "fixed_unpredData.bin";
            std::string compressedFileName = subDir + "fixed_compressed_data.bin";
            std::string signFileName = subDir + "fixed_sign_data.bin";

            //std::cout << "Sign file path: " << signFileName << std::endl;

            std::ifstream unpredStream(unpredDataFileName, std::ios::binary);
            std::ifstream compStream(compressedFileName, std::ios::binary);
            std::ifstream signStream(signFileName, std::ios::binary);


            if (!unpredStream || !compStream || !signStream) {
            throw std::runtime_error("Failed to open one or more input files.");
            }

            /*//读取头文件，方便后续解压的时候access 

            // --- 读取 unpredSizes ---
           
            std::vector<unsigned char> unpredSizes(totalBlocks);
            unpredStream.read(reinterpret_cast<char*>(unpredSizes.data()), totalBlocks);
           

            // --- 读取压缩数据头部 (bitCounts 和 compSizes) ---
            std::vector<unsigned char> bitCounts(totalBlocks);
            std::vector<unsigned char> compSizes(totalBlocks);
            compStream.read(reinterpret_cast<char*>(bitCounts.data()), totalBlocks);
            compStream.read(reinterpret_cast<char*>(compSizes.data()), totalBlocks);

           
            const size_t signBytesPerBlock = (blockSize + 7) / 8;



            std::vector<std::vector<double>> allUnpredData;
            std::vector<std::vector<unsigned char>> allCompData;
            std::vector<std::vector<unsigned char>> allSignBits;*/


            // 获取文件大小
            unpredStream.seekg(0, std::ios::end);
            size_t unpredFileSize = unpredStream.tellg();
            unpredStream.seekg(0, std::ios::beg);

            compStream.seekg(0, std::ios::end);
            size_t compFileSize = compStream.tellg();
            compStream.seekg(0, std::ios::beg);

            signStream.seekg(0, std::ios::end);
            size_t signFileSize = signStream.tellg();
            signStream.seekg(0, std::ios::beg);

            // 分配足够大的内存
            std::vector<unsigned char> allUnpredDataRead(unpredFileSize);
            std::vector<unsigned char> allCompDataRead(compFileSize);
            std::vector<unsigned char> allSignDataRead(signFileSize);

            // 读取文件数据到内存
            unpredStream.read(reinterpret_cast<char*>(allUnpredDataRead.data()), unpredFileSize);
            compStream.read(reinterpret_cast<char*>(allCompDataRead.data()), compFileSize);
            signStream.read(reinterpret_cast<char*>(allSignDataRead.data()), signFileSize);


            // 读取 unpredSizes 数据 (文件的头部数据)
            std::vector<unsigned char> unpredSizes(totalBlocks);
            std::memcpy(unpredSizes.data(), allUnpredDataRead.data(), totalBlocks);

            // 分配 bitCounts 和 compSizes
            std::vector<unsigned char> bitCounts(totalBlocks);
            std::vector<unsigned char> compSizes(totalBlocks);

            // 将 allCompData 中的前 totalBlocks 个字节分配给 bitCounts
            std::memcpy(bitCounts.data(), allCompDataRead.data(), totalBlocks);

            // 将 allCompData 中的接下来的 totalBlocks 个字节分配给 compSizes
            std::memcpy(compSizes.data(), allCompDataRead.data() + totalBlocks, totalBlocks);

            const size_t signBytesPerBlock = (blockSize + 7) / 8;

            auto endReadOriginalBlocks = std::chrono::high_resolution_clock::now(); 


            std::chrono::duration<double> readOriginalBlocks = endReadOriginalBlocks - beginReadOriginalBlocks;
            std::cerr << "[Rank " << rank << "[Time6]: read all compressed small blocks time: " << readOriginalBlocks.count() << " seconds" << std::endl;



            // === 提前准备所有偏移 ===
            std::vector<size_t> unpredOffsets(totalBlocks , 0);
            std::vector<size_t> compOffsets(totalBlocks , 0);

            //先记录所有偏移，后续不用每次累加
            for (size_t i = 1; i < totalBlocks; ++i) {
                unpredOffsets[i] = unpredOffsets[i - 1] + unpredSizes[i - 1];
                compOffsets[i] = compOffsets[i - 1] + compSizes[i - 1];
            }

            // === 循环外预分配缓冲区（复用）===
            std::vector<double> unpredData;
            std::vector<unsigned char> compData;
            std::vector<unsigned char> signBits(signBytesPerBlock);
            //std::vector<int> quant_inds(blockSize);
            //std::vector<double> decompressedBlock(blockSize);


            /*std::vector<std::vector<double>> allUnpredData;
            std::vector<std::vector<unsigned char>> allCompData;
            std::vector<std::vector<unsigned char>> allSignBits;*/


            for (size_t bid : blockIds) {

                //auto t1 = std::chrono::high_resolution_clock::now();
                /*// 读取 unpredData
                size_t unpredOffset = totalBlocks + sizeof(double) * std::accumulate(unpredSizes.begin(), unpredSizes.begin() + bid, size_t(0));
                unsigned char unpredSize = unpredSizes[bid];
                std::vector<double> unpredData(unpredSize);

                if (unpredSize > 0) {
                    unpredStream.seekg(unpredOffset, std::ios::beg);
                    unpredStream.read(reinterpret_cast<char*>(unpredData.data()), sizeof(double) * unpredSize);
                }
                allUnpredData.push_back(std::move(unpredData));

                // 读取压缩数据 compData
                unsigned char bitCount = bitCounts[bid];
                unsigned char dataSize = compSizes[bid];
                size_t compOffset = 2 * totalBlocks + std::accumulate(compSizes.begin(), compSizes.begin() + bid, size_t(0));
                std::vector<unsigned char> compData(dataSize);
                compStream.seekg(compOffset, std::ios::beg);
                compStream.read(reinterpret_cast<char*>(compData.data()), dataSize);
                allCompData.push_back(std::move(compData));

                // 读取符号位 signBits
                size_t signOffset = bid * signBytesPerBlock;
                std::vector<unsigned char> signBits(signBytesPerBlock);
                signStream.seekg(signOffset, std::ios::beg);
                signStream.read(reinterpret_cast<char*>(signBits.data()), signBytesPerBlock);
                allSignBits.push_back(std::move(signBits));*/



                // Unpred
                unsigned char unpredSize = unpredSizes[bid];
                size_t unpredOffset = totalBlocks + sizeof(double) * unpredOffsets[bid];
                unpredData.resize(unpredSize);
                std::memcpy(unpredData.data(), allUnpredDataRead.data() + unpredOffset, unpredSize * sizeof(double));
             

                // Compressed
                unsigned char bitCount = bitCounts[bid];
                unsigned char dataSize = compSizes[bid];
                size_t compOffset = 2 * totalBlocks + compOffsets[bid];
                compData.resize(dataSize);
                std::memcpy(compData.data(), allCompDataRead.data() + compOffset, dataSize);
           
              

                // Sign
                size_t signOffset = signBytesPerBlock * bid;
                std::memcpy(signBits.data(), allSignDataRead.data() + signOffset, signBytesPerBlock);

                //auto t2 = std::chrono::high_resolution_clock::now();
                //memory_access_time += std::chrono::duration<double>(t2 - t1).count();

              
           

            /*}

            auto endReadOriginalBlocks = std::chrono::high_resolution_clock::now(); 


            std::chrono::duration<double> readOriginalBlocks = endReadOriginalBlocks - beginReadOriginalBlocks;
            std::cout << "[Time6]: read all compressed small blocks time: " << readOriginalBlocks.count() << " seconds" << std::endl;



            for (size_t i = 0; i < blockIds.size(); ++i) {
                size_t bid = blockIds[i];
                unsigned char bitCount = bitCounts[bid];
                unsigned char unpredSize = unpredSizes[bid];


                const auto& unpredData = allUnpredData[i];
                const auto& compData = allCompData[i];
                const auto& signBits = allSignBits[i];*/

                

                // === 解码 bit-packed 数据 ===
                //包括恢复符号
                int radius = 512;
                std::vector<int> quant_inds(blockSize);
                /*size_t bitPos = 0, dataIdx = 0;

                for (size_t i = 0; i < blockSize; ++i) {
                    unsigned int val = 0, bitsRead = 0;
                    while (bitsRead < bitCount && dataIdx < compData.size()) {
                        
                        unsigned int available = std::min<unsigned int>(bitCount - bitsRead, 8u - static_cast<unsigned int>(bitPos % 8));

                        //std::cout << "available: " << available<<  std::endl;

                        val |= ((compData[dataIdx] >> (bitPos % 8)) & ((1 << available) - 1)) << bitsRead;
                        bitsRead += available;
                        bitPos += available;
                        if (bitPos % 8 == 0) ++dataIdx;
                    }
                    

                    //signBits是对应的块的全部sign
                    int sign = (signBits[i / 8] >> (i % 8)) & 1;
                    quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);

                    
                    quant_inds[i] += radius;
                }*/

                // for test
                /*std::cout << "quant_inds: ";
                for (const auto& q : quant_inds) {
                    std::cout << q << " ";
                }
                std::cout << std::endl;*/


                size_t bitPos = 0;

                for (size_t i = 0; i < blockSize; ++i) {
                    unsigned int val = 0, bitsRead = 0;

                    while (bitsRead < bitCount && (bitPos / 8) < compData.size()) {
                        size_t byteIndex = bitPos / 8;
                        size_t bitOffset = bitPos % 8;

                        size_t available = std::min(static_cast<size_t>(bitCount - bitsRead), 8u - bitOffset);


                        val |= ((compData[byteIndex] >> bitOffset) & ((1 << available) - 1)) << bitsRead;

                        bitsRead += available;
                        bitPos += available;
                    }

                    int sign = (signBits[i / 8] >> (i % 8)) & 1;
                    quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);
                    quant_inds[i] += radius;
                }



                //恢复metadataBuffer
                std::vector<SZ3::uchar> metadataBuffer;
                metadataBuffer.push_back(0b00000010);  // 标志位

                // 写 error_bound (double)
                metadataBuffer.insert(metadataBuffer.end(),
                    reinterpret_cast<SZ3::uchar*>(&error_bound),
                    reinterpret_cast<SZ3::uchar*>(&error_bound) + sizeof(double));

                // 写 radius (int)
                int correctRadius = radius;
                metadataBuffer.insert(metadataBuffer.end(),
                    reinterpret_cast<SZ3::uchar*>(&correctRadius),
                    reinterpret_cast<SZ3::uchar*>(&correctRadius) + sizeof(int));

                // 写 unpredSize (size_t)
                size_t correctUnpredSize = static_cast<size_t>(unpredSize);
                metadataBuffer.insert(metadataBuffer.end(),
                    reinterpret_cast<SZ3::uchar*>(&correctUnpredSize),
                    reinterpret_cast<SZ3::uchar*>(&correctUnpredSize) + sizeof(size_t));

                // 写 unpredData (double[])
                metadataBuffer.insert(metadataBuffer.end(),
                    reinterpret_cast<const SZ3::uchar*>(unpredData.data()),
                    reinterpret_cast<const SZ3::uchar*>(unpredData.data() + unpredSize));

            

                // === 反量化 ===


                // 配置 SZ3 参数
                SZ3::Config conf(blockSize);
                conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
                conf.lorenzo = true;
                conf.regression = false;
                conf.errorBoundMode = SZ3::EB_ABS;
                conf.absErrorBound = error_bound;
                conf.quantbinCnt = 1024;

                auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
                auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
                auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);

       
                std::vector<double> decompressedBlock(blockSize); 
                
                const SZ3::uchar* metaPtr = metadataBuffer.data();
                size_t metaSize = metadataBuffer.size();


             
                size_t test_unpred_size = *reinterpret_cast<const size_t *>(metaPtr + 1 + sizeof(double) + sizeof(int));
                /*std::cout << "Extracted unpred_size = " << test_unpred_size << std::endl;
                std::cout << "metadataBuffer raw bytes: ";
                for (size_t i = 0; i < metadataBuffer.size(); ++i) {
                    std::cout << static_cast<int>(metadataBuffer[i]) << " ";
                }
                std::cout << std::endl;*/


                decompose.load(metaPtr, metaSize);  // 传引用，函数里可以正确移动



                //decompose.load(metadataBuffer.data(), metadataBuffer.size()); // 加载元数据
                decompose.decompress(conf, quant_inds, decompressedBlock.data()); // 解压缩数据

                decompressedOriginalresult.push_back(std::move(decompressedBlock));

            }

            auto endDecompress = std::chrono::high_resolution_clock::now(); 


            std::chrono::duration<double> decompressSmallBlocks = endDecompress - endReadOriginalBlocks;
            std::cerr << "[Rank " << rank << "[Time7]:  decompress small blocks time: " << decompressSmallBlocks.count() << " seconds" << std::endl;

            //std::cout << "[Time_mem_access]: Memory access time: " << memory_access_time << " seconds" << std::endl;


        return decompressedOriginalresult;
    }



    //blockSize每小块数据个数大小，totalBlocks全部原数据小块大小
    std::vector<std::vector<double>> batchDecompressBlocksSeek(const std::string& subDir,
        const std::vector<size_t>& blockIds,
        size_t blockSize,
        size_t totalBlocks,
        double error_bound) {

            std::cout << "seek: "  << std::endl;

            auto beginReadOriginalBlocks = std::chrono::high_resolution_clock::now(); 


            std::vector<std::vector<double>> decompressedOriginalresult;

            std::string unpredDataFileName = subDir + "fixed_unpredData.bin";
            std::string compressedFileName = subDir + "fixed_compressed_data.bin";
            std::string signFileName = subDir + "fixed_sign_data.bin";

            //std::cout << "Sign file path: " << signFileName << std::endl;

            std::ifstream unpredStream(unpredDataFileName, std::ios::binary);
            std::ifstream compStream(compressedFileName, std::ios::binary);
            std::ifstream signStream(signFileName, std::ios::binary);


            if (!unpredStream || !compStream || !signStream) {
            throw std::runtime_error("Failed to open one or more input files.");
            }

            //读取头文件，方便后续解压的时候access 

            // --- 读取 unpredSizes ---
           
            std::vector<unsigned char> unpredSizes(totalBlocks);
            unpredStream.read(reinterpret_cast<char*>(unpredSizes.data()), totalBlocks);
           

            // --- 读取压缩数据头部 (bitCounts 和 compSizes) ---
            std::vector<unsigned char> bitCounts(totalBlocks);
            std::vector<unsigned char> compSizes(totalBlocks);
            compStream.read(reinterpret_cast<char*>(bitCounts.data()), totalBlocks);
            compStream.read(reinterpret_cast<char*>(compSizes.data()), totalBlocks);

           
            const size_t signBytesPerBlock = (blockSize + 7) / 8;



            std::vector<std::vector<double>> allUnpredData;
            std::vector<std::vector<unsigned char>> allCompData;
            std::vector<std::vector<unsigned char>> allSignBits;


            for (size_t bid : blockIds) {
                // 读取 unpredData
                size_t unpredOffset = totalBlocks + sizeof(double) * std::accumulate(unpredSizes.begin(), unpredSizes.begin() + bid, size_t(0));
                unsigned char unpredSize = unpredSizes[bid];
                std::vector<double> unpredData(unpredSize);

                if (unpredSize > 0) {
                    unpredStream.seekg(unpredOffset, std::ios::beg);
                    unpredStream.read(reinterpret_cast<char*>(unpredData.data()), sizeof(double) * unpredSize);
                }
                allUnpredData.push_back(std::move(unpredData));

                // 读取压缩数据 compData
                unsigned char bitCount = bitCounts[bid];
                unsigned char dataSize = compSizes[bid];
                size_t compOffset = 2 * totalBlocks + std::accumulate(compSizes.begin(), compSizes.begin() + bid, size_t(0));
                std::vector<unsigned char> compData(dataSize);
                compStream.seekg(compOffset, std::ios::beg);
                compStream.read(reinterpret_cast<char*>(compData.data()), dataSize);
                allCompData.push_back(std::move(compData));

                // 读取符号位 signBits
                size_t signOffset = bid * signBytesPerBlock;
                std::vector<unsigned char> signBits(signBytesPerBlock);
                signStream.seekg(signOffset, std::ios::beg);
                signStream.read(reinterpret_cast<char*>(signBits.data()), signBytesPerBlock);
                allSignBits.push_back(std::move(signBits));



            }

            auto endReadOriginalBlocks = std::chrono::high_resolution_clock::now(); 


            std::chrono::duration<double> readOriginalBlocks = endReadOriginalBlocks - beginReadOriginalBlocks;
            std::cout << "[Time6]: read all compressed small blocks time: " << readOriginalBlocks.count() << " seconds" << std::endl;



            for (size_t i = 0; i < blockIds.size(); ++i) {
                size_t bid = blockIds[i];
                unsigned char bitCount = bitCounts[bid];
                unsigned char unpredSize = unpredSizes[bid];


                const auto& unpredData = allUnpredData[i];
                const auto& compData = allCompData[i];
                const auto& signBits = allSignBits[i];

                

                // === 解码 bit-packed 数据 ===
                //包括恢复符号
                int radius = 512;
                std::vector<int> quant_inds(blockSize);
                /*size_t bitPos = 0, dataIdx = 0;

                for (size_t i = 0; i < blockSize; ++i) {
                    unsigned int val = 0, bitsRead = 0;
                    while (bitsRead < bitCount && dataIdx < compData.size()) {
                        
                        unsigned int available = std::min<unsigned int>(bitCount - bitsRead, 8u - static_cast<unsigned int>(bitPos % 8));

                        //std::cout << "available: " << available<<  std::endl;

                        val |= ((compData[dataIdx] >> (bitPos % 8)) & ((1 << available) - 1)) << bitsRead;
                        bitsRead += available;
                        bitPos += available;
                        if (bitPos % 8 == 0) ++dataIdx;
                    }
                    

                    //signBits是对应的块的全部sign
                    int sign = (signBits[i / 8] >> (i % 8)) & 1;
                    quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);

                    
                    quant_inds[i] += radius;
                }*/

                // for test
                /*std::cout << "quant_inds: ";
                for (const auto& q : quant_inds) {
                    std::cout << q << " ";
                }
                std::cout << std::endl;*/

                size_t bitPos = 0;

                for (size_t i = 0; i < blockSize; ++i) {
                    unsigned int val = 0, bitsRead = 0;

                    while (bitsRead < bitCount && (bitPos / 8) < compData.size()) {
                        size_t byteIndex = bitPos / 8;
                        size_t bitOffset = bitPos % 8;

                        size_t available = std::min(static_cast<size_t>(bitCount - bitsRead), 8u - bitOffset);


                        val |= ((compData[byteIndex] >> bitOffset) & ((1 << available) - 1)) << bitsRead;

                        bitsRead += available;
                        bitPos += available;
                    }

                    int sign = (signBits[i / 8] >> (i % 8)) & 1;
                    quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);
                    quant_inds[i] += radius;
                }


                //恢复metadataBuffer
                std::vector<SZ3::uchar> metadataBuffer;
                metadataBuffer.push_back(0b00000010);  // 标志位

                // 写 error_bound (double)
                metadataBuffer.insert(metadataBuffer.end(),
                    reinterpret_cast<SZ3::uchar*>(&error_bound),
                    reinterpret_cast<SZ3::uchar*>(&error_bound) + sizeof(double));

                // 写 radius (int)
                int correctRadius = radius;
                metadataBuffer.insert(metadataBuffer.end(),
                    reinterpret_cast<SZ3::uchar*>(&correctRadius),
                    reinterpret_cast<SZ3::uchar*>(&correctRadius) + sizeof(int));

                // 写 unpredSize (size_t)
                size_t correctUnpredSize = static_cast<size_t>(unpredSize);
                metadataBuffer.insert(metadataBuffer.end(),
                    reinterpret_cast<SZ3::uchar*>(&correctUnpredSize),
                    reinterpret_cast<SZ3::uchar*>(&correctUnpredSize) + sizeof(size_t));

                // 写 unpredData (double[])
                metadataBuffer.insert(metadataBuffer.end(),
                    reinterpret_cast<const SZ3::uchar*>(unpredData.data()),
                    reinterpret_cast<const SZ3::uchar*>(unpredData.data() + unpredSize));

            

                // === 反量化 ===


                // 配置 SZ3 参数
                SZ3::Config conf(blockSize);
                conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
                conf.lorenzo = true;
                conf.regression = false;
                conf.errorBoundMode = SZ3::EB_ABS;
                conf.absErrorBound = error_bound;
                conf.quantbinCnt = 1024;

                auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
                auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
                auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);

       
                std::vector<double> decompressedBlock(blockSize); 
                
                const SZ3::uchar* metaPtr = metadataBuffer.data();
                size_t metaSize = metadataBuffer.size();


             
                size_t test_unpred_size = *reinterpret_cast<const size_t *>(metaPtr + 1 + sizeof(double) + sizeof(int));
                /*std::cout << "Extracted unpred_size = " << test_unpred_size << std::endl;
                std::cout << "metadataBuffer raw bytes: ";
                for (size_t i = 0; i < metadataBuffer.size(); ++i) {
                    std::cout << static_cast<int>(metadataBuffer[i]) << " ";
                }
                std::cout << std::endl;*/


                decompose.load(metaPtr, metaSize);  // 传引用，函数里可以正确移动



                //decompose.load(metadataBuffer.data(), metadataBuffer.size()); // 加载元数据
                decompose.decompress(conf, quant_inds, decompressedBlock.data()); // 解压缩数据

                decompressedOriginalresult.push_back(std::move(decompressedBlock));

            }

            auto endDecompress = std::chrono::high_resolution_clock::now(); 


            std::chrono::duration<double> decompressSmallBlocks = endDecompress - endReadOriginalBlocks;
            std::cout << "[Time7]:  decompress small blocks time: " << decompressSmallBlocks.count() << " seconds" << std::endl;



        return decompressedOriginalresult;
    }

    std::vector<std::vector<double>> batchDecompressBlocksAllRead(const std::string& subDir,
        const std::vector<size_t>& blockIds,
        size_t blockSize,
        size_t totalBlocks,
        double error_bound) {
    std::cout << "all read: "  << std::endl;
    auto start_time_read = std::chrono::high_resolution_clock::now();      
    std::vector<std::vector<double>> decompressedOriginalresult;


    std::string unpredDataFileName = subDir + "fixed_unpredData.bin";
    std::string compressedFileName = subDir + "fixed_compressed_data.bin";
    std::string signFileName = subDir + "fixed_sign_data.bin";

    // 打开文件流
    std::ifstream unpredStream(unpredDataFileName, std::ios::binary);
    std::ifstream compStream(compressedFileName, std::ios::binary);
    std::ifstream signStream(signFileName, std::ios::binary);

    if (!unpredStream || !compStream || !signStream) {
        throw std::runtime_error("Failed to open one or more input files.");
    }

    // 获取文件大小
    unpredStream.seekg(0, std::ios::end);
    size_t unpredFileSize = unpredStream.tellg();
    unpredStream.seekg(0, std::ios::beg);

    compStream.seekg(0, std::ios::end);
    size_t compFileSize = compStream.tellg();
    compStream.seekg(0, std::ios::beg);

    signStream.seekg(0, std::ios::end);
    size_t signFileSize = signStream.tellg();
    signStream.seekg(0, std::ios::beg);

    // 分配足够大的内存
    std::vector<unsigned char> allUnpredData(unpredFileSize);
    std::vector<unsigned char> allCompData(compFileSize);
    std::vector<unsigned char> allSignData(signFileSize);

    // 读取文件数据到内存
    unpredStream.read(reinterpret_cast<char*>(allUnpredData.data()), unpredFileSize);
    compStream.read(reinterpret_cast<char*>(allCompData.data()), compFileSize);
    signStream.read(reinterpret_cast<char*>(allSignData.data()), signFileSize);


    // 读取 unpredSizes 数据 (文件的头部数据)
    std::vector<unsigned char> unpredSizes(totalBlocks);
    std::memcpy(unpredSizes.data(), allUnpredData.data(), totalBlocks);

    // 分配 bitCounts 和 compSizes
    std::vector<unsigned char> bitCounts(totalBlocks);
    std::vector<unsigned char> compSizes(totalBlocks);

    // 将 allCompData 中的前 totalBlocks 个字节分配给 bitCounts
    std::memcpy(bitCounts.data(), allCompData.data(), totalBlocks);

    // 将 allCompData 中的接下来的 totalBlocks 个字节分配给 compSizes
    std::memcpy(compSizes.data(), allCompData.data() + totalBlocks, totalBlocks);

    const size_t signBytesPerBlock = (blockSize + 7) / 8;

    auto begin_decompress = std::chrono::high_resolution_clock::now(); 


    // === 提前准备所有偏移 ===
    //偏移的字节数
    std::vector<size_t> unpredOffsets(totalBlocks , 0);
    std::vector<size_t> compOffsets(totalBlocks , 0);

    //先记录所有偏移，后续不用每次累加
    for (size_t i = 1; i < totalBlocks; ++i) {
        unpredOffsets[i] = unpredOffsets[i - 1] + unpredSizes[i - 1];
        compOffsets[i] = compOffsets[i - 1] + compSizes[i - 1];
    }

    // === 循环外预分配缓冲区（复用）===
    std::vector<double> unpredData;
    std::vector<unsigned char> compData;
    std::vector<unsigned char> signBits(signBytesPerBlock);
    std::vector<int> quant_inds(blockSize);
    std::vector<double> decompressedBlock(blockSize);


    for (size_t bid : blockIds) {
       
        // Unpred
        unsigned char unpredSize = unpredSizes[bid];
        size_t unpredOffset = totalBlocks + sizeof(double) * unpredOffsets[bid];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredData.data() + unpredOffset, unpredSize * sizeof(double));

        // Compressed
        unsigned char bitCount = bitCounts[bid];
        unsigned char dataSize = compSizes[bid];
        size_t compOffset = 2 * totalBlocks + compOffsets[bid];
        compData.resize(dataSize);
        std::memcpy(compData.data(), allCompData.data() + compOffset, dataSize);

        // Sign
        size_t signOffset = signBytesPerBlock * bid;
        std::memcpy(signBits.data(), allSignData.data() + signOffset, signBytesPerBlock);

        
       // 解码数据
       int radius = 512;
       std::vector<int> quant_inds(blockSize);
      

       size_t bitPos = 0;

        for (size_t i = 0; i < blockSize; ++i) {
            unsigned int val = 0, bitsRead = 0;

            while (bitsRead < bitCount && (bitPos / 8) < compData.size()) {
                size_t byteIndex = bitPos / 8;
                size_t bitOffset = bitPos % 8;

                size_t available = std::min(static_cast<size_t>(bitCount - bitsRead), 8u - bitOffset);


                val |= ((compData[byteIndex] >> bitOffset) & ((1 << available) - 1)) << bitsRead;

                bitsRead += available;
                bitPos += available;
            }

            int sign = (signBits[i / 8] >> (i % 8)) & 1;
            quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);
            quant_inds[i] += radius;
        }



        
        // === Metadata + decompression ===
        std::vector<SZ3::uchar> metadataBuffer;
        metadataBuffer.push_back(0b00000010);
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&error_bound),
            reinterpret_cast<SZ3::uchar*>(&error_bound) + sizeof(double));
       
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&radius),
            reinterpret_cast<SZ3::uchar*>(&radius) + sizeof(int));
        size_t correctUnpredSize = static_cast<size_t>(unpredSize);
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize) + sizeof(size_t));
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data()),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data() + unpredSize));

        // 解压缩配置
        SZ3::Config conf(blockSize);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;

        auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);

        const SZ3::uchar* metaPtr = metadataBuffer.data();
        size_t metaSize = metadataBuffer.size();

        decompose.load(metaPtr, metaSize);
        decompose.decompress(conf, quant_inds, decompressedBlock.data());
        decompressedOriginalresult.push_back(decompressedBlock);  
    }

    

    auto end_decompress = std::chrono::high_resolution_clock::now(); 
    

    std::chrono::duration<double> readTime = begin_decompress - start_time_read;
    
     std::chrono::duration<double> compressTime = end_decompress - begin_decompress;
    
     std::cout << "[Time6]: read all compressed small blocks time: "<< readTime.count() << " seconds" << std::endl;
    
     std::cout << "[Time7]:  decompress small blocks time: " << compressTime.count() << " seconds" << std::endl;

    return decompressedOriginalresult;
}



// 自包含的通用解压函数，无需额外依赖

// 内联工具函数：计算分块配置参数
struct BlockConfig {
    size_t blockSize;
    size_t unpredSizeBits;
    size_t dataSizeBytes;
    size_t signBytesPerBlock;
    
    BlockConfig(size_t bs) : blockSize(bs) {
        // 根据分块大小计算参数
        if (bs == 64) {           // 4³
            unpredSizeBits = 6;
            dataSizeBytes = 1;
            signBytesPerBlock = 8;
        } else if (bs == 4096) {  // 16³
            unpredSizeBits = 12;
            dataSizeBytes = 2;
            signBytesPerBlock = 512;
        } else if (bs == 262144) { // 64³
            unpredSizeBits = 18;
            dataSizeBytes = 3;
            signBytesPerBlock = 32768;
        } else if (bs == 16777216) { // 256³
            unpredSizeBits = 24;
            dataSizeBytes = 4;
            signBytesPerBlock = 2097152;
        } else if (bs == 1073741824) { // 1024³
            unpredSizeBits = 30;
            dataSizeBytes = 4;
            signBytesPerBlock = 134217728;
        } else {
            throw std::runtime_error("Unsupported block size");
        }
    }
};

// 内联工具函数：解包位数据
void unpackBitsInline(const std::vector<uint8_t>& input, std::vector<size_t>& output, 
                     size_t bitsPerValue, size_t numValues) {
    output.clear();
    output.reserve(numValues);
    
    if (input.empty() || numValues == 0) return;
    
    size_t byteIndex = 0;
    size_t bitIndex = 0;
    uint64_t mask = (1ULL << bitsPerValue) - 1;
    
    for (size_t i = 0; i < numValues; i++) {
        uint64_t value = 0;
        size_t bitsRead = 0;
        
        while (bitsRead < bitsPerValue) {
            if (byteIndex >= input.size()) {
                throw std::runtime_error("Insufficient data for unpacking");
            }
            
            size_t bitsToRead = std::min(bitsPerValue - bitsRead, 8 - bitIndex);
            uint8_t byteMask = ((1 << bitsToRead) - 1) << bitIndex;
            uint8_t bits = (input[byteIndex] & byteMask) >> bitIndex;
            
            value |= (static_cast<uint64_t>(bits) << bitsRead);
            bitsRead += bitsToRead;
            bitIndex += bitsToRead;
            
            if (bitIndex >= 8) {
                bitIndex = 0;
                byteIndex++;
            }
        }
        
        output.push_back(static_cast<size_t>(value & mask));
    }
}

// 内联工具函数：读取多字节数值
uint64_t readValueInline(const uint8_t* data, size_t numBytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < numBytes; i++) {
        value |= (static_cast<uint64_t>(data[i]) << (i * 8));
    }
    return value;
}

// 通用解压函数 - 选择性读取版本
std::vector<std::vector<double>> universalBatchDecompressBlocks(
    const std::string& subDir,
    const std::vector<size_t>& blockIds,
    size_t blockSize,
    size_t totalBlocks,
    double error_bound) {
    
    int rank = 0;
    std::cout << "universal read: " << std::endl;
    
    // 获取块配置
    BlockConfig config(blockSize);
    
    auto beginReadOriginalBlocks = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> decompressedOriginalresult;

    std::cout << "subDir = " << subDir << "\n";
    std::cout << "universal read1: " << std::endl;


    
    // 文件名
    std::string unpredDataFileName = subDir + "universal_unpredData.bin";
    std::string compressedFileName = subDir + "universal_compressed_data.bin";
    std::string signFileName = subDir + "universal_sign_data.bin";
    
    std::ifstream unpredStream(unpredDataFileName, std::ios::binary);
    std::ifstream compStream(compressedFileName, std::ios::binary);
    std::ifstream signStream(signFileName, std::ios::binary);
    
    if (!unpredStream || !compStream || !signStream) {
        throw std::runtime_error("Failed to open one or more input files.");
    }

    std::cout << "universal read2: " << std::endl;
    
    // 获取文件大小
    unpredStream.seekg(0, std::ios::end);
    size_t unpredFileSize = unpredStream.tellg();
    unpredStream.seekg(0, std::ios::beg);
    
    compStream.seekg(0, std::ios::end);
    size_t compFileSize = compStream.tellg();
    compStream.seekg(0, std::ios::beg);
    
    signStream.seekg(0, std::ios::end);
    size_t signFileSize = signStream.tellg();
    signStream.seekg(0, std::ios::beg);

    std::cout << "universal read3: " << std::endl;
    
    // 读取所有文件到内存
    std::vector<uint8_t> allUnpredDataRead(unpredFileSize);
    std::vector<uint8_t> allCompDataRead(compFileSize);
    std::vector<uint8_t> allSignDataRead(signFileSize);

    
    unpredStream.read(reinterpret_cast<char*>(allUnpredDataRead.data()), unpredFileSize);
    compStream.read(reinterpret_cast<char*>(allCompDataRead.data()), compFileSize);
    signStream.read(reinterpret_cast<char*>(allSignDataRead.data()), signFileSize);

    std::cout << "universal read4: " << std::endl;
    
    // 解析unpredSizes
    size_t unpredSizeBytes = (totalBlocks * config.unpredSizeBits + 7) / 8;
    std::vector<uint8_t> compressedUnpredSizes(allUnpredDataRead.begin(),
                                              allUnpredDataRead.begin() + unpredSizeBytes);
    std::vector<size_t> unpredSizes;
    unpackBitsInline(compressedUnpredSizes, unpredSizes, config.unpredSizeBits, totalBlocks);
    
    // 读取bitCounts
    std::vector<uint8_t> bitCounts(totalBlocks);
    std::memcpy(bitCounts.data(), allCompDataRead.data(), totalBlocks);
    
    // 读取compSizes
    std::vector<size_t> compSizes(totalBlocks);
    for (size_t i = 0; i < totalBlocks; i++) {
        size_t offset = totalBlocks + i * config.dataSizeBytes;
        compSizes[i] = static_cast<size_t>(readValueInline(
            allCompDataRead.data() + offset, config.dataSizeBytes));
    }
    
    auto endReadOriginalBlocks = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> readOriginalBlocks = endReadOriginalBlocks - beginReadOriginalBlocks;
    std::cerr << "[Rank " << rank << "][Time6]: read all compressed small blocks time: " 
              << readOriginalBlocks.count() << " seconds" << std::endl;
    
    // 计算偏移量
    std::vector<size_t> unpredOffsets(totalBlocks, 0);
    std::vector<size_t> compOffsets(totalBlocks, 0);
    
    for (size_t i = 1; i < totalBlocks; ++i) {
        unpredOffsets[i] = unpredOffsets[i - 1] + unpredSizes[i - 1];
        compOffsets[i] = compOffsets[i - 1] + compSizes[i - 1];
    }
    
    // 预分配缓冲区
    std::vector<double> unpredData;
    std::vector<uint8_t> compData;
    std::vector<uint8_t> signBits(config.signBytesPerBlock);
    
    for (size_t bid : blockIds) {
        // 读取unpredData
        size_t unpredSize = unpredSizes[bid];
        size_t unpredOffset = unpredSizeBytes + sizeof(double) * unpredOffsets[bid];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredDataRead.data() + unpredOffset,
                   unpredSize * sizeof(double));
        
        // 读取压缩数据
        uint8_t bitCount = bitCounts[bid];
        size_t dataSize = compSizes[bid];
        size_t compOffset = totalBlocks + totalBlocks * config.dataSizeBytes + compOffsets[bid];
        compData.resize(dataSize);
        std::memcpy(compData.data(), allCompDataRead.data() + compOffset, dataSize);
        
        // 读取符号数据
        size_t signOffset = config.signBytesPerBlock * bid;
        std::memcpy(signBits.data(), allSignDataRead.data() + signOffset, config.signBytesPerBlock);
        
        // 解码量化索引
        int radius = 512;
        std::vector<int> quant_inds(blockSize);
        size_t bitPos = 0;
        
        for (size_t i = 0; i < blockSize; ++i) {
            uint32_t val = 0, bitsRead = 0;
            
            while (bitsRead < bitCount && (bitPos / 8) < compData.size()) {
                size_t byteIndex = bitPos / 8;
                size_t bitOffset = bitPos % 8;
                size_t available = std::min(static_cast<size_t>(bitCount - bitsRead), 8u - bitOffset);
                
                val |= ((compData[byteIndex] >> bitOffset) & ((1 << available) - 1)) << bitsRead;
                bitsRead += available;
                bitPos += available;
            }
            
            int sign = (signBits[i / 8] >> (i % 8)) & 1;
            quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);
            quant_inds[i] += radius;
        }
        
        // 重建metadata
        std::vector<SZ3::uchar> metadataBuffer;
        metadataBuffer.push_back(0b00000010);
        
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&error_bound),
            reinterpret_cast<SZ3::uchar*>(&error_bound) + sizeof(double));
        
        int correctRadius = radius;
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&correctRadius),
            reinterpret_cast<SZ3::uchar*>(&correctRadius) + sizeof(int));
        
        size_t correctUnpredSize = static_cast<size_t>(unpredSize);
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize) + sizeof(size_t));
        
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data()),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data() + unpredSize));
        
        // SZ3解压配置
        SZ3::Config conf(blockSize);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;
        
        auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);
        
        std::vector<double> decompressedBlock(blockSize);
        const SZ3::uchar* metaPtr = metadataBuffer.data();
        size_t metaSize = metadataBuffer.size();
        
        decompose.load(metaPtr, metaSize);
        decompose.decompress(conf, quant_inds, decompressedBlock.data());
        
        decompressedOriginalresult.push_back(std::move(decompressedBlock));
    }
    
    auto endDecompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompressSmallBlocks = endDecompress - endReadOriginalBlocks;
    std::cerr << "[Rank " << rank << "][Time7]: decompress small blocks time: " 
              << decompressSmallBlocks.count() << " seconds" << std::endl;
    
    return decompressedOriginalresult;
}

// 通用解压函数 - 全量读取版本
std::vector<std::vector<double>> universalBatchDecompressBlocksAllRead(
    const std::string& subDir,
    const std::vector<size_t>& blockIds,
    size_t blockSize,
    size_t totalBlocks,
    double error_bound) {
    
    std::cout << "universal all read: " << std::endl;
    auto start_time_read = std::chrono::high_resolution_clock::now();
    
    // 获取块配置
    BlockConfig config(blockSize);
    
    std::vector<std::vector<double>> decompressedOriginalresult;
    
    // 文件名
    std::string unpredDataFileName = subDir + "universal_unpredData.bin";
    std::string compressedFileName = subDir + "universal_compressed_data.bin";
    std::string signFileName = subDir + "universal_sign_data.bin";
    
    std::ifstream unpredStream(unpredDataFileName, std::ios::binary);
    std::ifstream compStream(compressedFileName, std::ios::binary);
    std::ifstream signStream(signFileName, std::ios::binary);
    
    if (!unpredStream || !compStream || !signStream) {
        throw std::runtime_error("Failed to open one or more input files.");
    }
    
    // 获取文件大小
    unpredStream.seekg(0, std::ios::end);
    size_t unpredFileSize = unpredStream.tellg();
    unpredStream.seekg(0, std::ios::beg);
    
    compStream.seekg(0, std::ios::end);
    size_t compFileSize = compStream.tellg();
    compStream.seekg(0, std::ios::beg);
    
    signStream.seekg(0, std::ios::end);
    size_t signFileSize = signStream.tellg();
    signStream.seekg(0, std::ios::beg);
    
    // 读取所有文件到内存
    std::vector<uint8_t> allUnpredData(unpredFileSize);
    std::vector<uint8_t> allCompData(compFileSize);
    std::vector<uint8_t> allSignData(signFileSize);
    
    unpredStream.read(reinterpret_cast<char*>(allUnpredData.data()), unpredFileSize);
    compStream.read(reinterpret_cast<char*>(allCompData.data()), compFileSize);
    signStream.read(reinterpret_cast<char*>(allSignData.data()), signFileSize);
    
    // 解析unpredSizes
    size_t unpredSizeBytes = (totalBlocks * config.unpredSizeBits + 7) / 8;
    std::vector<uint8_t> compressedUnpredSizes(allUnpredData.begin(),
                                              allUnpredData.begin() + unpredSizeBytes);
    std::vector<size_t> unpredSizes;
    unpackBitsInline(compressedUnpredSizes, unpredSizes, config.unpredSizeBits, totalBlocks);
    
    // 读取bitCounts
    std::vector<uint8_t> bitCounts(totalBlocks);
    std::memcpy(bitCounts.data(), allCompData.data(), totalBlocks);
    
    // 读取compSizes
    std::vector<size_t> compSizes(totalBlocks);
    for (size_t i = 0; i < totalBlocks; i++) {
        size_t offset = totalBlocks + i * config.dataSizeBytes;
        compSizes[i] = static_cast<size_t>(readValueInline(
            allCompData.data() + offset, config.dataSizeBytes));
    }
    
    auto begin_decompress = std::chrono::high_resolution_clock::now();
    
    // 计算偏移量
    std::vector<size_t> unpredOffsets(totalBlocks, 0);
    std::vector<size_t> compOffsets(totalBlocks, 0);
    
    for (size_t i = 1; i < totalBlocks; ++i) {
        unpredOffsets[i] = unpredOffsets[i - 1] + unpredSizes[i - 1];
        compOffsets[i] = compOffsets[i - 1] + compSizes[i - 1];
    }
    
    // 预分配缓冲区
    std::vector<double> unpredData;
    std::vector<uint8_t> compData;
    std::vector<uint8_t> signBits(config.signBytesPerBlock);
    std::vector<int> quant_inds(blockSize);
    std::vector<double> decompressedBlock(blockSize);
    
    for (size_t bid : blockIds) {
        // 读取unpredData
        size_t unpredSize = unpredSizes[bid];
        size_t unpredOffset = unpredSizeBytes + sizeof(double) * unpredOffsets[bid];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredData.data() + unpredOffset,
                   unpredSize * sizeof(double));
        
        // 读取压缩数据
        uint8_t bitCount = bitCounts[bid];
        size_t dataSize = compSizes[bid];
        size_t compOffset = totalBlocks + totalBlocks * config.dataSizeBytes + compOffsets[bid];
        compData.resize(dataSize);
        std::memcpy(compData.data(), allCompData.data() + compOffset, dataSize);
        
        // 读取符号数据
        size_t signOffset = config.signBytesPerBlock * bid;
        std::memcpy(signBits.data(), allSignData.data() + signOffset, config.signBytesPerBlock);
        
        // 解码数据
        int radius = 512;
        size_t bitPos = 0;
        
        for (size_t i = 0; i < blockSize; ++i) {
            uint32_t val = 0, bitsRead = 0;
            
            while (bitsRead < bitCount && (bitPos / 8) < compData.size()) {
                size_t byteIndex = bitPos / 8;
                size_t bitOffset = bitPos % 8;
                size_t available = std::min(static_cast<size_t>(bitCount - bitsRead), 8u - bitOffset);
                
                val |= ((compData[byteIndex] >> bitOffset) & ((1 << available) - 1)) << bitsRead;
                bitsRead += available;
                bitPos += available;
            }
            
            int sign = (signBits[i / 8] >> (i % 8)) & 1;
            quant_inds[i] = sign ? -static_cast<int>(val) : static_cast<int>(val);
            quant_inds[i] += radius;
        }
        
        // 重建metadata
        std::vector<SZ3::uchar> metadataBuffer;
        metadataBuffer.push_back(0b00000010);
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&error_bound),
            reinterpret_cast<SZ3::uchar*>(&error_bound) + sizeof(double));
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&radius),
            reinterpret_cast<SZ3::uchar*>(&radius) + sizeof(int));
        size_t correctUnpredSize = static_cast<size_t>(unpredSize);
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize),
            reinterpret_cast<SZ3::uchar*>(&correctUnpredSize) + sizeof(size_t));
        metadataBuffer.insert(metadataBuffer.end(),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data()),
            reinterpret_cast<const SZ3::uchar*>(unpredData.data() + unpredSize));
        
        // SZ3解压配置
        SZ3::Config conf(blockSize);
        conf.cmprAlgo = SZ3::ALGO_LORENZO_REG;
        conf.lorenzo = true;
        conf.regression = false;
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound = error_bound;
        conf.quantbinCnt = 1024;
        
        auto predictor = SZ3::LorenzoPredictor<double, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<double>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<double, 1>(conf, quantizer);
        
        const SZ3::uchar* metaPtr = metadataBuffer.data();
        size_t metaSize = metadataBuffer.size();
        
        decompose.load(metaPtr, metaSize);
        decompose.decompress(conf, quant_inds, decompressedBlock.data());
        decompressedOriginalresult.push_back(decompressedBlock);
    }
    
    auto end_decompress = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> readTime = begin_decompress - start_time_read;
    std::chrono::duration<double> compressTime = end_decompress - begin_decompress;
    
    std::cout << "[Time6]: read all compressed small blocks time: " << readTime.count() << " seconds" << std::endl;
    std::cout << "[Time7]: decompress small blocks time: " << compressTime.count() << " seconds" << std::endl;
    
    return decompressedOriginalresult;
}


    std::vector<std::vector<double>> rangeFilterData(
        const std::vector<std::vector<double>>& blocks,
        double queryLow,
        double queryHigh) 
    {
        std::vector<std::vector<double>> filteredBlocks;
    
        for (const auto& block : blocks) {
            std::vector<double> filtered;
            for (double val : block) {
                if (val >= queryLow && val <= queryHigh) {
                    filtered.push_back(val);
                }
            }
            if (!filtered.empty()) {
                filteredBlocks.push_back(std::move(filtered));
            }
        }
    
        return filteredBlocks;
    }


    // ============================================================================
// 辅助函数1（double版）：获取邻居块指针
// 功能：根据方向偏移，从blockMap中查找邻居块的数据指针
// ============================================================================
inline const double* getNeighborBlockPointer(
    size_t blockId,
    int dx, int dy, int dz,
    const std::unordered_map<size_t, const double*>& blockMap,
    const std::vector<size_t>& blockCountPerDim)
{
    // 计算当前块的3D坐标
    size_t bz = blockId % blockCountPerDim[2];
    size_t remaining = blockId / blockCountPerDim[2];
    size_t by = remaining % blockCountPerDim[1];
    size_t bx = remaining / blockCountPerDim[1];    
    
    // 计算邻居块的3D坐标
    int nbr_x = static_cast<int>(bx) + dx;
    int nbr_y = static_cast<int>(by) + dy;
    int nbr_z = static_cast<int>(bz) + dz;
    
    // 边界检查：邻居是否在有效范围内
    if (nbr_x < 0 || nbr_x >= static_cast<int>(blockCountPerDim[0]) ||
        nbr_y < 0 || nbr_y >= static_cast<int>(blockCountPerDim[1]) ||
        nbr_z < 0 || nbr_z >= static_cast<int>(blockCountPerDim[2])) {
        return nullptr;
    }
    
    // 计算邻居块的全局ID（row-major顺序）
    size_t nbrId = static_cast<size_t>(nbr_x) * blockCountPerDim[1] * blockCountPerDim[2] + 
                   static_cast<size_t>(nbr_y) * blockCountPerDim[2] + 
                   static_cast<size_t>(nbr_z);
    
    // 从映射中查找邻居块数据
    auto it = blockMap.find(nbrId);
    return (it != blockMap.end()) ? it->second : nullptr;
}

// ============================================================================
// 辅助函数2（double版）：处理内部cube
// 功能：处理块内部的cube（8个顶点全部在当前块内）
// ============================================================================
inline void processInternalCube(
    const double* blockData,
    size_t local_x, size_t local_y, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    // 计算块内基础索引（row-major）
    size_t base = local_x + local_y * Bx + local_z * Bx * By;
    
    // ========== 8个顶点值 ==========
    std::array<double, 8> cubeValues = {{
        blockData[base],                      // 顶点0: (x, y, z)
        blockData[base + 1],                  // 顶点1: (x+1, y, z)
        blockData[base + 1 + Bx],             // 顶点2: (x+1, y+1, z)
        blockData[base + Bx],                 // 顶点3: (x, y+1, z)
        blockData[base + Bx * By],            // 顶点4: (x, y, z+1)
        blockData[base + 1 + Bx * By],        // 顶点5: (x+1, y, z+1)
        blockData[base + 1 + Bx + Bx * By],   // 顶点6: (x+1, y+1, z+1)
        blockData[base + Bx + Bx * By]        // 顶点7: (x, y+1, z+1)
    }};
    
    // 计算Marching Cubes的case ID
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;  // 完全在等值面内部或外部，无三角形
    }
    
    // ========== 8个顶点的全局坐标 ==========
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // ========== 计算梯度（法向量）==========
    std::array<std::array<double, 3>, 8> cubeGradients;
    for (int i = 0; i < 8; ++i) {
        size_t px = local_x + (i & 1);
        size_t py = local_y + ((i >> 1) & 1);
        size_t pz = local_z + ((i >> 2) & 1);
        
        // 钳位到块内有效范围
        size_t px_safe = std::min(px, Bx - 1);
        size_t py_safe = std::min(py, By - 1);
        size_t pz_safe = std::min(pz, Bx - 1); // 注意：这里沿用你原来的写法（Bz==Bx）
        
        std::array<double, 3> grad = {0.0, 0.0, 0.0};
        
        // X方向梯度
        if (px_safe == 0) {
            grad[0] = blockData[px_safe + py_safe*Bx + pz_safe*Bx*By] - 
                      blockData[(px_safe+1) + py_safe*Bx + pz_safe*Bx*By];
        } else if (px_safe == Bx - 1) {
            grad[0] = blockData[(px_safe-1) + py_safe*Bx + pz_safe*Bx*By] - 
                      blockData[px_safe + py_safe*Bx + pz_safe*Bx*By];
        } else {
            grad[0] = (blockData[(px_safe-1) + py_safe*Bx + pz_safe*Bx*By] - 
                       blockData[(px_safe+1) + py_safe*Bx + pz_safe*Bx*By]) / 2.0;
        }
        
        // Y方向梯度
        if (py_safe == 0) {
            grad[1] = blockData[px_safe + py_safe*Bx + pz_safe*Bx*By] - 
                      blockData[px_safe + (py_safe+1)*Bx + pz_safe*Bx*By];
        } else if (py_safe == By - 1) {
            grad[1] = blockData[px_safe + (py_safe-1)*Bx + pz_safe*Bx*By] - 
                      blockData[px_safe + py_safe*Bx + pz_safe*Bx*By];
        } else {
            grad[1] = (blockData[px_safe + (py_safe-1)*Bx + pz_safe*Bx*By] - 
                       blockData[px_safe + (py_safe+1)*Bx + pz_safe*Bx*By]) / 2.0;
        }
        
        // Z方向梯度
        if (pz_safe == 0) {
            grad[2] = blockData[px_safe + py_safe*Bx + pz_safe*Bx*By] - 
                      blockData[px_safe + py_safe*Bx + (pz_safe+1)*Bx*By];
        } else if (pz_safe == Bx - 1) {
            grad[2] = blockData[px_safe + py_safe*Bx + (pz_safe-1)*Bx*By] - 
                      blockData[px_safe + py_safe*Bx + pz_safe*Bx*By];
        } else {
            grad[2] = (blockData[px_safe + py_safe*Bx + (pz_safe-1)*Bx*By] - 
                       blockData[px_safe + py_safe*Bx + (pz_safe+1)*Bx*By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    // ========== 生成三角形 ==========
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            
            // 边的唯一标识（用于去重）
            size_t localEdgeIdx = (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            // 检查该边的顶点是否已经创建
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;  // 复用已有顶点
            } else {
                // 创建新顶点：在边上插值
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                
                double w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数3（double版）：处理X+边界cube
// ============================================================================
inline void processBoundaryXPlusCube(
    const double* blockData,
    const double* neighbor_xplus,
    size_t local_y, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;  // 固定在X边界
    
    // 当前块索引：x=Bx-1
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    // 邻居块索引：x=0
    size_t base_nbr = 0 + local_y * Bx + local_z * Bx * By;
    
    // ========== 8个顶点：4个在当前块，4个在X+邻居块 ==========
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],                        // 顶点0: 当前块 (Bx-1, y, z)
        neighbor_xplus[base_nbr],                    // 顶点1: 邻居块 (0, y, z)
        neighbor_xplus[base_nbr + Bx],               // 顶点2: 邻居块 (0, y+1, z)
        blockData[base_curr + Bx],                   // 顶点3: 当前块 (Bx-1, y+1, z)
        blockData[base_curr + Bx * By],              // 顶点4: 当前块 (Bx-1, y, z+1)
        neighbor_xplus[base_nbr + Bx * By],          // 顶点5: 邻居块 (0, y, z+1)
        neighbor_xplus[base_nbr + Bx + Bx * By],     // 顶点6: 邻居块 (0, y+1, z+1)
        blockData[base_curr + Bx + Bx * By]          // 顶点7: 当前块 (Bx-1, y+1, z+1)
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // ========== 梯度 ==========
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x,   local_y,   local_z,   false}, // 0: 当前块
        {0,         local_y,   local_z,   true},  // 1: X+邻居
        {0,         local_y+1, local_z,   true},  // 2: X+邻居
        {local_x,   local_y+1, local_z,   false}, // 3: 当前块
        {local_x,   local_y,   local_z+1, false}, // 4: 当前块
        {0,         local_y,   local_z+1, true},  // 5: X+邻居
        {0,         local_y+1, local_z+1, true},  // 6: X+邻居
        {local_x,   local_y+1, local_z+1, false}  // 7: 当前块
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const double* dataPtr = vi.in_neighbor ? neighbor_xplus : blockData;
        
        size_t py_safe = std::min(std::max(vi.py, size_t(1)), By - 2);
        size_t pz_safe = std::min(std::max(vi.pz, size_t(1)), Bx - 2);
        
        std::array<double, 3> grad;
        
        // X方向梯度（跨块）
        if (vi.in_neighbor) {
            // 顶点在邻居块（x=0）
            if (vi.px == 0) {
                double val_left  = blockData[local_x + vi.py * Bx + vi.pz * Bx * By];
                double val_right = (vi.px + 1 < Bx) ? 
                    neighbor_xplus[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By] : val_left;
                grad[0] = (val_left - val_right) / 2.0;
            } else {
                grad[0] = (neighbor_xplus[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                           neighbor_xplus[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
            }
        } else {
            // 顶点在当前块（x=Bx-1）
            if (vi.px == Bx - 1) {
                double val_left  = (vi.px > 0) ?
                    blockData[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] :
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                double val_right = neighbor_xplus[0 + vi.py * Bx + vi.pz * Bx * By];
                grad[0] = (val_left - val_right) / 2.0;
            } else {
                grad[0] = (blockData[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                           blockData[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
            }
        }
        
        // Y方向梯度（块内）
        if (vi.py == 0) {
            grad[1] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                       dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // Z方向梯度（块内）
        if (vi.pz == 0) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                       dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    // ========== 生成三角形 ==========
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx = (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                
                double w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数4（double版）：处理Y+边界cube
// ============================================================================
inline void processBoundaryYPlusCube(
    const double* blockData,
    const double* neighbor_yplus,
    size_t local_x, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_y = By - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr  = local_x + 0       * Bx + local_z * Bx * By;
    
    // 顶点0,1在当前块；2,3在Y+邻居；4,5在当前块；6,7在Y+邻居
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],
        blockData[base_curr + 1],
        neighbor_yplus[base_nbr + 1],
        neighbor_yplus[base_nbr],
        blockData[base_curr + Bx * By],
        blockData[base_curr + 1 + Bx * By],
        neighbor_yplus[base_nbr + 1 + Bx * By],
        neighbor_yplus[base_nbr + Bx * By]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // 梯度
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x,   local_y,   local_z,   false}, // 0: 当前块
        {local_x+1, local_y,   local_z,   false}, // 1: 当前块
        {local_x+1, 0,         local_z,   true},  // 2: Y+邻居
        {local_x,   0,         local_z,   true},  // 3: Y+邻居
        {local_x,   local_y,   local_z+1, false}, // 4: 当前块
        {local_x+1, local_y,   local_z+1, false}, // 5: 当前块
        {local_x+1, 0,         local_z+1, true},  // 6: Y+邻居
        {local_x,   0,         local_z+1, true}   // 7: Y+邻居
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const double* dataPtr = vi.in_neighbor ? neighbor_yplus : blockData;
        
        std::array<double, 3> grad;
        
        // X方向梯度（块内）
        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                       dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // Y方向梯度（跨块）
        if (vi.in_neighbor) {
            // 顶点在邻居块（y=0）
            if (vi.py == 0) {
                double val_left  = blockData[vi.px + local_y * Bx + vi.pz * Bx * By];
                double val_right = (vi.py + 1 < By) ? 
                    neighbor_yplus[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By] : val_left;
                grad[1] = (val_left - val_right) / 2.0;
            } else {
                grad[1] = (neighbor_yplus[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                           neighbor_yplus[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
            }
        } else {
            // 顶点在当前块（y=By-1）
            if (vi.py == By - 1) {
                double val_left  = (vi.py > 0) ? 
                    blockData[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] :
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                double val_right = neighbor_yplus[vi.px + 0 * Bx + vi.pz * Bx * By];
                grad[1] = (val_left - val_right) / 2.0;
            } else {
                grad[1] = (blockData[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                           blockData[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
            }
        }
        
        // Z方向梯度（块内）
        if (vi.pz == 0) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                       dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    // 生成三角形
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx = (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                
                double w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数5（double版）：处理Z+边界cube
// ============================================================================
inline void processBoundaryZPlusCube(
    const double* blockData,
    const double* neighbor_zplus,
    size_t local_x, size_t local_y,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_z = Bx - 1;  // 假设Bz = Bx
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr  = local_x + local_y * Bx + 0 * Bx * By;
    
    // 顶点0-3在当前块，顶点4-7在Z+邻居
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],
        blockData[base_curr + 1],
        blockData[base_curr + 1 + Bx],
        blockData[base_curr + Bx],
        neighbor_zplus[base_nbr],
        neighbor_zplus[base_nbr + 1],
        neighbor_zplus[base_nbr + 1 + Bx],
        neighbor_zplus[base_nbr + Bx]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // 梯度
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x,   local_y,   local_z,   false}, // 0: 当前块
        {local_x+1, local_y,   local_z,   false}, // 1: 当前块
        {local_x+1, local_y+1, local_z,   false}, // 2: 当前块
        {local_x,   local_y+1, local_z,   false}, // 3: 当前块
        {local_x,   local_y,   0,         true},  // 4: Z+邻居
        {local_x+1, local_y,   0,         true},  // 5: Z+邻居
        {local_x+1, local_y+1, 0,         true},  // 6: Z+邻居
        {local_x,   local_y+1, 0,         true}   // 7: Z+邻居
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const double* dataPtr = vi.in_neighbor ? neighbor_zplus : blockData;
        
        std::array<double, 3> grad;
        
        // X方向梯度（块内）
        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                       dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // Y方向梯度（块内）
        if (vi.py == 0) {
            grad[1] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                       dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // Z方向梯度（跨块）
        if (vi.in_neighbor) {
            // 顶点在邻居块（z=0）
            if (vi.pz == 0) {
                double val_left  = blockData[vi.px + vi.py * Bx + local_z * Bx * By];
                double val_right = (vi.pz + 1 < Bx) ? 
                    neighbor_zplus[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By] : val_left;
                grad[2] = (val_left - val_right) / 2.0;
            } else {
                grad[2] = (neighbor_zplus[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                           neighbor_zplus[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
            }
        } else {
            // 顶点在当前块（z=Bx-1）
            if (vi.pz == Bx - 1) {
                double val_left  = (vi.pz > 0) ? 
                    blockData[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] :
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                double val_right = neighbor_zplus[vi.px + vi.py * Bx + 0 * Bx * By];
                grad[2] = (val_left - val_right) / 2.0;
            } else {
                grad[2] = (blockData[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                           blockData[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
            }
        }
        
        cubeGradients[i] = grad;
    }
    
    // 生成三角形
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx = (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                
                double w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}


// ============================================================================
// 辅助函数6（double版）：处理XY边棱cube
// 功能：处理 lx=Bx-1 且 ly=By-1 的 cube，8 个顶点分布在 4 个块中
// ============================================================================
inline void processEdgeXYCube(
    const double* blockData,
    const double* neighbor_xplus,
    const double* neighbor_yplus,
    const double* neighbor_xy_diagonal,
    size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;
    size_t local_y = By - 1;
    
    // 当前块的索引
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    // X+邻居的索引（x=0, y=By-1）
    size_t base_xp = 0 + local_y * Bx + local_z * Bx * By;
    // Y+邻居的索引（x=Bx-1, y=0）
    size_t base_yp = local_x + 0 * Bx + local_z * Bx * By;
    // XY对角块的索引（x=0, y=0）
    size_t base_xy = 0 + 0 * Bx + local_z * Bx * By;
    
    // ========== 8个顶点分布在4个块中 ==========
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],                      // 顶点0: (Bx-1, By-1, z) 当前块
        neighbor_xplus[base_xp],                   // 顶点1: (0, By-1, z) X+邻居
        neighbor_xy_diagonal[base_xy],             // 顶点2: (0, 0, z) XY对角
        neighbor_yplus[base_yp],                   // 顶点3: (Bx-1, 0, z) Y+邻居
        blockData[base_curr + Bx * By],            // 顶点4: (Bx-1, By-1, z+1) 当前块
        neighbor_xplus[base_xp + Bx * By],         // 顶点5: (0, By-1, z+1) X+邻居
        neighbor_xy_diagonal[base_xy + Bx * By],   // 顶点6: (0, 0, z+1) XY对角
        neighbor_yplus[base_yp + Bx * By]          // 顶点7: (Bx-1, 0, z+1) Y+邻居
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // ========== 梯度计算 ==========
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;           // 顶点在其所属块中的局部坐标
        const double* dataPtr;       // 所属块的数据指针
        const double* neighbor_x;    // X方向邻居（用于梯度）
        const double* neighbor_y;    // Y方向邻居
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        // 顶点0: 当前块 (Bx-1, By-1, z)
        {local_x, local_y, local_z, blockData,          neighbor_xplus,      neighbor_yplus},
        // 顶点1: X+邻居 (0, By-1, z)
        {0,       local_y, local_z, neighbor_xplus,     blockData,           neighbor_xy_diagonal},
        // 顶点2: XY对角 (0, 0, z)
        {0,       0,       local_z, neighbor_xy_diagonal, neighbor_yplus,   neighbor_xplus},
        // 顶点3: Y+邻居 (Bx-1, 0, z)
        {local_x, 0,       local_z, neighbor_yplus,     neighbor_xy_diagonal, blockData},
        // 顶点4: 当前块 (Bx-1, By-1, z+1)
        {local_x, local_y, local_z+1, blockData,        neighbor_xplus,      neighbor_yplus},
        // 顶点5: X+邻居 (0, By-1, z+1)
        {0,       local_y, local_z+1, neighbor_xplus,   blockData,           neighbor_xy_diagonal},
        // 顶点6: XY对角 (0, 0, z+1)
        {0,       0,       local_z+1, neighbor_xy_diagonal, neighbor_yplus, neighbor_xplus},
        // 顶点7: Y+邻居 (Bx-1, 0, z+1)
        {local_x, 0,       local_z+1, neighbor_yplus,   neighbor_xy_diagonal, blockData}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<double, 3> grad;
        
        // === X方向梯度（跨块） ===
        if (vi.px == 0) {
            // 块左边界：左值来自 neighbor_x 的右边界
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            double val_right = (vi.px + 1 < Bx)
                ? vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[0] = (val_left - val_right) / 2.0;
        } else if (vi.px == Bx - 1) {
            // 块右边界：右值来自 neighbor_x 的左边界
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = (vi.px > 0)
                ? vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0;
        } else {
            // 块内部
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // === Y方向梯度（跨块） ===
        if (vi.py == 0) {
            // 块下边界：下值来自 neighbor_y 的上边界
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            double val_up   = (vi.py + 1 < By)
                ? vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[1] = (val_down - val_up) / 2.0;
        } else if (vi.py == By - 1) {
            // 块上边界：上值来自 neighbor_y 的下边界
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = (vi.py > 0)
                ? vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_up   = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0;
        } else {
            // 块内部
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // === Z方向梯度（块内） ===
        if (vi.pz == 0) {
            grad[2] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    // ========== 生成三角形 ==========
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数7（double版）：处理XZ边棱cube
// ============================================================================
inline void processEdgeXZCube(
    const double* blockData,
    const double* neighbor_xplus,
    const double* neighbor_zplus,
    const double* neighbor_xz_diagonal,
    size_t local_y,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;
    size_t local_z = Bx - 1;  // 假设 Bz = Bx
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_xp   = 0       + local_y * Bx + local_z * Bx * By;
    size_t base_zp   = local_x + local_y * Bx + 0 * Bx * By;
    size_t base_xz   = 0       + local_y * Bx + 0 * Bx * By;
    
    // 顶点分布：0,3在当前块；1,2在X+邻居；4,7在Z+邻居；5,6在XZ对角
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],                   // 0: (Bx-1, y, Bx-1) 当前块
        neighbor_xplus[base_xp],                // 1: (0, y, Bx-1) X+邻居
        neighbor_xplus[base_xp + Bx],           // 2: (0, y+1, Bx-1) X+邻居
        blockData[base_curr + Bx],              // 3: (Bx-1, y+1, Bx-1) 当前块
        neighbor_zplus[base_zp],                // 4: (Bx-1, y, 0) Z+邻居
        neighbor_xz_diagonal[base_xz],          // 5: (0, y, 0) XZ对角
        neighbor_xz_diagonal[base_xz + Bx],     // 6: (0, y+1, 0) XZ对角
        neighbor_zplus[base_zp + Bx]            // 7: (Bx-1, y+1, 0) Z+邻居
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // 梯度计算
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        const double* dataPtr;
        const double* neighbor_x;
        const double* neighbor_z;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        // 顶点0: 当前块 (Bx-1, y, Bx-1)
        {local_x, local_y, local_z, blockData,          neighbor_xplus,    neighbor_zplus},
        // 顶点1: X+邻居 (0, y, Bx-1)
        {0,       local_y, local_z, neighbor_xplus,     blockData,         neighbor_xz_diagonal},
        // 顶点2: X+邻居 (0, y+1, Bx-1)
        {0,       local_y+1, local_z, neighbor_xplus,   blockData,         neighbor_xz_diagonal},
        // 顶点3: 当前块 (Bx-1, y+1, Bx-1)
        {local_x, local_y+1, local_z, blockData,        neighbor_xplus,    neighbor_zplus},
        // 顶点4: Z+邻居 (Bx-1, y, 0)
        {local_x, local_y,   0,       neighbor_zplus,   neighbor_xz_diagonal, blockData},
        // 顶点5: XZ对角 (0, y, 0)
        {0,       local_y,   0,       neighbor_xz_diagonal, neighbor_zplus, neighbor_xplus},
        // 顶点6: XZ对角 (0, y+1, 0)
        {0,       local_y+1, 0,       neighbor_xz_diagonal, neighbor_zplus, neighbor_xplus},
        // 顶点7: Z+邻居 (Bx-1, y+1, 0)
        {local_x, local_y+1, 0,       neighbor_zplus,   neighbor_xz_diagonal, blockData}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<double, 3> grad;
        
        // === X方向梯度（跨块） ===
        if (vi.px == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            double val_right = (vi.px + 1 < Bx)
                ? vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[0] = (val_left - val_right) / 2.0;
        } else if (vi.px == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = (vi.px > 0)
                ? vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0;
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // === Y方向梯度（块内） ===
        if (vi.py == 0) {
            grad[1] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // === Z方向梯度（跨块） ===
        if (vi.pz == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            double val_front = (vi.pz + 1 < Bx)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]
                : val_curr;
            grad[2] = (val_back - val_front) / 2.0;
        } else if (vi.pz == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = (vi.pz > 0)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By]
                : val_curr;
            double val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    // 生成三角形
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数8（double版）：处理YZ边棱cube
// ============================================================================
inline void processEdgeYZCube(
    const double* blockData,
    const double* neighbor_yplus,
    const double* neighbor_zplus,
    const double* neighbor_yz_diagonal,
    size_t local_x,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_y = By - 1;
    size_t local_z = Bx - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_yp   = local_x + 0 * Bx       + local_z * Bx * By;
    size_t base_zp   = local_x + local_y * Bx + 0 * Bx * By;
    size_t base_yz   = local_x + 0 * Bx       + 0 * Bx * By;
    
    // 顶点分布：0,1在当前块；2,3在Y+邻居；4,5在Z+邻居；6,7在YZ对角
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],                   // 0: (x, By-1, Bx-1) 当前块
        blockData[base_curr + 1],               // 1: (x+1, By-1, Bx-1) 当前块
        neighbor_yplus[base_yp + 1],            // 2: (x+1, 0, Bx-1) Y+邻居
        neighbor_yplus[base_yp],                // 3: (x, 0, Bx-1) Y+邻居
        neighbor_zplus[base_zp],                // 4: (x, By-1, 0) Z+邻居
        neighbor_zplus[base_zp + 1],            // 5: (x+1, By-1, 0) Z+邻居
        neighbor_yz_diagonal[base_yz + 1],      // 6: (x+1, 0, 0) YZ对角
        neighbor_yz_diagonal[base_yz]           // 7: (x, 0, 0) YZ对角
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // 梯度计算
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        const double* dataPtr;
        const double* neighbor_y;
        const double* neighbor_z;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        // 顶点0: 当前块 (x, By-1, Bx-1)
        {local_x,   local_y,   local_z,   blockData,        neighbor_yplus,   neighbor_zplus},
        // 顶点1: 当前块 (x+1, By-1, Bx-1)
        {local_x+1, local_y,   local_z,   blockData,        neighbor_yplus,   neighbor_zplus},
        // 顶点2: Y+邻居 (x+1, 0, Bx-1)
        {local_x+1, 0,         local_z,   neighbor_yplus,   blockData,        neighbor_yz_diagonal},
        // 顶点3: Y+邻居 (x, 0, Bx-1)
        {local_x,   0,         local_z,   neighbor_yplus,   blockData,        neighbor_yz_diagonal},
        // 顶点4: Z+邻居 (x, By-1, 0)
        {local_x,   local_y,   0,         neighbor_zplus,   neighbor_yz_diagonal, blockData},
        // 顶点5: Z+邻居 (x+1, By-1, 0)
        {local_x+1, local_y,   0,         neighbor_zplus,   neighbor_yz_diagonal, blockData},
        // 顶点6: YZ对角 (x+1, 0, 0)
        {local_x+1, 0,         0,         neighbor_yz_diagonal, neighbor_zplus, neighbor_yplus},
        // 顶点7: YZ对角 (x, 0, 0)
        {local_x,   0,         0,         neighbor_yz_diagonal, neighbor_zplus, neighbor_yplus}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<double, 3> grad;
        
        // === X方向梯度（块内） ===
        if (vi.px == 0) {
            grad[0] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                      vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // === Y方向梯度（跨块） ===
        if (vi.py == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            double val_up   = (vi.py + 1 < By)
                ? vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[1] = (val_down - val_up) / 2.0;
        } else if (vi.py == By - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = (vi.py > 0)
                ? vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_up   = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0;
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // === Z方向梯度（跨块） ===
        if (vi.pz == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            double val_front = (vi.pz + 1 < Bx)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]
                : val_curr;
            grad[2] = (val_back - val_front) / 2.0;
        } else if (vi.pz == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = (vi.pz > 0)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By]
                : val_curr;
            double val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    // 生成三角形
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数9（double版）：处理XYZ角点cube
// ============================================================================
inline void processCornerXYZCube(
    const double* blockData,
    const double* neighbor_xplus,
    const double* neighbor_yplus,
    const double* neighbor_zplus,
    const double* neighbor_xy_diagonal,
    const double* neighbor_xz_diagonal,
    const double* neighbor_yz_diagonal,
    const double* neighbor_xyz_diagonal,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;
    size_t local_y = By - 1;
    size_t local_z = Bx - 1;
    
    // 8个块的对应索引
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_xp   = 0       + local_y * Bx + local_z * Bx * By;
    size_t base_yp   = local_x + 0       * Bx + local_z * Bx * By;
    size_t base_zp   = local_x + local_y * Bx + 0       * Bx * By;
    size_t base_xy   = 0       + 0       * Bx + local_z * Bx * By;
    size_t base_xz   = 0       + local_y * Bx + 0       * Bx * By;
    size_t base_yz   = local_x + 0       * Bx + 0       * Bx * By;
    size_t base_xyz  = 0       + 0       * Bx + 0       * Bx * By;
    
    // 8个顶点分别在8个不同的块中
    std::array<double, 8> cubeValues = {{
        blockData[base_curr],            // 0: 当前块
        neighbor_xplus[base_xp],         // 1: X+邻居
        neighbor_xy_diagonal[base_xy],   // 2: XY对角
        neighbor_yplus[base_yp],         // 3: Y+邻居
        neighbor_zplus[base_zp],         // 4: Z+邻居
        neighbor_xz_diagonal[base_xz],   // 5: XZ对角
        neighbor_xyz_diagonal[base_xyz], // 6: XYZ对角
        neighbor_yz_diagonal[base_yz]    // 7: YZ对角
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // 梯度计算，处理8个块
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;              // 顶点在其所属块中的局部坐标
        const double* dataPtr;          // 顶点所在块的数据指针
        const double* neighbor_x;       // X方向邻居块
        const double* neighbor_y;       // Y方向邻居块
        const double* neighbor_z;       // Z方向邻居块
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        // 0: 当前块 (Bx-1, By-1, Bx-1)
        {local_x, local_y, local_z, blockData,
         neighbor_xplus, neighbor_yplus, neighbor_zplus},
        
        // 1: X+邻居 (0, By-1, Bx-1)
        {0,       local_y, local_z, neighbor_xplus,
         blockData, neighbor_xy_diagonal, neighbor_xz_diagonal},
        
        // 2: XY对角 (0, 0, Bx-1)
        {0, 0, local_z, neighbor_xy_diagonal,
         neighbor_yplus, neighbor_xplus, neighbor_xyz_diagonal},
        
        // 3: Y+邻居 (Bx-1, 0, Bx-1)
        {local_x, 0, local_z, neighbor_yplus,
         neighbor_xy_diagonal, blockData, neighbor_yz_diagonal},
        
        // 4: Z+邻居 (Bx-1, By-1, 0)
        {local_x, local_y, 0, neighbor_zplus,
         neighbor_xz_diagonal, neighbor_yz_diagonal, blockData},
        
        // 5: XZ对角 (0, By-1, 0)
        {0, local_y, 0, neighbor_xz_diagonal,
         neighbor_zplus, neighbor_xyz_diagonal, neighbor_xplus},
        
        // 6: XYZ对角 (0, 0, 0)
        {0, 0, 0, neighbor_xyz_diagonal,
         neighbor_yz_diagonal, neighbor_xz_diagonal, neighbor_xy_diagonal},
        
        // 7: YZ对角 (Bx-1, 0, 0)
        {local_x, 0, 0, neighbor_yz_diagonal,
         neighbor_xyz_diagonal, neighbor_zplus, neighbor_yplus}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<double, 3> grad;
        
        // === X方向梯度（跨块） ===
        if (vi.px == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            double val_right = (vi.px + 1 < Bx)
                ? vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[0] = (val_left - val_right) / 2.0;
        } else if (vi.px == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_left = (vi.px > 0)
                ? vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0;
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // === Y方向梯度（跨块） ===
        if (vi.py == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            double val_up   = (vi.py + 1 < By)
                ? vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]
                : val_curr;
            grad[1] = (val_down - val_up) / 2.0;
        } else if (vi.py == By - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_down = (vi.py > 0)
                ? vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By]
                : val_curr;
            double val_up   = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0;
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // === Z方向梯度（跨块） ===
        if (vi.pz == 0) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            double val_front = (vi.pz + 1 < Bx)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]
                : val_curr;
            grad[2] = (val_back - val_front) / 2.0;
        } else if (vi.pz == Bx - 1) {
            double val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            double val_back = (vi.pz > 0)
                ? vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By]
                : val_curr;
            double val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    // 生成三角形
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数（double版）：X- 边界 cube
// ============================================================================
inline void processBoundaryXMinusCube(
    const double* blockData,
    const double* neighbor_xminus,
    size_t local_y, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = 0;  // 固定在X-边界
    
    // 当前块索引：x=0
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    // X-邻居块索引：x=Bx-1
    size_t base_nbr  = (Bx-1) + local_y * Bx + local_z * Bx * By;
    
    // 8个顶点：4个在X-邻居，4个在当前块
    std::array<double, 8> cubeValues = {{
        neighbor_xminus[base_nbr],                   // 0: X-邻居 (Bx-1, y, z)
        blockData[base_curr],                        // 1: 当前块 (0, y, z)
        blockData[base_curr + Bx],                   // 2: 当前块 (0, y+1, z)
        neighbor_xminus[base_nbr + Bx],              // 3: X-邻居 (Bx-1, y+1, z)
        neighbor_xminus[base_nbr + Bx * By],         // 4: X-邻居 (Bx-1, y, z+1)
        blockData[base_curr + Bx * By],              // 5: 当前块 (0, y, z+1)
        blockData[base_curr + Bx + Bx * By],         // 6: 当前块 (0, y+1, z+1)
        neighbor_xminus[base_nbr + Bx + Bx * By]     // 7: X-邻居 (Bx-1, y+1, z+1)
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // 梯度计算
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;  // true=在X-邻居块，false=在当前块
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {Bx-1, local_y,   local_z,   true},   // 0: X-邻居
        {0,    local_y,   local_z,   false},  // 1: 当前块
        {0,    local_y+1, local_z,   false},  // 2: 当前块
        {Bx-1, local_y+1, local_z,   true},   // 3: X-邻居
        {Bx-1, local_y,   local_z+1, true},   // 4: X-邻居
        {0,    local_y,   local_z+1, false},  // 5: 当前块
        {0,    local_y+1, local_z+1, false},  // 6: 当前块
        {Bx-1, local_y+1, local_z+1, true}    // 7: X-邻居
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const double* dataPtr = vi.in_neighbor ? neighbor_xminus : blockData;
        
        std::array<double, 3> grad;
        
        // X方向梯度（跨块）
        if (vi.in_neighbor) {
            // 在X-邻居（x=Bx-1）
            if (vi.px == Bx - 1) {
                double val_left = (vi.px > 0)
                    ? neighbor_xminus[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By]
                    : neighbor_xminus[vi.px + vi.py * Bx + vi.pz * Bx * By];
                double val_right = blockData[0 + vi.py * Bx + vi.pz * Bx * By];
                grad[0] = (val_left - val_right) / 2.0;
            } else {
                grad[0] = (neighbor_xminus[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                           neighbor_xminus[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
            }
        } else {
            // 在当前块（x=0）
            if (vi.px == 0) {
                double val_left  = neighbor_xminus[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
                double val_right = (vi.px + 1 < Bx)
                    ? blockData[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]
                    : blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                grad[0] = (val_left - val_right) / 2.0;
            } else {
                grad[0] = (blockData[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                           blockData[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
            }
        }
        
        // Y方向梯度（块内）
        if (vi.py == 0) {
            grad[1] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // Z方向梯度（块内）
        if (vi.pz == 0) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    // 生成三角形
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 边界 Y-（double版）
// ============================================================================
inline void processBoundaryYMinusCube(
    const double* blockData,
    const double* neighbor_yminus,
    size_t local_x, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_y = 0;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr  = local_x + (By-1) * Bx + local_z * Bx * By;
    
    std::array<double, 8> cubeValues = {{
        neighbor_yminus[base_nbr],
        neighbor_yminus[base_nbr + 1],
        blockData[base_curr + 1],
        blockData[base_curr],
        neighbor_yminus[base_nbr + Bx * By],
        neighbor_yminus[base_nbr + 1 + Bx * By],
        blockData[base_curr + 1 + Bx * By],
        blockData[base_curr + Bx * By]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // 梯度计算（结构同 X-）
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x,   By-1, local_z,   true},
        {local_x+1, By-1, local_z,   true},
        {local_x+1, 0,    local_z,   false},
        {local_x,   0,    local_z,   false},
        {local_x,   By-1, local_z+1, true},
        {local_x+1, By-1, local_z+1, true},
        {local_x+1, 0,    local_z+1, false},
        {local_x,   0,    local_z+1, false}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const double* dataPtr = vi.in_neighbor ? neighbor_yminus : blockData;
        
        std::array<double, 3> grad;
        
        // X方向（块内）
        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // Y方向（跨块）
        if (vi.in_neighbor) {
            if (vi.py == By - 1) {
                double val_down = (vi.py > 0)
                    ? neighbor_yminus[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By]
                    : neighbor_yminus[vi.px + vi.py * Bx + vi.pz * Bx * By];
                double val_up = blockData[vi.px + 0 * Bx + vi.pz * Bx * By];
                grad[1] = (val_down - val_up) / 2.0;
            } else {
                grad[1] = (neighbor_yminus[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                           neighbor_yminus[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
            }
        } else {
            if (vi.py == 0) {
                double val_down = neighbor_yminus[vi.px + (By-1) * Bx + vi.pz * Bx * By];
                double val_up   = (vi.py + 1 < By)
                    ? blockData[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]
                    : blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                grad[1] = (val_down - val_up) / 2.0;
            } else {
                grad[1] = (blockData[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                           blockData[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
            }
        }
        
        // Z方向（块内）
        if (vi.pz == 0) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                       dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
        }
        
        cubeGradients[i] = grad;
    }
    
    // 生成三角形
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 边界 Z-（double版）
// ============================================================================
inline void processBoundaryZMinusCube(
    const double* blockData,
    const double* neighbor_zminus,
    size_t local_x, size_t local_y,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    double isovalue,
    std::vector<std::array<double, 3>>& localPoints,
    std::vector<std::array<double, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_z = 0;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr  = local_x + local_y * Bx + (Bx-1) * Bx * By;
    
    std::array<double, 8> cubeValues = {{
        neighbor_zminus[base_nbr],
        neighbor_zminus[base_nbr + 1],
        neighbor_zminus[base_nbr + 1 + Bx],
        neighbor_zminus[base_nbr + Bx],
        blockData[base_curr],
        blockData[base_curr + 1],
        blockData[base_curr + 1 + Bx],
        blockData[base_curr + Bx]
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<double, 3>, 8> cubePositions = {{
        {double(global_x),   double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y),   double(global_z)},
        {double(global_x+1), double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y+1), double(global_z)},
        {double(global_x),   double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y),   double(global_z+1)},
        {double(global_x+1), double(global_y+1), double(global_z+1)},
        {double(global_x),   double(global_y+1), double(global_z+1)}
    }};
    
    // 梯度计算
    std::array<std::array<double, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x,   local_y,   Bx-1, true},
        {local_x+1, local_y,   Bx-1, true},
        {local_x+1, local_y+1, Bx-1, true},
        {local_x,   local_y+1, Bx-1, true},
        {local_x,   local_y,   0,    false},
        {local_x+1, local_y,   0,    false},
        {local_x+1, local_y+1, 0,    false},
        {local_x,   local_y+1, 0,    false}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const double* dataPtr = vi.in_neighbor ? neighbor_zminus : blockData;
        
        std::array<double, 3> grad;
        
        // X方向（块内）
        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] -
                       dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // Y方向（块内）
        if (vi.py == 0) {
            grad[1] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                      dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] -
                       dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0;
        }
        
        // Z方向（跨块）
        if (vi.in_neighbor) {
            if (vi.pz == Bx - 1) {
                double val_back  = (vi.pz > 0)
                    ? neighbor_zminus[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By]
                    : neighbor_zminus[vi.px + vi.py * Bx + vi.pz * Bx * By];
                double val_front = blockData[vi.px + vi.py * Bx + 0 * Bx * By];
                grad[2] = (val_back - val_front) / 2.0;
            } else {
                grad[2] = (neighbor_zminus[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                           neighbor_zminus[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
            }
        } else {
            if (vi.pz == 0) {
                double val_back  = neighbor_zminus[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
                double val_front = (vi.pz + 1 < Bx)
                    ? blockData[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]
                    : blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                grad[2] = (val_back - val_front) / 2.0;
            } else {
                grad[2] = (blockData[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] -
                           blockData[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0;
            }
        }
        
        cubeGradients[i] = grad;
    }
    
    // 生成三角形
    const int *triEdges = util::caseTrianglesEdges[cellCaseId];
    for (; *triEdges != -1; triEdges += 3) {
        std::array<int, 3> tri;
        for (int i = 0; i < 3; ++i) {
            int edgeIdx = triEdges[i];
            size_t localEdgeIdx =
                (local_z * By * Bx + local_y * Bx + local_x) * 12 + edgeIdx;
            
            auto it = localPointMap.find(localEdgeIdx);
            if (it != localPointMap.end()) {
                tri[i] = it->second;
            } else {
                const int *vs = util::edgeVertices[edgeIdx];
                int v1 = vs[0];
                int v2 = vs[1];
                double w = (isovalue - cubeValues[v1]) /
                           (cubeValues[v2] - cubeValues[v1]);
                
                std::array<double, 3> newPt   = util::interpolate(cubePositions[v1],  cubePositions[v2],  w);
                std::array<double, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}




























// ============================================================================
// 主函数：新版 RunMarchingCubesOnDecompressedBlocks （double 版本）
// 采用方案1：分区域处理，追求最高效率
// ============================================================================
util::TriangleMesh<double> RunMarchingCubesOnDecompressedBlocks(
    const std::vector<std::vector<double>>& decompressedBlocks,
    const std::vector<size_t>& expandGlobalSmallBlockIds,
    const std::vector<size_t>& localGlobalSmallBlockIds,
    double isovalue,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& dataShape)
{
    /*if (decompressedBlocks.size() != localGlobalSmallBlockIds.size()) {
        throw std::runtime_error("decompressedBlocks size must match localGlobalSmallBlockIds size");
    }*/

    // ========== 初始化：计算块的布局信息 ==========
    std::vector<size_t> blockCountOnEachDim(3);
    for (size_t i = 0; i < 3; i++) {
        blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];
    }

    size_t Bx = blockShape[0];
    size_t By = blockShape[1];
    size_t Bz = blockShape[2];

    // ========== 步骤1：建立块ID到数据指针的映射（一次性，O(n)）==========
    std::unordered_map<size_t, const double*> blockMap;
    blockMap.reserve(expandGlobalSmallBlockIds.size());
    for (size_t i = 0; i < expandGlobalSmallBlockIds.size(); i++) {
        blockMap[expandGlobalSmallBlockIds[i]] = decompressedBlocks[i].data();
    }

    // ========== 步骤2：排序块ID以提升Cache命中率 ==========
    // 相邻块连续处理，减少Cache miss
    std::vector<size_t> sortedBlockIds = expandGlobalSmallBlockIds;
    std::sort(sortedBlockIds.begin(), sortedBlockIds.end());

    // ========== 全局结果容器 ==========
    std::vector<std::array<double, 3>> globalPoints;
    std::vector<std::array<double, 3>> globalNormals;
    std::vector<std::array<int, 3>> globalTriangles;

    // 预分配内存（避免频繁realloc）
    size_t estimatedPoints = decompressedBlocks.size() * (Bx-1) * (By-1) * (Bz-1) * 6;
    globalPoints.reserve(estimatedPoints);
    globalNormals.reserve(estimatedPoints);
    globalTriangles.reserve(estimatedPoints / 3);

    int totalVertexOffset = 0;

    // ========== 步骤3：遍历每个核心块 ==========
    for (size_t globalBlockId : sortedBlockIds) {
        
        // === 3.1 获取当前块数据指针（O(1)查找）===
        const double* blockData = blockMap[globalBlockId];
        
        // === 3.2 计算块的3D坐标和全局起始位置 ===

        size_t block_z = globalBlockId % blockCountOnEachDim[2];
        size_t remaining = globalBlockId / blockCountOnEachDim[2];
        size_t block_y = remaining % blockCountOnEachDim[1];
        size_t block_x = remaining / blockCountOnEachDim[1];

        size_t global_x_start = block_x * Bx;
        size_t global_y_start = block_y * By;
        size_t global_z_start = block_z * Bz;

        // === 3.3 预获取6个邻居块的指针（每块只做3次查找：X+, Y+, Z+）===
        // 关键优化：整个块处理过程中不再有HashMap查找！
        const double* neighbor_xplus = getNeighborBlockPointer(globalBlockId, 1, 0, 0, blockMap, blockCountOnEachDim);
        const double* neighbor_yplus = getNeighborBlockPointer(globalBlockId, 0, 1, 0, blockMap, blockCountOnEachDim);
        const double* neighbor_zplus = getNeighborBlockPointer(globalBlockId, 0, 0, 1, blockMap, blockCountOnEachDim);


        //minus方向
        //const double* neighbor_xminus = getNeighborBlockPointer(globalBlockId, -1, 0, 0, blockMap, blockCountOnEachDim);
        //const double* neighbor_yminus = getNeighborBlockPointer(globalBlockId, 0, -1, 0, blockMap, blockCountOnEachDim);
        //const double* neighbor_zminus = getNeighborBlockPointer(globalBlockId, 0, 0, -1, blockMap, blockCountOnEachDim);



        // === 3.4 计算块的有效范围（考虑数据边界）===
        size_t max_x = std::min(Bx, dataShape[0] - global_x_start);
        size_t max_y = std::min(By, dataShape[1] - global_y_start);
        size_t max_z = std::min(Bz, dataShape[2] - global_z_start);

        // === 3.5 块内局部结果容器 ===
        std::vector<std::array<double, 3>> localPoints;
        std::vector<std::array<double, 3>> localNormals;
        std::vector<std::array<int, 3>> localTriangles;
        std::unordered_map<size_t, int> localPointMap;
        int localPtIdx = 0;

        // ========== 区域1：内部cube（95%的工作量）==========
        // 范围：[0, Bx-2] × [0, By-2] × [0, Bz-2]
        // 特点：纯数组访问，零判断，最高效率
        size_t internal_max_x = (max_x > 1) ? (max_x - 1) : 0;
        size_t internal_max_y = (max_y > 1) ? (max_y - 1) : 0;
        size_t internal_max_z = (max_z > 1) ? (max_z - 1) : 0;

        for (size_t lz = 0; lz < internal_max_z; ++lz) {
            for (size_t ly = 0; ly < internal_max_y; ++ly) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + lz;

                    // 全局边界检查
                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    // 调用内部cube处理函数（高效路径）
                    processInternalCube(blockData, lx, ly, lz, Bx, By, gx, gy, gz,
                                       isovalue, localPoints, localNormals, 
                                       localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域2：X+ 面边界 ==========
        // 条件：有X+邻居且块达到完整尺寸
        // 范围：lx=Bx-1, ly ∈ [0, By-2], lz ∈ [0, Bz-2]
        if (neighbor_xplus != nullptr && max_x == Bx) {
            for (size_t lz = 0; lz < internal_max_z; ++lz) {
                for (size_t ly = 0; ly < internal_max_y; ++ly) {
                    
                    size_t gx = global_x_start + (Bx - 1);
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + lz;

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processBoundaryXPlusCube(blockData, neighbor_xplus, ly, lz, Bx, By,
                                            gx, gy, gz, isovalue, localPoints, localNormals,
                                            localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域3：Y+ 面边界 ==========
        // 范围：ly=By-1, lx ∈ [0, Bx-2], lz ∈ [0, Bz-2]
        // 注意：不包括lx=Bx-1（避免与X+边界重复）
        if (neighbor_yplus != nullptr && max_y == By) {
            for (size_t lz = 0; lz < internal_max_z; ++lz) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + (By - 1);
                    size_t gz = global_z_start + lz;

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processBoundaryYPlusCube(blockData, neighbor_yplus, lx, lz, Bx, By,
                                            gx, gy, gz, isovalue, localPoints, localNormals,
                                            localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域4：Z+ 面边界 ==========
        // 范围：lz=Bz-1, lx ∈ [0, Bx-2], ly ∈ [0, By-2]
        // 注意：不包括lx=Bx-1和ly=By-1（避免与X+、Y+边界重复）
        if (neighbor_zplus != nullptr && max_z == Bz) {
            for (size_t ly = 0; ly < internal_max_y; ++ly) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + (Bz - 1);

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processBoundaryZPlusCube(blockData, neighbor_zplus, lx, ly, Bx, By,
                                            gx, gy, gz, isovalue, localPoints, localNormals,
                                            localTriangles, localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域5-8：边棱和角点处理 ==========
        // ========== 区域5：XY边棱（lx=31, ly=31, lz∈[0,30]）==========
        if (neighbor_xplus != nullptr && neighbor_yplus != nullptr && 
            max_x == Bx && max_y == By) {
            
            // 获取XY对角块
            const double* neighbor_xy = getNeighborBlockPointer(globalBlockId, 1, 1, 0, 
                                                               blockMap, blockCountOnEachDim);
            
            if (neighbor_xy != nullptr) {
                for (size_t lz = 0; lz < internal_max_z; ++lz) {
                    
                    size_t gx = global_x_start + (Bx - 1);
                    size_t gy = global_y_start + (By - 1);
                    size_t gz = global_z_start + lz;

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processEdgeXYCube(blockData, neighbor_xplus, neighbor_yplus, neighbor_xy,
                                     lz, Bx, By, gx, gy, gz, isovalue,
                                     localPoints, localNormals, localTriangles, 
                                     localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域6：XZ边棱（lx=31, lz=31, ly∈[0,30]）==========
        if (neighbor_xplus != nullptr && neighbor_zplus != nullptr && 
            max_x == Bx && max_z == Bz) {
            
            const double* neighbor_xz = getNeighborBlockPointer(globalBlockId, 1, 0, 1, 
                                                               blockMap, blockCountOnEachDim);
            
            if (neighbor_xz != nullptr) {
                for (size_t ly = 0; ly < internal_max_y; ++ly) {
                    
                    size_t gx = global_x_start + (Bx - 1);
                    size_t gy = global_y_start + ly;
                    size_t gz = global_z_start + (Bz - 1);

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processEdgeXZCube(blockData, neighbor_xplus, neighbor_zplus, neighbor_xz,
                                     ly, Bx, By, gx, gy, gz, isovalue,
                                     localPoints, localNormals, localTriangles, 
                                     localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域7：YZ边棱（ly=31, lz=31, lx∈[0,30]）==========
        if (neighbor_yplus != nullptr && neighbor_zplus != nullptr && 
            max_y == By && max_z == Bz) {
            
            const double* neighbor_yz = getNeighborBlockPointer(globalBlockId, 0, 1, 1, 
                                                               blockMap, blockCountOnEachDim);
            
            if (neighbor_yz != nullptr) {
                for (size_t lx = 0; lx < internal_max_x; ++lx) {
                    
                    size_t gx = global_x_start + lx;
                    size_t gy = global_y_start + (By - 1);
                    size_t gz = global_z_start + (Bz - 1);

                    if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                        continue;
                    }

                    processEdgeYZCube(blockData, neighbor_yplus, neighbor_zplus, neighbor_yz,
                                     lx, Bx, By, gx, gy, gz, isovalue,
                                     localPoints, localNormals, localTriangles, 
                                     localPointMap, localPtIdx);
                }
            }
        }

        // ========== 区域8：XYZ角点（lx=31, ly=31, lz=31）==========
        if (neighbor_xplus != nullptr && neighbor_yplus != nullptr && neighbor_zplus != nullptr &&
            max_x == Bx && max_y == By && max_z == Bz) {
            
            // 获取3个边对角块和1个体对角块
            const double* neighbor_xy = getNeighborBlockPointer(globalBlockId, 1, 1, 0, 
                                                               blockMap, blockCountOnEachDim);
            const double* neighbor_xz = getNeighborBlockPointer(globalBlockId, 1, 0, 1, 
                                                               blockMap, blockCountOnEachDim);
            const double* neighbor_yz = getNeighborBlockPointer(globalBlockId, 0, 1, 1, 
                                                               blockMap, blockCountOnEachDim);
            const double* neighbor_xyz = getNeighborBlockPointer(globalBlockId, 1, 1, 1, 
                                                                blockMap, blockCountOnEachDim);
            
            if (neighbor_xy != nullptr && neighbor_xz != nullptr && 
                neighbor_yz != nullptr && neighbor_xyz != nullptr) {
                
                size_t gx = global_x_start + (Bx - 1);
                size_t gy = global_y_start + (By - 1);
                size_t gz = global_z_start + (Bz - 1);

                if (gx + 1 < dataShape[0] && gy + 1 < dataShape[1] && gz + 1 < dataShape[2]) {
                    processCornerXYZCube(blockData, neighbor_xplus, neighbor_yplus, neighbor_zplus,
                                        neighbor_xy, neighbor_xz, neighbor_yz, neighbor_xyz,
                                        Bx, By, gx, gy, gz, isovalue,
                                        localPoints, localNormals, localTriangles, 
                                        localPointMap, localPtIdx);
                }
            }
        }





/* // ========== 新增：-方向边界处理 ==========
//minus

// 区域9：X- 面边界
if (neighbor_xminus != nullptr) {
    for (size_t lz = 0; lz < internal_max_z; ++lz) {
        for (size_t ly = 0; ly < internal_max_y; ++ly) {
            size_t gx = global_x_start + 0;  // X-边界的全局坐标
            size_t gy = global_y_start + ly;
            size_t gz = global_z_start + lz;
            
            if (gx >= dataShape[0] || gy + 1 >= dataShape[1] || gz + 1 >= dataShape[2]) {
                continue;
            }
            
            processBoundaryXMinusCube(blockData, neighbor_xminus, ly, lz, Bx, By,
                                     gx, gy, gz, isovalue, localPoints, localNormals,
                                     localTriangles, localPointMap, localPtIdx);
        }
    }
}

// 区域10：Y- 面边界
if (neighbor_yminus != nullptr) {
    for (size_t lz = 0; lz < internal_max_z; ++lz) {
        for (size_t lx = 0; lx < internal_max_x; ++lx) {
            size_t gx = global_x_start + lx;
            size_t gy = global_y_start + 0;  // Y-边界
            size_t gz = global_z_start + lz;
            
            if (gx + 1 >= dataShape[0] || gy >= dataShape[1] || gz + 1 >= dataShape[2]) {
                continue;
            }
            
            processBoundaryYMinusCube(blockData, neighbor_yminus, lx, lz, Bx, By,
                                     gx, gy, gz, isovalue, localPoints, localNormals,
                                     localTriangles, localPointMap, localPtIdx);
        }
    }
}

// 区域11：Z- 面边界
if (neighbor_zminus != nullptr) {
    for (size_t ly = 0; ly < internal_max_y; ++ly) {
        for (size_t lx = 0; lx < internal_max_x; ++lx) {
            size_t gx = global_x_start + lx;
            size_t gy = global_y_start + ly;
            size_t gz = global_z_start + 0;  // Z-边界
            
            if (gx + 1 >= dataShape[0] || gy + 1 >= dataShape[1] || gz >= dataShape[2]) {
                continue;
            }
            
            processBoundaryZMinusCube(blockData, neighbor_zminus, lx, ly, Bx, By,
                                     gx, gy, gz, isovalue, localPoints, localNormals,
                                     localTriangles, localPointMap, localPtIdx);
        }
    }
}

// ========== 新增：-方向边界处理 ==========
*/


        // === 3.6 合并局部结果到全局 ===
        globalPoints.insert(globalPoints.end(), localPoints.begin(), localPoints.end());
        globalNormals.insert(globalNormals.end(), localNormals.begin(), localNormals.end());

        // 调整三角形顶点索引（加上全局偏移）
        for (const auto& tri : localTriangles) {
            std::array<int, 3> adjustedTri = {{
                tri[0] + totalVertexOffset,
                tri[1] + totalVertexOffset,
                tri[2] + totalVertexOffset
            }};
            globalTriangles.push_back(adjustedTri);
        }

        // 检查溢出
        if (localPoints.size() > std::numeric_limits<int>::max() - totalVertexOffset) {
            throw std::runtime_error("Too many vertices for int indexing");
        }
        
        totalVertexOffset += static_cast<int>(localPoints.size());
    }

    return util::TriangleMesh<double>(globalPoints, globalNormals, globalTriangles);
}

// ============================================================
// Hole Filling：检测并填充mesh中的开放边界hole
// ============================================================
/*void fillMeshHoles(
    std::vector<std::array<double, 3>>& points,
    std::vector<std::array<double, 3>>& normals,
    std::vector<std::array<int, 3>>& triangles)
{
    auto startHoleFill = std::chrono::high_resolution_clock::now();

    // ========== Step1：找所有边界边 ==========
    // 边界边 = 只属于一个三角形的边
    std::map<std::pair<int,int>, std::vector<int>> edgeToTri;
    // edgeToTri[{min_v, max_v}] = 包含这条边的三角形index列表

    for (int ti = 0; ti < (int)triangles.size(); ++ti) {
        const auto& tri = triangles[ti];
        for (int i = 0; i < 3; ++i) {
            int a = tri[i];
            int b = tri[(i+1)%3];
            auto e = std::make_pair(std::min(a,b), std::max(a,b));
            edgeToTri[e].push_back(ti);
        }
    }

    // 收集所有边界边，建立顶点邻接关系
    // boundaryAdj[v] = 与v相连的边界边对端顶点列表
    std::unordered_map<int, std::vector<int>> boundaryAdj;
    for (const auto& [edge, tris] : edgeToTri) {
        if (tris.size() == 1) {
            // 边界边
            boundaryAdj[edge.first].push_back(edge.second);
            boundaryAdj[edge.second].push_back(edge.first);
        }
    }

    if (boundaryAdj.empty()) {
        std::cout << "[HoleFill] No holes found." << std::endl;
        return;
    }

    // ========== Step2：追踪所有hole loop ==========
    std::unordered_set<int> visitedVerts;
    std::vector<std::vector<int>> holeLoops;

    for (const auto& [startV, neighbors] : boundaryAdj) {
        if (visitedVerts.count(startV)) continue;

        // 从startV开始追踪一个loop
        std::vector<int> loop;
        int cur  = startV;
        int prev = -1;

        while (true) {
            if (visitedVerts.count(cur) && cur != startV) break;
            visitedVerts.insert(cur);
            loop.push_back(cur);

            // 找下一个未访问的邻居（排除prev）
            const auto& nbrs = boundaryAdj[cur];
            int next = -1;
            for (int nb : nbrs) {
                if (nb == prev) continue;
                if (nb == startV && loop.size() > 2) {
                    // 回到起点，loop完成
                    next = startV;
                    break;
                }
                if (!visitedVerts.count(nb)) {
                    next = nb;
                    break;
                }
            }

            if (next == -1 || next == startV) break;
            prev = cur;
            cur  = next;
        }

        if (loop.size() >= 3) {
            holeLoops.push_back(loop);
        }
    }

    std::cout << "[HoleFill] Found " << holeLoops.size() << " holes." << std::endl;

    // ========== Step3：对每个hole做填充 ==========
    int filledHoles    = 0;
    int skippedHoles   = 0;
    int totalNewTris   = 0;

    // 获取某个三角形的法向量方向，用于新三角形定向
    auto getTriNormal = [&](int v0, int v1, int v2) -> std::array<double,3> {
        const auto& p0 = points[v0];
        const auto& p1 = points[v1];
        const auto& p2 = points[v2];
        double ax = p1[0]-p0[0], ay = p1[1]-p0[1], az = p1[2]-p0[2];
        double bx = p2[0]-p0[0], by = p2[1]-p0[1], bz = p2[2]-p0[2];
        return {ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx};
    };

    auto dotProduct = [](const std::array<double,3>& a,
                         const std::array<double,3>& b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };

    for (const auto& loop : holeLoops) {
        int N = (int)loop.size();

        // 跳过极大的hole（超过500个顶点，插值质量太差）
        if (N > 500) {
            std::cout << "[HoleFill] Skipping large hole with " 
                      << N << " vertices." << std::endl;
            ++skippedHoles;
            continue;
        }

        // === 计算hole的centroid ===
        std::array<double,3> centroid = {0.0, 0.0, 0.0};
        for (int v : loop) {
            centroid[0] += points[v][0];
            centroid[1] += points[v][1];
            centroid[2] += points[v][2];
        }
        centroid[0] /= N;
        centroid[1] /= N;
        centroid[2] /= N;

        // === centroid处的法向量：用loop顶点处法向量平均 ===
        std::array<double,3> avgNormal = {0.0, 0.0, 0.0};
        for (int v : loop) {
            avgNormal[0] += normals[v][0];
            avgNormal[1] += normals[v][1];
            avgNormal[2] += normals[v][2];
        }
        double len = std::sqrt(avgNormal[0]*avgNormal[0] +
                               avgNormal[1]*avgNormal[1] +
                               avgNormal[2]*avgNormal[2]);
        if (len > 0) {
            avgNormal[0] /= len;
            avgNormal[1] /= len;
            avgNormal[2] /= len;
        }

        // === 添加centroid为新顶点 ===
        int centroidIdx = (int)points.size();
        points.push_back(centroid);
        normals.push_back(avgNormal);

        // === Fan triangulation：centroid连接loop相邻顶点对 ===
        for (int i = 0; i < N; ++i) {
            int v0 = loop[i];
            int v1 = loop[(i+1) % N];
            int vc = centroidIdx;

            // 检查方向：新三角形法向量应与邻近已有法向量一致
            auto newNorm = getTriNormal(v0, v1, vc);
            double d = dotProduct(newNorm, avgNormal);

            std::array<int,3> newTri;
            if (d >= 0) {
                newTri = {v0, v1, vc};
            } else {
                // 反转方向
                newTri = {v1, v0, vc};
            }

            triangles.push_back(newTri);
            ++totalNewTris;
        }

        ++filledHoles;
    }

    auto endHoleFill = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> holeFillTime = endHoleFill - startHoleFill;

    std::cout << "[HoleFill] Filled:  " << filledHoles  << " holes" << std::endl;
    std::cout << "[HoleFill] Skipped: " << skippedHoles << " holes (too large)" << std::endl;
    std::cout << "[HoleFill] Added:   " << totalNewTris << " new triangles" << std::endl;
    std::cerr << "[Time_holefill] Hole filling time: " 
              << holeFillTime.count() << " seconds" << std::endl;
}*/








void RunAndSaveIsosurfaceMesh(
    const std::vector<std::vector<double>>& decompressedBlocks,  
    const std::vector<size_t>& expandedBlockIds,
    const std::vector<size_t>& localGlobalSmallBlockIds,
    double isovalue,  
    const std::vector<size_t>& smallBlockShape,
    const std::vector<size_t>& stepDataShape,
    const std::string& outFile
) {
    // 打印基本信息
    std::cout << "[Info] Running Marching Cubes on decompressed blocks\n";
    std::cout << "       Volume shape: (" << stepDataShape[0] << ", " << stepDataShape[1] << ", " << stepDataShape[2] << ")\n";
    std::cout << "       Block shape:  (" << smallBlockShape[0] << ", " << smallBlockShape[1] << ", " << smallBlockShape[2] << ")\n";
    std::cout << "       Isovalue:     " << isovalue << "\n";
    std::cout << "       Output file:  " << outFile << "\n";

    // 启动计时器
    util::Timer timer;
    timer.start();

    // 执行 Marching Cubes 提取（使用 double）
    util::TriangleMesh<double> mesh = RunMarchingCubesOnDecompressedBlocks(  
        decompressedBlocks,
        expandedBlockIds,
        localGlobalSmallBlockIds,
        isovalue,
        smallBlockShape,
        stepDataShape 
    );

    timer.stop();

    // 打印结果信息
    std::cout << "[Result] Mesh vertices:  " << mesh.numberOfVertices() << "\n";
    std::cout << "[Result] Mesh triangles: " << mesh.numberOfTriangles() << "\n";
    std::cout << "[Timing] CPU time (sec):  " << timer.getCPUtime() << "\n";
    std::cout << "[Timing] Wall time (sec): " << timer.getWallTime() << "\n";


    /*fillMeshHoles(mesh.getPoints(), mesh.getNormals(), mesh.getIndexTriangles());

    std::cout << "[Result] After hole fill - vertices:  " 
              << mesh.numberOfVertices()  << "\n";
    std::cout << "[Result] After hole fill - triangles: " 
              << mesh.numberOfTriangles() << "\n";*/

    // 写入 VTK
    util::saveTriangleMesh(mesh, outFile.c_str());
}


void gaussianSmoothBlocks(
    std::vector<std::vector<double>>& blockData,
    const std::vector<size_t>& blockIds,
    const std::vector<size_t>& blockShape,
    const std::vector<size_t>& blockCountOnEachDim,
    double sigma)
{
    size_t Bx = blockShape[0];
    size_t By = blockShape[1];
    size_t Bz = blockShape[2];

    // 建立 blockId -> index 的映射，方便跨块访问
    std::unordered_map<size_t, size_t> blockIdToIdx;
    for (size_t i = 0; i < blockIds.size(); ++i) {
        blockIdToIdx[blockIds[i]] = i;
    }

    // 计算1D高斯核 (size=3, 即-1,0,+1)
    double k0 = std::exp(-0.5 * 1.0 / (sigma * sigma));  // offset=1
    double k1 = 1.0;                                       // offset=0
    double ksum = 2.0 * k0 + k1;
    k0 /= ksum;
    k1 /= ksum;
    // kern[0]=k0 (offset=-1), kern[1]=k1 (offset=0), kern[2]=k0 (offset=+1)
    double kern[3] = {k0, k1, k0};

    // 对每个块做平滑，结果写入新的vector
    std::vector<std::vector<double>> smoothedData(blockData.size());

    for (size_t bidx = 0; bidx < blockIds.size(); ++bidx) {
        size_t blockId = blockIds[bidx];
        const std::vector<double>& src = blockData[bidx];

        // 解码块3D坐标
        size_t block_z = blockId % blockCountOnEachDim[2];
        size_t rem     = blockId / blockCountOnEachDim[2];
        size_t block_y = rem % blockCountOnEachDim[1];
        size_t block_x = rem / blockCountOnEachDim[1];

        std::vector<double> smoothed(Bx * By * Bz, 0.0);

        for (size_t lz = 0; lz < Bz; ++lz) {
        for (size_t ly = 0; ly < By; ++ly) {
        for (size_t lx = 0; lx < Bx; ++lx) {

            double val  = 0.0;
            double wsum = 0.0;

            for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {

                double w = kern[dx+1] * kern[dy+1] * kern[dz+1];

                int nx = (int)lx + dx;
                int ny = (int)ly + dy;
                int nz = (int)lz + dz;

                // 判断是否需要跨块
                int bdx = 0, bdy = 0, bdz = 0;
                if      (nx < 0)        { nx += (int)Bx; bdx = -1; }
                else if (nx >= (int)Bx) { nx -= (int)Bx; bdx = +1; }
                if      (ny < 0)        { ny += (int)By; bdy = -1; }
                else if (ny >= (int)By) { ny -= (int)By; bdy = +1; }
                if      (nz < 0)        { nz += (int)Bz; bdz = -1; }
                else if (nz >= (int)Bz) { nz -= (int)Bz; bdz = +1; }

                const std::vector<double>* srcPtr = &src;

                if (bdx != 0 || bdy != 0 || bdz != 0) {
                    // 需要访问邻居块
                    int nbx = (int)block_x + bdx;
                    int nby = (int)block_y + bdy;
                    int nbz = (int)block_z + bdz;

                    if (nbx < 0 || nbx >= (int)blockCountOnEachDim[0] ||
                        nby < 0 || nby >= (int)blockCountOnEachDim[1] ||
                        nbz < 0 || nbz >= (int)blockCountOnEachDim[2]) {
                        // 超出数据边界，跳过这个邻居（权重归一化会自动处理）
                        continue;
                    }

                    size_t nbrId = (size_t)nbx * blockCountOnEachDim[1] * blockCountOnEachDim[2]
                                 + (size_t)nby * blockCountOnEachDim[2]
                                 + (size_t)nbz;

                    auto it = blockIdToIdx.find(nbrId);
                    if (it == blockIdToIdx.end()) {
                        // 邻居块没有被加载（在expanded范围之外），跳过
                        continue;
                    }
                    srcPtr = &blockData[it->second];
                }

                size_t nidx = (size_t)nx + (size_t)ny * Bx + (size_t)nz * Bx * By;
                val  += w * (*srcPtr)[nidx];
                wsum += w;

            }}}

            size_t cidx = lx + ly * Bx + lz * Bx * By;
            smoothed[cidx] = (wsum > 0.0) ? (val / wsum) : src[cidx];

        }}}

        smoothedData[bidx] = std::move(smoothed);
    }

    // 写回
    blockData = std::move(smoothedData);
}


    







int main(int argc, char* argv[]) {

    auto totalStart = std::chrono::high_resolution_clock::now(); 

    MPI_Init(&argc, &argv);

    /*int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);*/

    int mpi_rank = 0;
    int mpi_size = 1;



    std::string inputFileName, variableName, indexDir;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, blockShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0, bigBlocksPerStep = 1, smallBlocksPerBig = 1 ,  smallBlockSize = 1 , totalBlocksNumber = 1; 
    std::vector<double> queryRange;
    double relative_error_bound = 1E-3;
    size_t extraValue =0;
    


    // **解析命令行参数**
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input_file" && i + 1 < argc) {
            inputFileName = argv[++i];
        } else if (arg == "--variable_name" && i + 1 < argc) {
            variableName = argv[++i];
        } else if (arg == "--dimensions" && i + 1 < argc)
        {
            nDim = std::stoul(argv[++i]);
        }
        else if (arg == "--begin_step" && i + 1 < argc) {
            beginStepNum = std::stoul(argv[++i]);
        } else if (arg == "--end_step" && i + 1 < argc) {
            endStepNum = std::stoul(argv[++i]);
        } 

        else if (arg == "--stepData_shape" && i + nDim < argc)
        {
            for (size_t j = 0; j < nDim; j++)
            {
                stepDataShape.push_back(std::stoul(argv[++i]));
            }
        }

        else if (arg == "--bigBlock_shape" && i + nDim < argc)
        {
            for (size_t j = 0; j < nDim; j++)
            {
                blockShape.push_back(std::stoul(argv[++i]));
            }
        }
        else if (arg == "--small_block_shape")
        {
            if (nDim)
            {
                if ((int)(i + nDim) < argc)
                {
                    for (size_t j = i + 1; j < i + 1 + nDim; j++)
                    {
                        smallBlockShape.push_back(atoi(argv[j]));
                    }
                }
                else
                {
                    std::cerr << "--small_block_shape option requires [# of dimensions] argument." << std::endl;
                    return 1;
                }
            }
        }else if (arg == "--query_range" && i + 2 < argc) {
            queryRange.push_back(std::stod(argv[++i]));
            queryRange.push_back(std::stod(argv[++i]));
        }else if (arg == "--relative_error" && i + 2 < argc) {
            relative_error_bound = std::stod(argv[++i]);
            extraValue = std::stoi(argv[++i]);
        } 
    }

    /*for (size_t d = 0; d < nDim; d++) {
        bigBlocksPerStep *= stepDataShape[d] / blockShape[d];             // 每维大块数量
        smallBlocksPerBig *= blockShape[d] / smallBlockShape[d];          // 每维小块数量
        smallBlockSize *= smallBlockShape[d];                             // 每个小块的数据量
    }

    size_t nSteps = endStepNum - beginStepNum + 1;
    //小块总数
    totalBlocksNumber = nSteps * bigBlocksPerStep*smallBlocksPerBig;*/

    // === 修改：去掉大块层，直接用 smallBlockShape 计算分块数量 ===
    size_t smallBlocksPerStep = 1;
    for (size_t d = 0; d < nDim; d++) {
        smallBlocksPerStep *= stepDataShape[d] / smallBlockShape[d];
        smallBlockSize     *= smallBlockShape[d];
    }
    size_t nSteps = endStepNum - beginStepNum + 1;
    totalBlocksNumber = nSteps * smallBlocksPerStep;



    if (queryRange.size() != 2) {
        //if (mpi_rank == 0) 
        std::cerr << "Error: Invalid query range. Use --query_range <low> <high>" << std::endl;
        //MPI_Finalize();
        return 1;
    }

    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    indexDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" +safeVarName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_newour_stagger1_index/";

    // === 新增：错位 index 目录 ===
    std::string staggerIndexDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + safeVarName + "_"
                                + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_"
                                + std::to_string(extraValue) + "_newour_stagger2_index/";

    // === 新增：错位块坐标参数 ===
    // half = smallBlockShape/2，错位分块在每维的块数 = uniformBlockCount + 1
    std::vector<size_t> halfBlockShape(nDim);
    std::vector<size_t> staggerBlockCount(nDim);
    std::vector<size_t> uniformBlockCount(nDim);
    for (size_t d = 0; d < nDim; d++) {
        halfBlockShape[d]    = smallBlockShape[d] / 2;
        uniformBlockCount[d] = stepDataShape[d] / smallBlockShape[d];
        staggerBlockCount[d] = uniformBlockCount[d] + 1;
    }


    /*//index所在文件夹的根路径
    indexDir = "/home/nyan/scidx/scidx/" + std::filesystem::path(inputFileName).filename().string() + "_" + 
                std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) + "_index/";


    std::string safeVarName = variableName;
    std::replace(safeVarName.begin(), safeVarName.end(), '/', '_');
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string mySubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" +safeVarName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_stmCompress_SZ3_default_originalDataCompression/";*/
    std::string mySubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" +safeVarName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_blockZFP_originalDataCompression/";
            


    /*std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string originalSubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_originalDataCompression/";
            
    std::string mySubDir = originalSubDir + "rank_0"  + "/";*/ 

    size_t x = 0;
    std::vector<size_t> selectedBlocks;

    auto afterProcessTime = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> preProcessTime = afterProcessTime - totalStart;

    std::cerr << "[Rank " << mpi_rank << "[Time1] : preProcessTime: " << preProcessTime.count() << " seconds" << std::endl;

    std::vector<ScidxInterval<double>> allBigBlockIndices = loadBigBlockIndexFile(indexDir + "big_block_minmax");
    
    // === 新增：加载错位 index 的 big_block_minmax ===
    std::vector<ScidxInterval<double>> allStaggerBigBlockIndices =
        loadBigBlockIndexFile(staggerIndexDir + "big_block_minmax");

    std::cout << "[Debug] uniform big_block_minmax entries: " << allBigBlockIndices.size() << "\n";
    std::cout << "[Debug] stagger big_block_minmax entries: " << allStaggerBigBlockIndices.size() << "\n";


    std::cout << "[Debug] allBigBlockIndices.size() = "
          << allBigBlockIndices.size() << "\n";

    
    double globalMin = std::numeric_limits<double>::max();
    double globalMax = std::numeric_limits<double>::lowest();

    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }

    std::cout << "Global Min: " << globalMin << std::endl;
    std::cout << "Global Max: " << globalMax << std::endl;

    double error_bound = relative_error_bound * (globalMax - globalMin);
    std::cout << "Computed error_bound: " << error_bound << std::endl;
    
   
    double isovalue = queryRange[0];


    //一个mpi_rank == 0去进行大块筛选
    //if (mpi_rank == 0) {
        /*for (size_t i = 0; i < allBigBlockIndices.size(); i++) {
            if (isovalue <= allBigBlockIndices[i].high && isovalue >= allBigBlockIndices[i].low) {
                selectedBlocks.push_back(i);
            }
        }

      

        std::cout << "Selected Big Blocks Size: " << selectedBlocks.size() << std::endl;*/


        // === 修改：均匀 + 错位 big_block_minmax 取并集，unordered_set 自动去重 ===
        // 去掉大块层后，每条 entry 对应一个 step（index = stepRelIdx = actualStep - beginStep）
        std::unordered_set<size_t> selectedStepsSet;
        for (size_t i = 0; i < allBigBlockIndices.size(); i++) {
            if (isovalue <= allBigBlockIndices[i].high && isovalue >= allBigBlockIndices[i].low) {
                selectedStepsSet.insert(i);
            }
        }
        for (size_t i = 0; i < allStaggerBigBlockIndices.size(); i++) {
            if (isovalue <= allStaggerBigBlockIndices[i].high && isovalue >= allStaggerBigBlockIndices[i].low) {
                selectedStepsSet.insert(i);
            }
        }
        std::vector<size_t> selectedSteps(selectedStepsSet.begin(), selectedStepsSet.end());
        std::sort(selectedSteps.begin(), selectedSteps.end());
        std::cout << "Selected Steps (uniform+stagger union): " << selectedSteps.size() << std::endl;
                
        /*for (size_t i = 0; i < selectedBlocks.size(); ++i) {
            std::cout << selectedBlocks[i] << " ";
        }
        std::cout << std::endl;*/




        //x = selectedBlocks.size();
    //}

    //MPI_Bcast(&x, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    //selectedBlocks.resize(x);
    //MPI_Bcast(selectedBlocks.data(), x, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    /*int num_active_processes = std::min(mpi_size, (int)x);
    if (mpi_rank >= x) {
        MPI_Finalize();
        return 0;
    }*/


    auto afterQueryBigBlock = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> readAndQueryBigBlock = afterQueryBigBlock - afterProcessTime;

    std::cerr << "[Rank " << mpi_rank << "[Time2] : read and query bigBlocks: " << readAndQueryBigBlock.count() << " seconds" << std::endl;


    std::vector<size_t> allGlobalSmallBlockIds;

    /*for (size_t i = mpi_rank; i < x; i += num_active_processes) {

        当前tree搜索返回的globalid
        std::vector<size_t>  globalSmallBlockIDs = process_query_task(selectedBlocks[0], beginStepNum, bigBlocksPerStep, queryRange, indexDir, error_bound, smallBlocksPerBig);
        汇总一起做原数据查询和解压
        allGlobalSmallBlockIds.insert(allGlobalSmallBlockIds.end(),
                                globalSmallBlockIDs.begin(),
                                globalSmallBlockIDs.end());

    }*/

    size_t total_blocks = selectedBlocks.size();
    /*size_t blocks_per_rank = (total_blocks + mpi_size - 1) / mpi_size;
    size_t start_idx = mpi_rank * blocks_per_rank;
    size_t end_idx = std::min(start_idx + blocks_per_rank, total_blocks);
    std::vector<size_t> localBlocks(selectedBlocks.begin() + start_idx,
                                    selectedBlocks.begin() + end_idx);*/
    
    /*std::vector<size_t> localBlocks = selectedBlocks;


    std::vector<size_t> localGlobalSmallBlockIds;
    for (size_t i = 0; i < localBlocks.size(); ++i) {
        std::cout << "localBlocks[" << i << "] = "
          << localBlocks[i] << "\n";

        std::vector<size_t> globalSmallBlockIDs = process_query_task(
            localBlocks[i], beginStepNum, bigBlocksPerStep,
            queryRange, indexDir, error_bound,
            stepDataShape[0], blockShape[0], smallBlockShape[0]);

        localGlobalSmallBlockIds.insert(
            localGlobalSmallBlockIds.end(),
            globalSmallBlockIDs.begin(), globalSmallBlockIDs.end());
    }*/

    
    // === 修改：双 index 精细查询 + 错位→均匀转换 + unordered_set 自动去重 ===
    std::unordered_set<size_t> hitUniformBlocksSet;
    std::chrono::duration<double> mergeTime;
    for (size_t stepRelIdx : selectedSteps) {
        size_t actualStepNum = beginStepNum + stepRelIdx;
        size_t stepOffset    = stepRelIdx * smallBlocksPerStep;

        // 均匀 index 查询（已有各自计时）
        std::vector<size_t> uniformRawIds =
            queryIndexRawIds(actualStepNum, indexDir, queryRange, error_bound);

        // 错位 index 查询（已有各自计时）
        std::vector<size_t> staggerRawIds =
            queryIndexRawIds(actualStepNum, staggerIndexDir, queryRange, error_bound);

        // === 计时起点：两次查询完成后，只统计后续映射+合并+去重 ===
        auto time_before_merge = std::chrono::high_resolution_clock::now();

        // 均匀 ID 直接 insert
        for (size_t rawId : uniformRawIds)
            if (rawId < smallBlocksPerStep)
                hitUniformBlocksSet.insert(stepOffset + rawId);

        // 错位 ID 坐标转换后 insert
        for (size_t staggerId : staggerRawIds) {
            auto uniformIds = convertStaggerToUniformIds(
                staggerId, stepDataShape, smallBlockShape,
                halfBlockShape, staggerBlockCount, uniformBlockCount);
            for (size_t uid : uniformIds)
                if (uid < smallBlocksPerStep)
                    hitUniformBlocksSet.insert(stepOffset + uid);
        }

        auto time_after_merge = std::chrono::high_resolution_clock::now();
        mergeTime = time_after_merge - time_before_merge;
        std::cerr << "[Time_merge] ID mapping + insert + dedup: "
                << mergeTime.count() << " seconds" << std::endl;
    }

    // set → vector + sort 单独统计
    auto time_before_sort = std::chrono::high_resolution_clock::now();

    std::vector<size_t> localGlobalSmallBlockIds(
        hitUniformBlocksSet.begin(), hitUniformBlocksSet.end());
    std::sort(localGlobalSmallBlockIds.begin(), localGlobalSmallBlockIds.end());

    auto time_after_sort = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sortTime = time_after_sort - time_before_sort;
    std::cerr << "[Time_sort] set->vector + sort: "
            << sortTime.count() << " seconds" << std::endl;

    std::cerr << "[CHECK] Combined hit uniform blocks (deduped): "
            << localGlobalSmallBlockIds.size() << std::endl;

    double totalTime_mapping_sort = mergeTime.count() + sortTime.count();

    std::cerr << "[Time_total] merge + sort: "
          << totalTime_mapping_sort << " seconds" << std::endl;



    /*for (size_t i = 0; i < selectedBlocks.size(); ++i) {
        std::vector<size_t> globalSmallBlockIDs = process_query_task(
            selectedBlocks[i], beginStepNum, bigBlocksPerStep,
            queryRange, indexDir, error_bound, stepDataShape[0], blockShape[0],smallBlockShape[0] );

        allGlobalSmallBlockIds.insert(
            allGlobalSmallBlockIds.end(),
            globalSmallBlockIDs.begin(),
            globalSmallBlockIDs.end());
    }*/
   //通过index选出来的快个数
    std::cerr << "[CHECK] allGlobalSmallBlockIds size = " << localGlobalSmallBlockIds.size() << std::endl;



    //std::cout << "All target  Small Blocks size: " << allGlobalSmallBlockIds.size() << std::endl;

       //for test
    for (size_t bid : allGlobalSmallBlockIds) {
        if (bid >= totalBlocksNumber) {
            std::cerr << "[BUG] bid " << bid << " >= totalBlocksNumber = " << totalBlocksNumber << std::endl;
            exit(1);
        }
    }


    // ========== 新增：邻居扩展逻辑（开始）==========
auto startExpansion = std::chrono::high_resolution_clock::now();

// 1. 计算块网格布局
std::vector<size_t> blockCountOnEachDim(3);
for (size_t i = 0; i < 3; i++) {
    blockCountOnEachDim[i] = (stepDataShape[i] + smallBlockShape[i] - 1) / smallBlockShape[i];
}

std::cout << "[Info] Block grid layout: " 
          << blockCountOnEachDim[0] << " × " 
          << blockCountOnEachDim[1] << " × " 
          << blockCountOnEachDim[2] << " = "
          << (blockCountOnEachDim[0] * blockCountOnEachDim[1] * blockCountOnEachDim[2])
          << " total blocks" << std::endl;

// 2. 用set存储所有需要解压的块（自动去重）
std::unordered_set<size_t> allBlocksSet;

// 先添加所有核心块
for (size_t coreId : localGlobalSmallBlockIds) {
    allBlocksSet.insert(coreId);
}

size_t originalCoreCount = localGlobalSmallBlockIds.size();
std::cout << "[Info] Expanding 26 neighbors for " << originalCoreCount << " core blocks..." << std::endl;

// 3. 定义26个全方向邻居
struct NeighborDir {
    int dx, dy, dz;
};

std::vector<NeighborDir> neighborDirections;
for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0 && dz == 0) continue;  // 跳过自己
            neighborDirections.push_back({dx, dy, dz});
        }
    }
}

// 4. 对每个核心块，添加其26个邻居
for (size_t coreBlockId : localGlobalSmallBlockIds) {
    
    // 从块ID计算3D坐标（使用源码的映射公式）
    size_t block_z = coreBlockId % blockCountOnEachDim[2];
    size_t remaining = coreBlockId / blockCountOnEachDim[2];
    size_t block_y = remaining % blockCountOnEachDim[1];
    size_t block_x = remaining / blockCountOnEachDim[1];
    
    // 遍历26个邻居方向
    for (const auto& dir : neighborDirections) {
        int neighbor_x = static_cast<int>(block_x) + dir.dx;
        int neighbor_y = static_cast<int>(block_y) + dir.dy;
        int neighbor_z = static_cast<int>(block_z) + dir.dz;
        
        // 边界检查
        if (neighbor_x < 0 || neighbor_x >= static_cast<int>(blockCountOnEachDim[0]) ||
            neighbor_y < 0 || neighbor_y >= static_cast<int>(blockCountOnEachDim[1]) ||
            neighbor_z < 0 || neighbor_z >= static_cast<int>(blockCountOnEachDim[2])) {
            continue;  // 超出边界，跳过
        }
        
        // 计算邻居块的全局ID（使用源码的映射公式）
        size_t neighborId = static_cast<size_t>(neighbor_x) * blockCountOnEachDim[1] * blockCountOnEachDim[2] +
                           static_cast<size_t>(neighbor_y) * blockCountOnEachDim[2] +
                           static_cast<size_t>(neighbor_z);
        
        // 添加到set（自动去重）
        allBlocksSet.insert(neighborId);
    }
}

// 5. 转换为vector
std::vector<size_t> expandedBlockIds(allBlocksSet.begin(), allBlocksSet.end());

// 6. 排序以提升Cache命中率
//std::sort(expandedBlockIds.begin(), expandedBlockIds.end());

auto endExpansion = std::chrono::high_resolution_clock::now();
std::chrono::duration<float> expansionTime = endExpansion - startExpansion;

// 7. 打印统计信息
size_t addedNeighbors = expandedBlockIds.size() - originalCoreCount;


std::cout << "========== Neighbor Expansion Statistics ==========" << std::endl;
std::cout << "Core blocks (from index):  " << originalCoreCount << std::endl;
std::cout << "Neighbor blocks added:     " << addedNeighbors << std::endl;
std::cout << "Total blocks to decompress: " << expandedBlockIds.size() << std::endl;
std::cout << "Expansion time:            " << expansionTime.count() << " seconds" << std::endl;
std::cout << "====================================================" << std::endl;
// ========== 邻居扩展逻辑（结束）==========




    //auto finishQueryAllOverlapTree = std::chrono::high_resolution_clock::now();
    
    //这个是全部块，用来全部解压
    std::vector<size_t> allDecompressGlobalSmallBlockIds(totalBlocksNumber);  
    for (size_t i = 0; i < totalBlocksNumber; ++i) {
        allDecompressGlobalSmallBlockIds[i] = i;
    }

    std::vector<std::vector<double>> decompressedTargetOriginalBlockdata;

    /*if (allGlobalSmallBlockIds.size() > totalBlocksNumber * 0.9) {
        // 如果请求的块数比总块数0.9还多，使用 AllRead（更高效）
        decompressedTargetOriginalBlockdata = universalBatchDecompressBlocksAllRead(
            mySubDir,
            allDecompressGlobalSmallBlockIds,
            smallBlockSize,
            totalBlocksNumber,
            error_bound
        );
    } else {
        // 否则使用普通版本
        decompressedTargetOriginalBlockdata = universalBatchDecompressBlocks(
            mySubDir,
            expandedBlockIds,
            smallBlockSize,
            totalBlocksNumber,
            error_bound
        );
    }*/

    // 否则使用普通版本
    decompressedTargetOriginalBlockdata = universalBatchDecompressBlocks(
            mySubDir,
            expandedBlockIds,
            smallBlockSize,
            totalBlocksNumber,
            error_bound
        );


    //解压命中的小块中数据
    //std::vector<std::vector<double>> decompressedTargetOriginalBlockdata = batchDecompressBlocks(originalSubDir, allGlobalSmallBlockIds, smallBlockSize, totalBlocksNumber, error_bound);

    auto finishDecompreeAllOverlapedSamllBlocks = std::chrono::high_resolution_clock::now(); 

    /*std::cout << "=== Decompressed Target Original Block Data ===" << std::endl;
    for (size_t i = 0; i < decompressedTargetOriginalBlockdata.size(); ++i) {
        std::cout << "Block " << i << " (" << decompressedTargetOriginalBlockdata[i].size() << " values): ";
        for (size_t j = 0; j < decompressedTargetOriginalBlockdata[i].size(); ++j) {
            std::cout << decompressedTargetOriginalBlockdata[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Total blocks: " << decompressedTargetOriginalBlockdata.size() << std::endl;*/



    
    // //从解压的小块中查询数据点
    // std::vector<std::vector<double>> filtereddata  = rangeFilterData(decompressedTargetOriginalBlockdata, queryRange[0], queryRange[1]);

    // auto finishQueryInAllOverlapedSmallBlocks= std::chrono::high_resolution_clock::now(); 

    // size_t totalElements = 0;

    // /*for (size_t i = 0; i < filtereddata.size(); ++i) {
    //     std::cout << "Block " << i << ": ";
    //     for (size_t j = 0; j < filtereddata[i].size(); ++j) {
    //         std::cout << filtereddata[i][j] << " ";
    //         ++totalElements;
    //     }
    //     std::cout << std::endl;
    // }*/

    // for (size_t i = 0; i < filtereddata.size(); ++i) {
    //     totalElements += filtereddata[i].size(); 
    // }

    // std::cerr << "Total number of filtered elements: " << totalElements << std::endl;

    

    // /*MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize();*/

    

    // //在所有命中的树中找到命中全部小块id的时间
    // //std::chrono::duration<double> queryAllOverlapTreeTime = finishQueryAllOverlapTree - totalStart;

    // //解压全部命中小块原数据时间
    // //std::chrono::duration<double> decompreeAllOverlapedSamllBlocksTime = finishDecompreeAllOverlapedSamllBlocks - finishQueryAllOverlapTree;

    // //在解压后小块中查找时间
    // std::chrono::duration<double> queryInAllOverlapedSmallBlocksTime = finishQueryInAllOverlapedSmallBlocks- finishDecompreeAllOverlapedSamllBlocks;
    
    
    


    // //std::cout << "Query All Overlap Tree to get All Overlap small blocks id Time: " << queryAllOverlapTreeTime.count() << " seconds" << std::endl;
    // //std::cout << "Decompress All Overlapped Small Blocks Time: " << decompreeAllOverlapedSamllBlocksTime.count() << " seconds" << std::endl;
    // std::cerr << "[Rank " << mpi_rank << "[Time8]: Query In All Overlapped Small Blocks Time: " << queryInAllOverlapedSmallBlocksTime.count() << " seconds" << std::endl;



    // auto totalEnd = std::chrono::high_resolution_clock::now(); 

    // std::chrono::duration<double> totalQueryTime = totalEnd - totalStart;
    // std::cerr << "[Rank " << mpi_rank << "[Time All]: Total query time: " << totalQueryTime.count() << " seconds" << std::endl;



    /*RunAndSaveIsosurfaceMesh(
        decompressedTargetOriginalBlockdata,         //分块解压数据
        expandedBlockIds,
        localGlobalSmallBlockIds,
        0.06,                                     //  isovalue值
        smallBlockShape,                                // block 大小
        stepDataShape,                           // 总volume 尺寸
        "isosurface_double_gapless_5.vtk"                  // 输出ISO surface文件名
    );*/


    // ===== 新增：MC之前对解压数据做高斯平滑 =====
    /*auto smoothStart = std::chrono::high_resolution_clock::now();

    double smoothSigma = 1;
    gaussianSmoothBlocks(
        decompressedTargetOriginalBlockdata,
        expandedBlockIds,
        smallBlockShape,
        blockCountOnEachDim,
        smoothSigma
    );

    auto smoothEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> smoothTime = smoothEnd - smoothStart;
    std::cerr << "[Time_smooth] Gaussian smoothing time: " << smoothTime.count() << " seconds" << std::endl;
    std::cout << "[Info] Gaussian smoothing done, sigma = " << smoothSigma << std::endl;*/
    // ===== 平滑结束 =====


    // ============================================================
    // 为每个step分别提取等值面
    // ============================================================

    std::cout << "\n========================================" << std::endl;
    std::cout << "Extracting isosurfaces for " << nSteps << " steps..." << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 计算每个step的块数
    //size_t smallBlocksPerStep = bigBlocksPerStep * smallBlocksPerBig;

    double ratio_block = (double)decompressedTargetOriginalBlockdata.size() / smallBlocksPerStep * 100.0;

    std::cout << "[Info] Blocks per step: " << smallBlocksPerStep << std::endl;
    std::cout << "[Info] Total decompressed blocks: " << decompressedTargetOriginalBlockdata.size() << std::endl;
    std::cout << "[Info] Decompression ratio: " << ratio_block << "%" << std::endl;

    for (size_t step = 0; step < nSteps; ++step) {
        auto stepStart = std::chrono::high_resolution_clock::now();
        
        size_t currentStepNum = beginStepNum + step;
        
        // 计算当前step的块ID范围
        size_t stepOffset = step * smallBlocksPerStep;
        
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Processing Step " << currentStepNum << std::endl;
        
        // === 提取当前step的块ID ===
        std::vector<size_t> stepExpandedBlockIds;
        std::vector<size_t> stepLocalGlobalSmallBlockIds;
        
        for (size_t id : expandedBlockIds) {
            if (id >= stepOffset && id < stepOffset + smallBlocksPerStep) {
                stepExpandedBlockIds.push_back(id);
            }
        }
        
        for (size_t id : localGlobalSmallBlockIds) {
            if (id >= stepOffset && id < stepOffset + smallBlocksPerStep) {
                stepLocalGlobalSmallBlockIds.push_back(id);
            }
        }
        
        std::cout << "[Info] Step " << currentStepNum << " expanded blocks: " 
                << stepExpandedBlockIds.size() << std::endl;
        std::cout << "[Info] Step " << currentStepNum << " core blocks: " 
                << stepLocalGlobalSmallBlockIds.size() << std::endl;
        
        // === 提取当前step的解压数据 ===
        // 根据expandedBlockIds找到对应的decompressedTargetOriginalBlockdata索引
        std::vector<std::vector<double>> stepBlocks;
        stepBlocks.reserve(stepExpandedBlockIds.size());
        
        // 创建expandedBlockIds到decompressedTargetOriginalBlockdata的映射
        std::unordered_map<size_t, size_t> blockIdToIndex;
        for (size_t i = 0; i < expandedBlockIds.size(); ++i) {
            blockIdToIndex[expandedBlockIds[i]] = i;
        }
        
        // 提取当前step的块数据
        for (size_t blockId : stepExpandedBlockIds) {
            auto it = blockIdToIndex.find(blockId);
            if (it != blockIdToIndex.end()) {
                stepBlocks.push_back(decompressedTargetOriginalBlockdata[it->second]);
            } else {
                std::cerr << "Error: Block " << blockId << " not found in decompressed data!" << std::endl;
            }
        }
        
        std::cout << "[Info] Extracted " << stepBlocks.size() << " blocks for Step " 
                << currentStepNum << std::endl;
        
        // === 生成输出文件名 ===
        //std::string outFile = "isosurface_step_" + std::to_string(currentStepNum) + "_iso_-0.037620_gapless_2index_fix_gaussian1.vtk";

        /*std::string outFile = "/expanse/lustre/scratch/sdi/temp_project/iso_results/isosurface_step_" 
    + std::to_string(currentStepNum) 
    + "_iso_0.218232_gapless_2index_fix_gaussian1.vtk";*/

    std::string outFile = "/expanse/lustre/scratch/sdi/temp_project/iso_results/isosurface_step_" 
    + std::to_string(currentStepNum) 
    + "_iso_" + std::to_string(queryRange[0]) + "_gapless_2index_fix_gaussian1.vtk";

        
        double iso_value = queryRange[0]  ;
        // === 执行Marching Cubes ===
        RunAndSaveIsosurfaceMesh(
            stepBlocks,
            stepExpandedBlockIds,
            stepLocalGlobalSmallBlockIds,
            iso_value,
            smallBlockShape,
            stepDataShape,
            outFile
        );
        
        auto stepEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> stepTime = stepEnd - stepStart;
        
        std::cout << "✅ Step " << currentStepNum << " completed in " 
                << stepTime.count() << " seconds" << std::endl;
        std::cout << "   Output: " << outFile << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "All " << nSteps << " isosurfaces generated!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}

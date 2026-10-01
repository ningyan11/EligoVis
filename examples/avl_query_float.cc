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
#include <unordered_set>
#include <iomanip>
#include "../miniIsosurface/marchingCubes/util/Image3D.h"
#include "../miniIsosurface/marchingCubes/util/TriangleMesh.h"
#include "../miniIsosurface/marchingCubes/util/SaveTriangleMesh.h"
#include "../miniIsosurface/marchingCubes/util/util.h"
#include "../miniIsosurface/marchingCubes/util/MarchingCubesTables.h"
#include "../miniIsosurface/marchingCubes/util/Timer.h"
#include "../miniIsosurface/marchingCubes/util/LoadImage.h"
#include "../miniIsosurface/marchingCubes/mantevoCommon/YAML_Doc.hpp"



 void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0777);
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
    }
}


std::vector<ScidxInterval<float>> loadBigBlockIndexFile(const std::string& filename) {
    std::vector<ScidxInterval<float>> bigBlockIndices;

    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return bigBlockIndices;  // 返回空
    }

    float minVal, maxVal;
    while (inFile.read(reinterpret_cast<char*>(&minVal), sizeof(float)) &&
           inFile.read(reinterpret_cast<char*>(&maxVal), sizeof(float))) {
        bigBlockIndices.push_back({minVal, maxVal});
    }

    inFile.close();
    return bigBlockIndices;
}




//计算index筛选出的块，在全局块中的id
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

std::vector<size_t> safeQueryOverlapIds(ScidxAVLNode<float>* root, const ScidxInterval<float>& query) {
    if (root && root->id == static_cast<size_t>(-1)) {
        return {};  // dummy 树，返回空
    }
    return queryOverlapIds(root, query);
}


//查找固定的大块中小块构建的tree，并查找samelow的interval，返回ID
std::vector<size_t> process_query_task(size_t globalBlockID, size_t beginStepNum, size_t total_bigBlocks_per_step, 
                        const std::vector<float>& queryRange, const std::string& indexDir, float error_bound, size_t Nx, 
                        size_t sx, 
                        size_t small_x) {

    int rank = 0;

   

    //auto start_time_processTree = std::chrono::high_resolution_clock::now();
    // **转换全局 blockID 为 stepID 和 localBlockID**

    
    size_t stepID = beginStepNum + (globalBlockID / total_bigBlocks_per_step);
    //step中对应的第n个大块
    size_t localBlockID = globalBlockID % total_bigBlocks_per_step;

   

    // **构造文件名**
    std::string blockPrefix = std::to_string(stepID) + "-" + std::to_string(localBlockID);
 
    std::string treeID = indexDir + blockPrefix;
    
    //std::cout << "treeID:" << treeID << std::endl;
    auto [decompressedTree, decodedSkippedMin, decodedSkippedMax, skippedIdOut] = decompressOptimizedAVL(treeID, error_bound);
    
   

    auto time_finish_decompressTree = std::chrono::high_resolution_clock::now();


    //std::cout << "[DEBUG] 2 = " << total_bigBlocks_per_step << std::endl;

    //display tree
    /*ScidxAVLIntervalTree<float> tree;
    tree.setRoot(decompressedTree);
    tree.display();*/

    //调整min,max,maxhigh查找区间（消除errorbound）
    adjustAndUpdateMaxHigh(decompressedTree, error_bound);
    //std::cout << "[DEBUG] 3 = " << total_bigBlocks_per_step << std::endl;

 


    //auto time_finish_adjustTree = std::chrono::high_resolution_clock::now();

    


    ScidxInterval<float> query;
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
        float minWithError = decodedSkippedMin[i] - error_bound;
        float maxWithError = decodedSkippedMax[i] + error_bound;

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

    std::chrono::duration<float> adjustAndQueryTree = time_finish_queryTree - time_finish_decompressTree;
    std::cerr << "[Rank " << rank << "[Time5]: adjustAndQueryInTree: " << adjustAndQueryTree.count() << " seconds" << std::endl;
    
    /*// **读取小块数据**
    std::vector<float> finalResults;
    for (int smallBlockID : smallBlockIDs) {
        std::string blockFile = indexDir + blockPrefix + "-" + std::to_string(smallBlockID) + "-data.dat";
        std::ifstream file(blockFile, std::ios::binary);
        if (!file) {
            std::cerr << "[Rank " << MPI::COMM_WORLD.Get_rank() << "] Error opening small block file " << blockFile << std::endl;
            continue;
        }
        float val;
        while (file.read(reinterpret_cast<char*>(&val), sizeof(float))) {
            finalResults.push_back(val);
        }
    }

   

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);  // 获取进程 rank

    //std::cout << "[Rank " << mpi_rank << "] Read & decompress index: " << index_read_time.count();*/

    return globalIDs;

}


    //blockSize每小块数据个数大小，totalBlocks全部原数据小块大小
    std::vector<std::vector<float>> batchDecompressBlocks(const std::string& subDir,
        const std::vector<size_t>& blockIds,
        size_t blockSize,
        size_t totalBlocks,
        float error_bound) {

            int rank = 0;

            std::cout << "read: "  << std::endl;

            auto beginReadOriginalBlocks = std::chrono::high_resolution_clock::now(); 


            std::vector<std::vector<float>> decompressedOriginalresult;

            //float memory_access_time = 0.0;
            //float decompress_only_time = 0.0;

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



            std::vector<std::vector<float>> allUnpredData;
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


            std::chrono::duration<float> readOriginalBlocks = endReadOriginalBlocks - beginReadOriginalBlocks;
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
            std::vector<float> unpredData;
            std::vector<unsigned char> compData;
            std::vector<unsigned char> signBits(signBytesPerBlock);
            //std::vector<int> quant_inds(blockSize);
            //std::vector<float> decompressedBlock(blockSize);


            /*std::vector<std::vector<float>> allUnpredData;
            std::vector<std::vector<unsigned char>> allCompData;
            std::vector<std::vector<unsigned char>> allSignBits;*/


            for (size_t bid : blockIds) {

                //auto t1 = std::chrono::high_resolution_clock::now();
                /*// 读取 unpredData
                size_t unpredOffset = totalBlocks + sizeof(float) * std::accumulate(unpredSizes.begin(), unpredSizes.begin() + bid, size_t(0));
                unsigned char unpredSize = unpredSizes[bid];
                std::vector<float> unpredData(unpredSize);

                if (unpredSize > 0) {
                    unpredStream.seekg(unpredOffset, std::ios::beg);
                    unpredStream.read(reinterpret_cast<char*>(unpredData.data()), sizeof(float) * unpredSize);
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
                size_t unpredOffset = totalBlocks + sizeof(float) * unpredOffsets[bid];
                unpredData.resize(unpredSize);
                std::memcpy(unpredData.data(), allUnpredDataRead.data() + unpredOffset, unpredSize * sizeof(float));
             

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
                //memory_access_time += std::chrono::duration<float>(t2 - t1).count();

              
           

            /*}

            auto endReadOriginalBlocks = std::chrono::high_resolution_clock::now(); 


            std::chrono::duration<float> readOriginalBlocks = endReadOriginalBlocks - beginReadOriginalBlocks;
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

                double error_bound_double = static_cast<double>(error_bound);

                //恢复metadataBuffer
                std::vector<SZ3::uchar> metadataBuffer;
                metadataBuffer.push_back(0b00000010);  // 标志位

                // 写 error_bound (error)
                metadataBuffer.insert(metadataBuffer.end(),
                    reinterpret_cast<SZ3::uchar*>(&error_bound_double),
                    reinterpret_cast<SZ3::uchar*>(&error_bound_double) + sizeof(double));

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

                // 写 unpredData (float[])
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

                auto predictor = SZ3::LorenzoPredictor<float, 1, 1>(conf.absErrorBound);
                auto quantizer = SZ3::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2);
                auto decompose = SZ3::make_decomposition_lorenzo_regression<float, 1>(conf, quantizer);

       
                std::vector<float> decompressedBlock(blockSize); 
                
                const SZ3::uchar* metaPtr = metadataBuffer.data();
                size_t metaSize = metadataBuffer.size();


             
                size_t test_unpred_size = *reinterpret_cast<const size_t *>(metaPtr + 1 + sizeof(float) + sizeof(int));
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


            std::chrono::duration<float> decompressSmallBlocks = endDecompress - endReadOriginalBlocks;
            std::cerr << "[Rank " << rank << "[Time7]:  decompress small blocks time: " << decompressSmallBlocks.count() << " seconds" << std::endl;

            //std::cout << "[Time_mem_access]: Memory access time: " << memory_access_time << " seconds" << std::endl;


        return decompressedOriginalresult;
    }



    //blockSize每小块数据个数大小，totalBlocks全部原数据小块大小
    std::vector<std::vector<float>> batchDecompressBlocksSeek(const std::string& subDir,
        const std::vector<size_t>& blockIds,
        size_t blockSize,
        size_t totalBlocks,
        float error_bound) {

            std::cout << "seek: "  << std::endl;

            auto beginReadOriginalBlocks = std::chrono::high_resolution_clock::now(); 


            std::vector<std::vector<float>> decompressedOriginalresult;

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



            std::vector<std::vector<float>> allUnpredData;
            std::vector<std::vector<unsigned char>> allCompData;
            std::vector<std::vector<unsigned char>> allSignBits;


            for (size_t bid : blockIds) {
                // 读取 unpredData
                size_t unpredOffset = totalBlocks + sizeof(float) * std::accumulate(unpredSizes.begin(), unpredSizes.begin() + bid, size_t(0));
                unsigned char unpredSize = unpredSizes[bid];
                std::vector<float> unpredData(unpredSize);

                if (unpredSize > 0) {
                    unpredStream.seekg(unpredOffset, std::ios::beg);
                    unpredStream.read(reinterpret_cast<char*>(unpredData.data()), sizeof(float) * unpredSize);
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


            std::chrono::duration<float> readOriginalBlocks = endReadOriginalBlocks - beginReadOriginalBlocks;
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

                // 写 error_bound (float)
                metadataBuffer.insert(metadataBuffer.end(),
                    reinterpret_cast<SZ3::uchar*>(&error_bound),
                    reinterpret_cast<SZ3::uchar*>(&error_bound) + sizeof(float));

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

                // 写 unpredData (float[])
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

                auto predictor = SZ3::LorenzoPredictor<float, 1, 1>(conf.absErrorBound);
                auto quantizer = SZ3::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2);
                auto decompose = SZ3::make_decomposition_lorenzo_regression<float, 1>(conf, quantizer);

       
                std::vector<float> decompressedBlock(blockSize); 
                
                const SZ3::uchar* metaPtr = metadataBuffer.data();
                size_t metaSize = metadataBuffer.size();


             
                size_t test_unpred_size = *reinterpret_cast<const size_t *>(metaPtr + 1 + sizeof(float) + sizeof(int));
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


            std::chrono::duration<float> decompressSmallBlocks = endDecompress - endReadOriginalBlocks;
            std::cout << "[Time7]:  decompress small blocks time: " << decompressSmallBlocks.count() << " seconds" << std::endl;



        return decompressedOriginalresult;
    }

    std::vector<std::vector<float>> batchDecompressBlocksAllRead(const std::string& subDir,
        const std::vector<size_t>& blockIds,
        size_t blockSize,
        size_t totalBlocks,
        float error_bound) {
    std::cout << "all read: "  << std::endl;
    auto start_time_read = std::chrono::high_resolution_clock::now();      
    std::vector<std::vector<float>> decompressedOriginalresult;


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
    std::vector<float> unpredData;
    std::vector<unsigned char> compData;
    std::vector<unsigned char> signBits(signBytesPerBlock);
    std::vector<int> quant_inds(blockSize);
    std::vector<float> decompressedBlock(blockSize);


    for (size_t bid : blockIds) {
       
        // Unpred
        unsigned char unpredSize = unpredSizes[bid];
        size_t unpredOffset = totalBlocks + sizeof(float) * unpredOffsets[bid];
        unpredData.resize(unpredSize);
        std::memcpy(unpredData.data(), allUnpredData.data() + unpredOffset, unpredSize * sizeof(float));

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
            reinterpret_cast<SZ3::uchar*>(&error_bound) + sizeof(float));
       
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

        auto predictor = SZ3::LorenzoPredictor<float, 1, 1>(conf.absErrorBound);
        auto quantizer = SZ3::LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto decompose = SZ3::make_decomposition_lorenzo_regression<float, 1>(conf, quantizer);

        const SZ3::uchar* metaPtr = metadataBuffer.data();
        size_t metaSize = metadataBuffer.size();

        decompose.load(metaPtr, metaSize);
        decompose.decompress(conf, quant_inds, decompressedBlock.data());
        decompressedOriginalresult.push_back(decompressedBlock);  
    }

    

    auto end_decompress = std::chrono::high_resolution_clock::now(); 
    

    std::chrono::duration<float> readTime = begin_decompress - start_time_read;
    
     std::chrono::duration<float> compressTime = end_decompress - begin_decompress;
    
     std::cout << "[Time6]: read all compressed small blocks time: "<< readTime.count() << " seconds" << std::endl;
    
     std::cout << "[Time7]:  decompress small blocks time: " << compressTime.count() << " seconds" << std::endl;

    return decompressedOriginalresult;
}






    std::vector<std::vector<float>> rangeFilterData(
        const std::vector<std::vector<float>>& blocks,
        float queryLow,
        float queryHigh) 
    {
        std::vector<std::vector<float>> filteredBlocks;
    
        for (const auto& block : blocks) {
            std::vector<float> filtered;
            for (float val : block) {
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

// 块内局部坐标访问函数
inline float getBlockValueLocal(
    const std::vector<float>& blockData,
    size_t local_x, size_t local_y, size_t local_z,
    const std::vector<size_t>& blockShape
) {
    size_t localOffset = local_x + local_y * blockShape[0] + local_z * blockShape[0] * blockShape[1];
    return blockData[localOffset];
}

//为了解决间隙，支持外插的访问，零阶外插
inline float getBlockValueWithExtrapolation(
    const std::vector<float>& blockData,
    int local_x, int local_y, int local_z,
    const std::vector<size_t>& blockShape
) {
    // 直接钳位，使用最近邻（零阶外插）
    size_t safe_x = std::min(std::max(local_x, 0), (int)blockShape[0] - 1);
    size_t safe_y = std::min(std::max(local_y, 0), (int)blockShape[1] - 1);
    size_t safe_z = std::min(std::max(local_z, 0), (int)blockShape[2] - 1);
    
    return getBlockValueLocal(blockData, safe_x, safe_y, safe_z, blockShape);
}


/*//这是一阶插值
inline float getBlockValueWithExtrapolation(
    const std::vector<float>& blockData,
    int local_x, int local_y, int local_z,
    const std::vector<size_t>& blockShape
) {
    // ========== 先钳位所有坐标到合法范围 ==========
    size_t safe_x = std::min(std::max(local_x, 0), (int)blockShape[0] - 1);
    size_t safe_y = std::min(std::max(local_y, 0), (int)blockShape[1] - 1);
    size_t safe_z = std::min(std::max(local_z, 0), (int)blockShape[2] - 1);
    
    // 在范围内，正常访问
    if (local_x >= 0 && local_x < (int)blockShape[0] &&
        local_y >= 0 && local_y < (int)blockShape[1] &&
        local_z >= 0 && local_z < (int)blockShape[2]) {
        return getBlockValueLocal(blockData, local_x, local_y, local_z, blockShape);
    }
    
    // ========== 一阶外插（线性） ==========
    
    // X方向外插（使用安全坐标）
    if (local_x >= (int)blockShape[0] && blockShape[0] >= 2) {
        float v1 = getBlockValueLocal(blockData, blockShape[0]-1, safe_y, safe_z, blockShape);
        float v2 = getBlockValueLocal(blockData, blockShape[0]-2, safe_y, safe_z, blockShape);
        return 2.0f * v1 - v2;  // 线性外插
    }
    
    // Y方向外插
    if (local_y >= (int)blockShape[1] && blockShape[1] >= 2) {
        float v1 = getBlockValueLocal(blockData, safe_x, blockShape[1]-1, safe_z, blockShape);
        float v2 = getBlockValueLocal(blockData, safe_x, blockShape[1]-2, safe_z, blockShape);
        return 2.0f * v1 - v2;
    }
    
    // Z方向外插
    if (local_z >= (int)blockShape[2] && blockShape[2] >= 2) {
        float v1 = getBlockValueLocal(blockData, safe_x, safe_y, blockShape[2]-1, blockShape);
        float v2 = getBlockValueLocal(blockData, safe_x, safe_y, blockShape[2]-2, blockShape);
        return 2.0f * v1 - v2;
    }
    
    // 兜底：返回最近的边界点（零阶）
    return getBlockValueLocal(blockData, safe_x, safe_y, safe_z, blockShape);
}*/


/*//修改后的一阶外插
inline float getBlockValueWithExtrapolation(
    const std::vector<float>& blockData,
    int local_x, int local_y, int local_z,
    const std::vector<size_t>& blockShape
) {
    // 在范围内，正常访问
    if (local_x >= 0 && local_x < (int)blockShape[0] &&
        local_y >= 0 && local_y < (int)blockShape[1] &&
        local_z >= 0 && local_z < (int)blockShape[2]) {
        return getBlockValueLocal(blockData, local_x, local_y, local_z, blockShape);
    }
    
    // ========== 关键修复：计算超出的维度数 ==========
    int out_dims = 0;
    if (local_x < 0 || local_x >= (int)blockShape[0]) out_dims++;
    if (local_y < 0 || local_y >= (int)blockShape[1]) out_dims++;
    if (local_z < 0 || local_z >= (int)blockShape[2]) out_dims++;
    
    // 多维超出：使用零阶（多维一阶外插过于复杂）
    if (out_dims > 1) {
        size_t safe_x = std::min(std::max(local_x, 0), (int)blockShape[0] - 1);
        size_t safe_y = std::min(std::max(local_y, 0), (int)blockShape[1] - 1);
        size_t safe_z = std::min(std::max(local_z, 0), (int)blockShape[2] - 1);
        return getBlockValueLocal(blockData, safe_x, safe_y, safe_z, blockShape);
    }
    
    // ========== 单维超出：使用一阶外插 ==========
    
    // 先钳位所有坐标（用于其他维度）
    size_t safe_x = std::min(std::max(local_x, 0), (int)blockShape[0] - 1);
    size_t safe_y = std::min(std::max(local_y, 0), (int)blockShape[1] - 1);
    size_t safe_z = std::min(std::max(local_z, 0), (int)blockShape[2] - 1);
    
    // X方向超出且块足够大
    if (local_x >= (int)blockShape[0] && blockShape[0] >= 2) {
        float v1 = getBlockValueLocal(blockData, blockShape[0]-1, safe_y, safe_z, blockShape);
        float v2 = getBlockValueLocal(blockData, blockShape[0]-2, safe_y, safe_z, blockShape);
        return 2.0f * v1 - v2;
    }
    
    // Y方向超出且块足够大
    if (local_y >= (int)blockShape[1] && blockShape[1] >= 2) {
        float v1 = getBlockValueLocal(blockData, safe_x, blockShape[1]-1, safe_z, blockShape);
        float v2 = getBlockValueLocal(blockData, safe_x, blockShape[1]-2, safe_z, blockShape);
        return 2.0f * v1 - v2;
    }
    
    // Z方向超出且块足够大
    if (local_z >= (int)blockShape[2] && blockShape[2] >= 2) {
        float v1 = getBlockValueLocal(blockData, safe_x, safe_y, blockShape[2]-1, blockShape);
        float v2 = getBlockValueLocal(blockData, safe_x, safe_y, blockShape[2]-2, blockShape);
        return 2.0f * v1 - v2;
    }
    
    // 负方向超出或块太小：兜底用零阶
    return getBlockValueLocal(blockData, safe_x, safe_y, safe_z, blockShape);
}*/




// //块内cube遍历，不跨块
// //decompressedBlocks：已解压的小块数据（每块是一维数组，大小=BxByBz），不连续的命中小块
// //localGlobalSmallBlockIds：与上面一一对应的全局块ID（当前块是全局块中的id值）
// //计算当前块在全局坐标系中的起始坐标，根据块遍历，但取值根据local坐标取
// util::TriangleMesh<float> RunMarchingCubesOnDecompressedBlocks(
//     const std::vector<std::vector<float>>& decompressedBlocks,
//     const std::vector<size_t>& localGlobalSmallBlockIds,
//     float isovalue,
//     const std::vector<size_t>& blockShape,
//     const std::vector<size_t>& dataShape
// )
// {
//     if (decompressedBlocks.size() != localGlobalSmallBlockIds.size()) {
//         throw std::runtime_error("decompressedBlocks size must match localGlobalSmallBlockIds size");
//     }

//     // 计算每个维度的块数量
//     std::vector<size_t> blockCountOnEachDim(3);
//     for (size_t i = 0; i < 3; i++) {
//         blockCountOnEachDim[i] = (dataShape[i] + blockShape[i] - 1) / blockShape[i];
//     }

//     // 全局结果容器 
//     std::vector<std::array<float, 3>> globalPoints;
//     std::vector<std::array<float, 3>> globalNormals;
//     std::vector<std::array<int, 3>> globalTriangles;  

//     // 预分配内存
//     size_t estimatedPoints = decompressedBlocks.size() * (blockShape[0]-1) * (blockShape[1]-1) * (blockShape[2]-1) * 6;
//     globalPoints.reserve(estimatedPoints);
//     globalNormals.reserve(estimatedPoints);
//     globalTriangles.reserve(estimatedPoints / 3);

//     int totalVertexOffset = 0;  
//     size_t processedCubes = 0;

//     // 遍历每个解压的块
//     for (size_t dataIdx = 0; dataIdx < decompressedBlocks.size(); ++dataIdx) {
        
//         // 从块ID获取3D坐标
//         size_t globalBlockId = localGlobalSmallBlockIds[dataIdx];


//         size_t block_z = globalBlockId % blockCountOnEachDim[2];  // 128
//         size_t remaining = globalBlockId / blockCountOnEachDim[2];
//         size_t block_y = remaining % blockCountOnEachDim[1]; // 128
//         size_t block_x = remaining / blockCountOnEachDim[1]; // 128

//         // 计算当前块在全局坐标系中的起始位置
//         size_t global_x_start = block_x * blockShape[0];
//         size_t global_y_start = block_y * blockShape[1];
//         size_t global_z_start = block_z * blockShape[2];

//         /*// 计算当前块的有效范围（避免越界）
//         size_t max_x = std::min(blockShape[0] - 1, dataShape[0] - global_x_start - 1);
//         size_t max_y = std::min(blockShape[1] - 1, dataShape[1] - global_y_start - 1);
//         size_t max_z = std::min(blockShape[2] - 1, dataShape[2] - global_z_start - 1);*/


//         // ========== 修改：允许处理到边界 ==========
//         // 注意：现在 max_x/y/z 表示可以遍历的cube起始点，而不是减1
//         size_t max_x = std::min(blockShape[0], dataShape[0] - global_x_start);
//         size_t max_y = std::min(blockShape[1], dataShape[1] - global_y_start);
//         size_t max_z = std::min(blockShape[2], dataShape[2] - global_z_start);

//         // 当前块的局部结果 
//         std::vector<std::array<float, 3>> localPoints;
//         std::vector<std::array<float, 3>> localNormals;
//         std::vector<std::array<int, 3>> localTriangles;  
//         std::unordered_map<size_t, int> localPointMap; 
//         int localPtIdx = 0;  

//         // 直接访问当前块的数据
//         const std::vector<float>& currentBlockData = decompressedBlocks[dataIdx];

//         // 遍历块内的每个cube
//         for (size_t local_z = 0; local_z < max_z; ++local_z) {
//             for (size_t local_y = 0; local_y < max_y; ++local_y) {
//                 for (size_t local_x = 0; local_x < max_x; ++local_x) {

//                     // 计算cube的全局坐标
//                     size_t global_x = global_x_start + local_x;
//                     size_t global_y = global_y_start + local_y;
//                     size_t global_z = global_z_start + local_z;

//                     // 检查cube的8个顶点是否都在有效范围内
//                     if (global_x + 1 >= dataShape[0] || global_y + 1 >= dataShape[1] || global_z + 1 >= dataShape[2]) {
//                         continue;
//                     }

//                     /*// 简化：直接从当前块数据获取cube的8个顶点值
//                     std::array<float, 8> cubeValues = {{
//                         getBlockValueLocal(currentBlockData, local_x, local_y, local_z, blockShape),         // 0
//                         getBlockValueLocal(currentBlockData, local_x+1, local_y, local_z, blockShape),       // 1
//                         getBlockValueLocal(currentBlockData, local_x+1, local_y+1, local_z, blockShape),     // 2
//                         getBlockValueLocal(currentBlockData, local_x, local_y+1, local_z, blockShape),       // 3
//                         getBlockValueLocal(currentBlockData, local_x, local_y, local_z+1, blockShape),       // 4
//                         getBlockValueLocal(currentBlockData, local_x+1, local_y, local_z+1, blockShape),     // 5
//                         getBlockValueLocal(currentBlockData, local_x+1, local_y+1, local_z+1, blockShape),   // 6
//                         getBlockValueLocal(currentBlockData, local_x, local_y+1, local_z+1, blockShape)      // 7
//                     }};*/

//                     /*// ========== 修改：使用支持外插的访问函数 ==========
//                     std::array<float, 8> cubeValues = {{
//                         getBlockValueWithExtrapolation(currentBlockData, local_x,   local_y,   local_z,   blockShape), // 0
//                         getBlockValueWithExtrapolation(currentBlockData, local_x+1, local_y,   local_z,   blockShape), // 1
//                         getBlockValueWithExtrapolation(currentBlockData, local_x+1, local_y+1, local_z,   blockShape), // 2
//                         getBlockValueWithExtrapolation(currentBlockData, local_x,   local_y+1, local_z,   blockShape), // 3
//                         getBlockValueWithExtrapolation(currentBlockData, local_x,   local_y,   local_z+1, blockShape), // 4
//                         getBlockValueWithExtrapolation(currentBlockData, local_x+1, local_y,   local_z+1, blockShape), // 5
//                         getBlockValueWithExtrapolation(currentBlockData, local_x+1, local_y+1, local_z+1, blockShape), // 6
//                         getBlockValueWithExtrapolation(currentBlockData, local_x,   local_y+1, local_z+1, blockShape)  // 7
//                     }};*/




//                     // ========== 关键优化：区分快慢路径 ==========
//                     std::array<float, 8> cubeValues;

//                     // 判断cube的8个顶点是否全部在块内
//                     // 只要有一个顶点可能超出，就走慢速路径
//                     bool needExtrapolation = (local_x + 1 >= blockShape[0] ||
//                                             local_y + 1 >= blockShape[1] ||
//                                             local_z + 1 >= blockShape[2]);

//                     if (!needExtrapolation) {
//                         //快速路径（约95%的情况）：全部在块内
//                         cubeValues = {{
//                             getBlockValueLocal(currentBlockData, local_x,   local_y,   local_z,   blockShape),
//                             getBlockValueLocal(currentBlockData, local_x+1, local_y,   local_z,   blockShape),
//                             getBlockValueLocal(currentBlockData, local_x+1, local_y+1, local_z,   blockShape),
//                             getBlockValueLocal(currentBlockData, local_x,   local_y+1, local_z,   blockShape),
//                             getBlockValueLocal(currentBlockData, local_x,   local_y,   local_z+1, blockShape),
//                             getBlockValueLocal(currentBlockData, local_x+1, local_y,   local_z+1, blockShape),
//                             getBlockValueLocal(currentBlockData, local_x+1, local_y+1, local_z+1, blockShape),
//                             getBlockValueLocal(currentBlockData, local_x,   local_y+1, local_z+1, blockShape)
//                         }};
//                     } else {
//                         // 慢速路径（约5%的情况）：使用一阶外插
//                         cubeValues = {{
//                             getBlockValueWithExtrapolation(currentBlockData, local_x,   local_y,   local_z,   blockShape),
//                             getBlockValueWithExtrapolation(currentBlockData, local_x+1, local_y,   local_z,   blockShape),
//                             getBlockValueWithExtrapolation(currentBlockData, local_x+1, local_y+1, local_z,   blockShape),
//                             getBlockValueWithExtrapolation(currentBlockData, local_x,   local_y+1, local_z,   blockShape),
//                             getBlockValueWithExtrapolation(currentBlockData, local_x,   local_y,   local_z+1, blockShape),
//                             getBlockValueWithExtrapolation(currentBlockData, local_x+1, local_y,   local_z+1, blockShape),
//                             getBlockValueWithExtrapolation(currentBlockData, local_x+1, local_y+1, local_z+1, blockShape),
//                             getBlockValueWithExtrapolation(currentBlockData, local_x,   local_y+1, local_z+1, blockShape)
//                         }};
//                     }

                    

//                     // 计算marching cubes的配置ID
//                     int cellCaseId = util::findCaseId(cubeValues, isovalue);
//                     if (cellCaseId == 0 || cellCaseId == 255) {
//                         continue;
//                     }

//                     processedCubes++;

//                     // 计算cube的8个顶点坐标
//                     std::array<std::array<float, 3>, 8> cubePositions = {{
//                         {float(global_x), float(global_y), float(global_z)},
//                         {float(global_x+1), float(global_y), float(global_z)},
//                         {float(global_x+1), float(global_y+1), float(global_z)},
//                         {float(global_x), float(global_y+1), float(global_z)},
//                         {float(global_x), float(global_y), float(global_z+1)},
//                         {float(global_x+1), float(global_y), float(global_z+1)},
//                         {float(global_x+1), float(global_y+1), float(global_z+1)},
//                         {float(global_x), float(global_y+1), float(global_z+1)}
//                     }};

//                     // 块内梯度计算
//                     std::array<std::array<float, 3>, 8> cubeGradients;
//                     for (int i = 0; i < 8; ++i) {
//                         size_t px = local_x + (i & 1);
//                         size_t py = local_y + ((i >> 1) & 1);
//                         size_t pz = local_z + ((i >> 2) & 1);

//                         // ========== 新增：钳位到块内 ==========
//                         size_t px_safe = std::min(px, blockShape[0] - 1);
//                         size_t py_safe = std::min(py, blockShape[1] - 1);
//                         size_t pz_safe = std::min(pz, blockShape[2] - 1);
                    
                        
//                         std::array<float, 3> grad = {0.0f, 0.0f, 0.0f};
                        
//                         /*// X方向梯度 (块内计算)
//                         if (px == 0) {
//                             grad[0] = (getBlockValueLocal(currentBlockData, px, py, pz, blockShape) - 
//                                       getBlockValueLocal(currentBlockData, px+1, py, pz, blockShape)) / 1.0f;
//                         }
//                         else if (px == blockShape[0] - 1) {
//                             grad[0] = (getBlockValueLocal(currentBlockData, px-1, py, pz, blockShape) - 
//                                       getBlockValueLocal(currentBlockData, px, py, pz, blockShape)) / 1.0f;
//                         }
//                         else {
//                             grad[0] = (getBlockValueLocal(currentBlockData, px-1, py, pz, blockShape) - 
//                                       getBlockValueLocal(currentBlockData, px+1, py, pz, blockShape)) / 2.0f;
//                         }
                        
//                         // Y方向梯度 (块内计算)
//                         if (py == 0) {
//                             grad[1] = (getBlockValueLocal(currentBlockData, px, py, pz, blockShape) - 
//                                       getBlockValueLocal(currentBlockData, px, py+1, pz, blockShape)) / 1.0f;
//                         }
//                         else if (py == blockShape[1] - 1) {
//                             grad[1] = (getBlockValueLocal(currentBlockData, px, py-1, pz, blockShape) - 
//                                       getBlockValueLocal(currentBlockData, px, py, pz, blockShape)) / 1.0f;
//                         }
//                         else {
//                             grad[1] = (getBlockValueLocal(currentBlockData, px, py-1, pz, blockShape) - 
//                                       getBlockValueLocal(currentBlockData, px, py+1, pz, blockShape)) / 2.0f;
//                         }
                        
//                         // Z方向梯度 (块内计算)
//                         if (pz == 0) {
//                             grad[2] = (getBlockValueLocal(currentBlockData, px, py, pz, blockShape) - 
//                                       getBlockValueLocal(currentBlockData, px, py, pz+1, blockShape)) / 1.0f;
//                         }
//                         else if (pz == blockShape[2] - 1) {
//                             grad[2] = (getBlockValueLocal(currentBlockData, px, py, pz-1, blockShape) - 
//                                       getBlockValueLocal(currentBlockData, px, py, pz, blockShape)) / 1.0f;
//                         }
//                         else {
//                             grad[2] = (getBlockValueLocal(currentBlockData, px, py, pz-1, blockShape) - 
//                                       getBlockValueLocal(currentBlockData, px, py, pz+1, blockShape)) / 2.0f;
//                         }*/


//                         // X方向梯度（用 px_safe 代替 px）
//                         if (px_safe == 0) {
//                             grad[0] = (getBlockValueLocal(currentBlockData, px_safe, py_safe, pz_safe, blockShape) - 
//                                     getBlockValueLocal(currentBlockData, px_safe+1, py_safe, pz_safe, blockShape)) / 1.0f;
//                         }
//                         else if (px_safe == blockShape[0] - 1) {  // 保持用 ==
//                             grad[0] = (getBlockValueLocal(currentBlockData, px_safe-1, py_safe, pz_safe, blockShape) - 
//                                     getBlockValueLocal(currentBlockData, px_safe, py_safe, pz_safe, blockShape)) / 1.0f;
//                         }
//                         else {
//                             grad[0] = (getBlockValueLocal(currentBlockData, px_safe-1, py_safe, pz_safe, blockShape) - 
//                                     getBlockValueLocal(currentBlockData, px_safe+1, py_safe, pz_safe, blockShape)) / 2.0f;
//                         }
                        
//                         // Y方向（用 py_safe）
//                         if (py_safe == 0) {
//                             grad[1] = (getBlockValueLocal(currentBlockData, px_safe, py_safe, pz_safe, blockShape) - 
//                                     getBlockValueLocal(currentBlockData, px_safe, py_safe+1, pz_safe, blockShape)) / 1.0f;
//                         }
//                         else if (py_safe == blockShape[1] - 1) {
//                             grad[1] = (getBlockValueLocal(currentBlockData, px_safe, py_safe-1, pz_safe, blockShape) - 
//                                     getBlockValueLocal(currentBlockData, px_safe, py_safe, pz_safe, blockShape)) / 1.0f;
//                         }
//                         else {
//                             grad[1] = (getBlockValueLocal(currentBlockData, px_safe, py_safe-1, pz_safe, blockShape) - 
//                                     getBlockValueLocal(currentBlockData, px_safe, py_safe+1, pz_safe, blockShape)) / 2.0f;
//                         }
                        
//                         // Z方向（用 pz_safe）
//                         if (pz_safe == 0) {
//                             grad[2] = (getBlockValueLocal(currentBlockData, px_safe, py_safe, pz_safe, blockShape) - 
//                                     getBlockValueLocal(currentBlockData, px_safe, py_safe, pz_safe+1, blockShape)) / 1.0f;
//                         }
//                         else if (pz_safe == blockShape[2] - 1) {
//                             grad[2] = (getBlockValueLocal(currentBlockData, px_safe, py_safe, pz_safe-1, blockShape) - 
//                                     getBlockValueLocal(currentBlockData, px_safe, py_safe, pz_safe, blockShape)) / 1.0f;
//                         }
//                         else {
//                             grad[2] = (getBlockValueLocal(currentBlockData, px_safe, py_safe, pz_safe-1, blockShape) - 
//                                     getBlockValueLocal(currentBlockData, px_safe, py_safe, pz_safe+1, blockShape)) / 2.0f;
//                         }
                        
//                         cubeGradients[i] = grad;
//                     }

//                     // 生成三角形
//                     const int *triEdges = util::caseTrianglesEdges[cellCaseId];

//                     for (; *triEdges != -1; triEdges += 3) {
//                         std::array<int, 3> tri;  
                        
//                         for (int i = 0; i < 3; ++i) {
//                             int edgeIdx = triEdges[i];
//                             size_t localEdgeIdx = (local_z * max_y * max_x + local_y * max_x + local_x) * 12 + edgeIdx;
                            
//                             auto it = localPointMap.find(localEdgeIdx);
//                             if (it != localPointMap.end()) {
//                                 tri[i] = it->second;
//                             } else {
//                                 const int *vs = util::edgeVertices[edgeIdx];
//                                 int v1 = vs[0];
//                                 int v2 = vs[1];
                                
//                                 float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                                
//                                 std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
//                                 std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                                
//                                 localPoints.push_back(newPt);
//                                 localNormals.push_back(newNorm);
                                
//                                 localPointMap[localEdgeIdx] = localPtIdx;
//                                 tri[i] = localPtIdx++;
//                             }
//                         }
                        
//                         if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
//                             localTriangles.push_back(tri);
//                         }
//                     }
//                 }
//             }
//         }

//         // 合并局部结果到全局
//         globalPoints.insert(globalPoints.end(), localPoints.begin(), localPoints.end());
//         globalNormals.insert(globalNormals.end(), localNormals.begin(), localNormals.end());

//         for (const auto& tri : localTriangles) {
//             std::array<int, 3> adjustedTri = {{  
//                 tri[0] + totalVertexOffset,
//                 tri[1] + totalVertexOffset,
//                 tri[2] + totalVertexOffset
//             }};
//             globalTriangles.push_back(adjustedTri);
//         }

//         if (localPoints.size() > std::numeric_limits<int>::max() - totalVertexOffset) {
//             throw std::runtime_error("Too many vertices for int indexing");
//         }
//         totalVertexOffset += static_cast<int>(localPoints.size());
//     }

//     return util::TriangleMesh<float>(globalPoints, globalNormals, globalTriangles);
// }



// ============================================================================
// 辅助函数1：获取邻居块指针
// 功能：根据方向偏移，从blockMap中查找邻居块的数据指针
// 参数：
//   blockId - 当前块的全局ID
//   dx, dy, dz - 方向偏移（-1, 0, 或 1）
//   blockMap - 块ID到数据指针的映射
//   blockCountPerDim - 每个维度的块数量
// 返回：邻居块的数据指针，如果不存在返回nullptr
// ============================================================================
inline const float* getNeighborBlockPointer(
    size_t blockId,
    int dx, int dy, int dz,
    const std::unordered_map<size_t, const float*>& blockMap,
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
// 辅助函数2：处理内部cube
// 功能：处理块内部的cube（8个顶点全部在当前块内）
// 特点：纯数组访问，零判断，最高效率
// ============================================================================
inline void processInternalCube(
    const float* blockData,
    size_t local_x, size_t local_y, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    // 计算块内基础索引（row-major）
    size_t base = local_x + local_y * Bx + local_z * Bx * By;
    
    // ========== 8个顶点值：纯数组访问 ==========
    std::array<float, 8> cubeValues = {{
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
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // ========== 计算梯度（法向量）==========
    std::array<std::array<float, 3>, 8> cubeGradients;
    for (int i = 0; i < 8; ++i) {
        size_t px = local_x + (i & 1);
        size_t py = local_y + ((i >> 1) & 1);
        size_t pz = local_z + ((i >> 2) & 1);
        
        // 钳位到块内有效范围
        size_t px_safe = std::min(px, Bx - 1);
        size_t py_safe = std::min(py, By - 1);
        size_t pz_safe = std::min(pz, Bx - 1);
        
        std::array<float, 3> grad = {0.0f, 0.0f, 0.0f};
        
        // X方向梯度（中心差分或单侧差分）
        if (px_safe == 0) {
            grad[0] = blockData[px_safe + py_safe*Bx + pz_safe*Bx*By] - 
                     blockData[(px_safe+1) + py_safe*Bx + pz_safe*Bx*By];
        } else if (px_safe == Bx - 1) {
            grad[0] = blockData[(px_safe-1) + py_safe*Bx + pz_safe*Bx*By] - 
                     blockData[px_safe + py_safe*Bx + pz_safe*Bx*By];
        } else {
            grad[0] = (blockData[(px_safe-1) + py_safe*Bx + pz_safe*Bx*By] - 
                      blockData[(px_safe+1) + py_safe*Bx + pz_safe*Bx*By]) / 2.0f;
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
                      blockData[px_safe + (py_safe+1)*Bx + pz_safe*Bx*By]) / 2.0f;
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
                      blockData[px_safe + py_safe*Bx + (pz_safe+1)*Bx*By]) / 2.0f;
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
                localPoints.push_back(newPt);
                localNormals.push_back(newNorm);
                
                localPointMap[localEdgeIdx] = localPtIdx;
                tri[i] = localPtIdx++;
            }
        }
        
        // 检查三角形顶点有效性
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]) {
            localTriangles.push_back(tri);
        }
    }
}

// ============================================================================
// 辅助函数3：处理X+边界cube
// 功能：处理块的X+边界（lx=Bx-1）上的cube，部分顶点在X+邻居块
// ============================================================================
inline void processBoundaryXPlusCube(
    const float* blockData,
    const float* neighbor_xplus,
    size_t local_y, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
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
    std::array<float, 8> cubeValues = {{
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
    
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // ========== ✅ 修复：正确计算梯度 ==========
    std::array<std::array<float, 3>, 8> cubeGradients;
    
    // 顶点信息：(局部x, 局部y, 局部z, 是否在邻居块)
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x, local_y, local_z, false},       // 0: 当前块
        {0, local_y, local_z, true},              // 1: X+邻居
        {0, local_y+1, local_z, true},            // 2: X+邻居
        {local_x, local_y+1, local_z, false},     // 3: 当前块
        {local_x, local_y, local_z+1, false},     // 4: 当前块
        {0, local_y, local_z+1, true},            // 5: X+邻居
        {0, local_y+1, local_z+1, true},          // 6: X+邻居
        {local_x, local_y+1, local_z+1, false}    // 7: 当前块
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const float* dataPtr = vi.in_neighbor ? neighbor_xplus : blockData;
        
        size_t py_safe = std::min(std::max(vi.py, size_t(1)), By - 2);
        size_t pz_safe = std::min(std::max(vi.pz, size_t(1)), Bx - 2);
        
        std::array<float, 3> grad;
        
        // === X方向梯度（需要跨块） ===
        if (vi.in_neighbor) {
            // 顶点在邻居块（x=0）
            if (vi.px == 0) {
                // 左邻居在当前块的x=Bx-1
                float val_left = blockData[local_x + vi.py * Bx + vi.pz * Bx * By];
                float val_right = (vi.px + 1 < Bx) ? 
                    neighbor_xplus[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By] : val_left;
                grad[0] = (val_left - val_right) / 2.0f;
            } else {
                // 正常中心差分
                grad[0] = (neighbor_xplus[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                          neighbor_xplus[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
            }
        } else {
            // 顶点在当前块（x=Bx-1）
            if (vi.px == Bx - 1) {
                // 右邻居在X+块的x=0
                float val_left = (vi.px > 0) ? 
                    blockData[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] : 
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                float val_right = neighbor_xplus[0 + vi.py * Bx + vi.pz * Bx * By];
                grad[0] = (val_left - val_right) / 2.0f;
            } else {
                // 正常中心差分
                grad[0] = (blockData[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                          blockData[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
            }
        }
        
        // === Y方向梯度（块内） ===
        if (vi.py == 0) {
            grad[1] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Z方向梯度（块内） ===
        if (vi.pz == 0) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
        }
        
        cubeGradients[i] = grad;
    }
    
    // ========== 生成三角形（保持不变）==========
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
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
// 辅助函数4：处理Y+边界cube
// ============================================================================
inline void processBoundaryYPlusCube(
    const float* blockData,
    const float* neighbor_yplus,
    size_t local_x, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_y = By - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr = local_x + 0 * Bx + local_z * Bx * By;
    
    // 顶点0,1在当前块；2,3在Y+邻居；4,5在当前块；6,7在Y+邻居
    std::array<float, 8> cubeValues = {{
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
    
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // ========== ✅ 修复：正确计算梯度 ==========
    std::array<std::array<float, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x, local_y, local_z, false},       // 0: 当前块
        {local_x+1, local_y, local_z, false},     // 1: 当前块
        {local_x+1, 0, local_z, true},            // 2: Y+邻居
        {local_x, 0, local_z, true},              // 3: Y+邻居
        {local_x, local_y, local_z+1, false},     // 4: 当前块
        {local_x+1, local_y, local_z+1, false},   // 5: 当前块
        {local_x+1, 0, local_z+1, true},          // 6: Y+邻居
        {local_x, 0, local_z+1, true}             // 7: Y+邻居
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const float* dataPtr = vi.in_neighbor ? neighbor_yplus : blockData;
        
        std::array<float, 3> grad;
        
        // === X方向梯度（块内） ===
        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Y方向梯度（需要跨块） ===
        if (vi.in_neighbor) {
            // 顶点在邻居块（y=0）
            if (vi.py == 0) {
                float val_left = blockData[vi.px + local_y * Bx + vi.pz * Bx * By];
                float val_right = (vi.py + 1 < By) ? 
                    neighbor_yplus[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By] : val_left;
                grad[1] = (val_left - val_right) / 2.0f;
            } else {
                grad[1] = (neighbor_yplus[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                          neighbor_yplus[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
            }
        } else {
            // 顶点在当前块（y=By-1）
            if (vi.py == By - 1) {
                float val_left = (vi.py > 0) ? 
                    blockData[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] : 
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                float val_right = neighbor_yplus[vi.px + 0 * Bx + vi.pz * Bx * By];
                grad[1] = (val_left - val_right) / 2.0f;
            } else {
                grad[1] = (blockData[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                          blockData[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
            }
        }
        
        // === Z方向梯度（块内） ===
        if (vi.pz == 0) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
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
// 辅助函数5：处理Z+边界cube
// ============================================================================
inline void processBoundaryZPlusCube(
    const float* blockData,
    const float* neighbor_zplus,
    size_t local_x, size_t local_y,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_z = Bx - 1;  // 假设Bz = Bx
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr = local_x + local_y * Bx + 0 * Bx * By;
    
    // 顶点0-3在当前块，顶点4-7在Z+邻居
    std::array<float, 8> cubeValues = {{
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
    
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // ========== ✅ 修复：正确计算梯度 ==========
    std::array<std::array<float, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x, local_y, local_z, false},       // 0: 当前块
        {local_x+1, local_y, local_z, false},     // 1: 当前块
        {local_x+1, local_y+1, local_z, false},   // 2: 当前块
        {local_x, local_y+1, local_z, false},     // 3: 当前块
        {local_x, local_y, 0, true},              // 4: Z+邻居
        {local_x+1, local_y, 0, true},            // 5: Z+邻居
        {local_x+1, local_y+1, 0, true},          // 6: Z+邻居
        {local_x, local_y+1, 0, true}             // 7: Z+邻居
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const float* dataPtr = vi.in_neighbor ? neighbor_zplus : blockData;
        
        std::array<float, 3> grad;
        
        // === X方向梯度（块内） ===
        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Y方向梯度（块内） ===
        if (vi.py == 0) {
            grad[1] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Z方向梯度（需要跨块） ===
        if (vi.in_neighbor) {
            // 顶点在邻居块（z=0）
            if (vi.pz == 0) {
                float val_left = blockData[vi.px + vi.py * Bx + local_z * Bx * By];
                float val_right = (vi.pz + 1 < Bx) ? 
                    neighbor_zplus[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By] : val_left;
                grad[2] = (val_left - val_right) / 2.0f;
            } else {
                grad[2] = (neighbor_zplus[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                          neighbor_zplus[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
            }
        } else {
            // 顶点在当前块（z=Bx-1）
            if (vi.pz == Bx - 1) {
                float val_left = (vi.pz > 0) ? 
                    blockData[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] : 
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                float val_right = neighbor_zplus[vi.px + vi.py * Bx + 0 * Bx * By];
                grad[2] = (val_left - val_right) / 2.0f;
            } else {
                grad[2] = (blockData[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                          blockData[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
            }
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
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
// 辅助函数6：处理XY边棱cube
// 功能：处理lx=Bx-1且ly=By-1的cube，8个顶点分布在4个块中
// ============================================================================
inline void processEdgeXYCube(
    const float* blockData,
    const float* neighbor_xplus,
    const float* neighbor_yplus,
    const float* neighbor_xy_diagonal,
    size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
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
    std::array<float, 8> cubeValues = {{
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
    
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // ========== ✅ 正确计算梯度，使用邻居块数据 ==========
    std::array<std::array<float, 3>, 8> cubeGradients;
    
    // 定义每个顶点的位置和所属块
    struct VertexInfo {
        size_t px, py, pz;           // 顶点在其所属块中的局部坐标
        const float* dataPtr;         // 所属块的数据指针
        const float* neighbor_x;      // X方向邻居（用于梯度计算）
        const float* neighbor_y;      // Y方向邻居
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        // 顶点0: 当前块 (Bx-1, By-1, z)
        {local_x, local_y, local_z, blockData, neighbor_xplus, neighbor_yplus},
        // 顶点1: X+邻居 (0, By-1, z)
        {0, local_y, local_z, neighbor_xplus, blockData, neighbor_xy_diagonal},
        // 顶点2: XY对角 (0, 0, z)
        {0, 0, local_z, neighbor_xy_diagonal, neighbor_yplus, neighbor_xplus},
        // 顶点3: Y+邻居 (Bx-1, 0, z)
        {local_x, 0, local_z, neighbor_yplus, neighbor_xy_diagonal, blockData},
        // 顶点4: 当前块 (Bx-1, By-1, z+1)
        {local_x, local_y, local_z+1, blockData, neighbor_xplus, neighbor_yplus},
        // 顶点5: X+邻居 (0, By-1, z+1)
        {0, local_y, local_z+1, neighbor_xplus, blockData, neighbor_xy_diagonal},
        // 顶点6: XY对角 (0, 0, z+1)
        {0, 0, local_z+1, neighbor_xy_diagonal, neighbor_yplus, neighbor_xplus},
        // 顶点7: Y+邻居 (Bx-1, 0, z+1)
        {local_x, 0, local_z+1, neighbor_yplus, neighbor_xy_diagonal, blockData}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<float, 3> grad;
        
        // === X方向梯度（需要跨块） ===
        if (vi.px == 0) {
            // 在块的左边界，左邻居在neighbor_x的右边界
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            float val_right = (vi.px + 1 < Bx) ? 
                vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By] : val_curr;
            grad[0] = (val_left - val_right) / 2.0f;
        } else if (vi.px == Bx - 1) {
            // 在块的右边界，右邻居在neighbor_x的左边界
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = (vi.px > 0) ? 
                vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] : val_curr;
            float val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0f;
        } else {
            // 在块内部，正常中心差分
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Y方向梯度（需要跨块） ===
        if (vi.py == 0) {
            // 在块的下边界，下邻居在neighbor_y的上边界
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            float val_up = (vi.py + 1 < By) ? 
                vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By] : val_curr;
            grad[1] = (val_down - val_up) / 2.0f;
        } else if (vi.py == By - 1) {
            // 在块的上边界，上邻居在neighbor_y的下边界
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = (vi.py > 0) ? 
                vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] : val_curr;
            float val_up = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0f;
        } else {
            // 在块内部，正常中心差分
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Z方向梯度（块内计算） ===
        if (vi.pz == 0) {
            grad[2] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                     vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
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
// 辅助函数7：处理XZ边棱cube
// ============================================================================
inline void processEdgeXZCube(
    const float* blockData,
    const float* neighbor_xplus,
    const float* neighbor_zplus,
    const float* neighbor_xz_diagonal,
    size_t local_y,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;
    size_t local_z = Bx - 1;  // 假设Bz = Bx
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_xp = 0 + local_y * Bx + local_z * Bx * By;
    size_t base_zp = local_x + local_y * Bx + 0 * Bx * By;
    size_t base_xz = 0 + local_y * Bx + 0 * Bx * By;
    
    // 顶点分布：0,3在当前块；1,2在X+邻居；4,7在Z+邻居；5,6在XZ对角
    std::array<float, 8> cubeValues = {{
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
    
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // ========== ✅ 正确计算梯度 ==========
    std::array<std::array<float, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        const float* dataPtr;
        const float* neighbor_x;
        const float* neighbor_z;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        // 顶点0: 当前块 (Bx-1, y, Bx-1)
        {local_x, local_y, local_z, blockData, neighbor_xplus, neighbor_zplus},
        // 顶点1: X+邻居 (0, y, Bx-1)
        {0, local_y, local_z, neighbor_xplus, blockData, neighbor_xz_diagonal},
        // 顶点2: X+邻居 (0, y+1, Bx-1)
        {0, local_y+1, local_z, neighbor_xplus, blockData, neighbor_xz_diagonal},
        // 顶点3: 当前块 (Bx-1, y+1, Bx-1)
        {local_x, local_y+1, local_z, blockData, neighbor_xplus, neighbor_zplus},
        // 顶点4: Z+邻居 (Bx-1, y, 0)
        {local_x, local_y, 0, neighbor_zplus, neighbor_xz_diagonal, blockData},
        // 顶点5: XZ对角 (0, y, 0)
        {0, local_y, 0, neighbor_xz_diagonal, neighbor_zplus, neighbor_xplus},
        // 顶点6: XZ对角 (0, y+1, 0)
        {0, local_y+1, 0, neighbor_xz_diagonal, neighbor_zplus, neighbor_xplus},
        // 顶点7: Z+邻居 (Bx-1, y+1, 0)
        {local_x, local_y+1, 0, neighbor_zplus, neighbor_xz_diagonal, blockData}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<float, 3> grad;
        
        // === X方向梯度（需要跨块） ===
        if (vi.px == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            float val_right = (vi.px + 1 < Bx) ? 
                vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By] : val_curr;
            grad[0] = (val_left - val_right) / 2.0f;
        } else if (vi.px == Bx - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = (vi.px > 0) ? 
                vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] : val_curr;
            float val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0f;
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Y方向梯度（块内计算） ===
        if (vi.py == 0) {
            grad[1] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                     vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Z方向梯度（需要跨块） ===
        if (vi.pz == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            float val_front = (vi.pz + 1 < Bx) ? 
                vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By] : val_curr;
            grad[2] = (val_back - val_front) / 2.0f;
        } else if (vi.pz == Bx - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = (vi.pz > 0) ? 
                vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] : val_curr;
            float val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0f;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
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
// 辅助函数8：处理YZ边棱cube
// ============================================================================
inline void processEdgeYZCube(
    const float* blockData,
    const float* neighbor_yplus,
    const float* neighbor_zplus,
    const float* neighbor_yz_diagonal,
    size_t local_x,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_y = By - 1;
    size_t local_z = Bx - 1;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_yp = local_x + 0 * Bx + local_z * Bx * By;
    size_t base_zp = local_x + local_y * Bx + 0 * Bx * By;
    size_t base_yz = local_x + 0 * Bx + 0 * Bx * By;
    
    // 顶点分布：0,1在当前块；2,3在Y+邻居；4,5在Z+邻居；6,7在YZ对角
    std::array<float, 8> cubeValues = {{
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
    
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // ========== ✅ 正确计算梯度 ==========
    std::array<std::array<float, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        const float* dataPtr;
        const float* neighbor_y;
        const float* neighbor_z;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        // 顶点0: 当前块 (x, By-1, Bx-1)
        {local_x, local_y, local_z, blockData, neighbor_yplus, neighbor_zplus},
        // 顶点1: 当前块 (x+1, By-1, Bx-1)
        {local_x+1, local_y, local_z, blockData, neighbor_yplus, neighbor_zplus},
        // 顶点2: Y+邻居 (x+1, 0, Bx-1)
        {local_x+1, 0, local_z, neighbor_yplus, blockData, neighbor_yz_diagonal},
        // 顶点3: Y+邻居 (x, 0, Bx-1)
        {local_x, 0, local_z, neighbor_yplus, blockData, neighbor_yz_diagonal},
        // 顶点4: Z+邻居 (x, By-1, 0)
        {local_x, local_y, 0, neighbor_zplus, neighbor_yz_diagonal, blockData},
        // 顶点5: Z+邻居 (x+1, By-1, 0)
        {local_x+1, local_y, 0, neighbor_zplus, neighbor_yz_diagonal, blockData},
        // 顶点6: YZ对角 (x+1, 0, 0)
        {local_x+1, 0, 0, neighbor_yz_diagonal, neighbor_zplus, neighbor_yplus},
        // 顶点7: YZ对角 (x, 0, 0)
        {local_x, 0, 0, neighbor_yz_diagonal, neighbor_zplus, neighbor_yplus}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<float, 3> grad;
        
        // === X方向梯度（块内计算） ===
        if (vi.px == 0) {
            grad[0] = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                     vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Y方向梯度（需要跨块） ===
        if (vi.py == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            float val_up = (vi.py + 1 < By) ? 
                vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By] : val_curr;
            grad[1] = (val_down - val_up) / 2.0f;
        } else if (vi.py == By - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = (vi.py > 0) ? 
                vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] : val_curr;
            float val_up = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0f;
        } else {
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Z方向梯度（需要跨块） ===
        if (vi.pz == 0) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            float val_front = (vi.pz + 1 < Bx) ? 
                vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By] : val_curr;
            grad[2] = (val_back - val_front) / 2.0f;
        } else if (vi.pz == Bx - 1) {
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = (vi.pz > 0) ? 
                vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] : val_curr;
            float val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0f;
        } else {
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
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
// 辅助函数9：处理XYZ角点cube
// ============================================================================
inline void processCornerXYZCube(
    const float* blockData,
    const float* neighbor_xplus,
    const float* neighbor_yplus,
    const float* neighbor_zplus,
    const float* neighbor_xy_diagonal,
    const float* neighbor_xz_diagonal,
    const float* neighbor_yz_diagonal,
    const float* neighbor_xyz_diagonal,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = Bx - 1;
    size_t local_y = By - 1;
    size_t local_z = Bx - 1;
    
    // 8个块的对应索引
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_xp = 0 + local_y * Bx + local_z * Bx * By;
    size_t base_yp = local_x + 0 * Bx + local_z * Bx * By;
    size_t base_zp = local_x + local_y * Bx + 0 * Bx * By;
    size_t base_xy = 0 + 0 * Bx + local_z * Bx * By;
    size_t base_xz = 0 + local_y * Bx + 0 * Bx * By;
    size_t base_yz = local_x + 0 * Bx + 0 * Bx * By;
    size_t base_xyz = 0 + 0 * Bx + 0 * Bx * By;
    
    // ========== 8个顶点分别在8个不同的块中 ==========
    std::array<float, 8> cubeValues = {{
        blockData[base_curr],                  // 顶点0: (Bx-1, By-1, Bx-1) 当前块
        neighbor_xplus[base_xp],               // 顶点1: (0, By-1, Bx-1) X+邻居
        neighbor_xy_diagonal[base_xy],         // 顶点2: (0, 0, Bx-1) XY对角
        neighbor_yplus[base_yp],               // 顶点3: (Bx-1, 0, Bx-1) Y+邻居
        neighbor_zplus[base_zp],               // 顶点4: (Bx-1, By-1, 0) Z+邻居
        neighbor_xz_diagonal[base_xz],         // 顶点5: (0, By-1, 0) XZ对角
        neighbor_xyz_diagonal[base_xyz],       // 顶点6: (0, 0, 0) XYZ对角
        neighbor_yz_diagonal[base_yz]          // 顶点7: (Bx-1, 0, 0) YZ对角
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // ========== ✅ 正确计算梯度，处理8个块 ==========
    std::array<std::array<float, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;           // 顶点在其所属块中的局部坐标
        const float* dataPtr;         // 顶点所在块的数据指针
        const float* neighbor_x;      // X方向邻居块（用于X梯度）
        const float* neighbor_y;      // Y方向邻居块（用于Y梯度）
        const float* neighbor_z;      // Z方向邻居块（用于Z梯度）
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        // 顶点0: 当前块 (Bx-1, By-1, Bx-1)
        {local_x, local_y, local_z, blockData, 
         neighbor_xplus, neighbor_yplus, neighbor_zplus},
        
        // 顶点1: X+邻居 (0, By-1, Bx-1)
        {0, local_y, local_z, neighbor_xplus, 
         blockData, neighbor_xy_diagonal, neighbor_xz_diagonal},
        
        // 顶点2: XY对角 (0, 0, Bx-1)
        {0, 0, local_z, neighbor_xy_diagonal, 
         neighbor_yplus, neighbor_xplus, neighbor_xyz_diagonal},
        
        // 顶点3: Y+邻居 (Bx-1, 0, Bx-1)
        {local_x, 0, local_z, neighbor_yplus, 
         neighbor_xy_diagonal, blockData, neighbor_yz_diagonal},
        
        // 顶点4: Z+邻居 (Bx-1, By-1, 0)
        {local_x, local_y, 0, neighbor_zplus, 
         neighbor_xz_diagonal, neighbor_yz_diagonal, blockData},
        
        // 顶点5: XZ对角 (0, By-1, 0)
        {0, local_y, 0, neighbor_xz_diagonal, 
         neighbor_zplus, neighbor_xyz_diagonal, neighbor_xplus},
        
        // 顶点6: XYZ对角 (0, 0, 0)
        {0, 0, 0, neighbor_xyz_diagonal, 
         neighbor_yz_diagonal, neighbor_xz_diagonal, neighbor_xy_diagonal},
        
        // 顶点7: YZ对角 (Bx-1, 0, 0)
        {local_x, 0, 0, neighbor_yz_diagonal, 
         neighbor_xyz_diagonal, neighbor_zplus, neighbor_yplus}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        std::array<float, 3> grad;
        
        // === X方向梯度（需要跨块） ===
        if (vi.px == 0) {
            // 在块的左边界
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = vi.neighbor_x[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
            float val_right = (vi.px + 1 < Bx) ? 
                vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By] : val_curr;
            grad[0] = (val_left - val_right) / 2.0f;
        } else if (vi.px == Bx - 1) {
            // 在块的右边界
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_left = (vi.px > 0) ? 
                vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] : val_curr;
            float val_right = vi.neighbor_x[0 + vi.py * Bx + vi.pz * Bx * By];
            grad[0] = (val_left - val_right) / 2.0f;
        } else {
            // 块内部
            grad[0] = (vi.dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      vi.dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Y方向梯度（需要跨块） ===
        if (vi.py == 0) {
            // 在块的下边界
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = vi.neighbor_y[vi.px + (By-1) * Bx + vi.pz * Bx * By];
            float val_up = (vi.py + 1 < By) ? 
                vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By] : val_curr;
            grad[1] = (val_down - val_up) / 2.0f;
        } else if (vi.py == By - 1) {
            // 在块的上边界
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_down = (vi.py > 0) ? 
                vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] : val_curr;
            float val_up = vi.neighbor_y[vi.px + 0 * Bx + vi.pz * Bx * By];
            grad[1] = (val_down - val_up) / 2.0f;
        } else {
            // 块内部
            grad[1] = (vi.dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      vi.dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Z方向梯度（需要跨块） ===
        if (vi.pz == 0) {
            // 在块的后边界
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = vi.neighbor_z[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
            float val_front = (vi.pz + 1 < Bx) ? 
                vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By] : val_curr;
            grad[2] = (val_back - val_front) / 2.0f;
        } else if (vi.pz == Bx - 1) {
            // 在块的前边界
            float val_curr = vi.dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
            float val_back = (vi.pz > 0) ? 
                vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] : val_curr;
            float val_front = vi.neighbor_z[vi.px + vi.py * Bx + 0 * Bx * By];
            grad[2] = (val_back - val_front) / 2.0f;
        } else {
            // 块内部
            grad[2] = (vi.dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      vi.dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
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
//minus方向
inline void processBoundaryXMinusCube(
    const float* blockData,
    const float* neighbor_xminus,
    size_t local_y, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_x = 0;  // 固定在X-边界
    
    // 当前块索引：x=0
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    // X-邻居块索引：x=Bx-1
    size_t base_nbr = (Bx-1) + local_y * Bx + local_z * Bx * By;
    
    // ========== 8个顶点：4个在X-邻居，4个在当前块 ==========
    std::array<float, 8> cubeValues = {{
        neighbor_xminus[base_nbr],                   // 顶点0: X-邻居 (Bx-1, y, z)
        blockData[base_curr],                        // 顶点1: 当前块 (0, y, z)
        blockData[base_curr + Bx],                   // 顶点2: 当前块 (0, y+1, z)
        neighbor_xminus[base_nbr + Bx],              // 顶点3: X-邻居 (Bx-1, y+1, z)
        neighbor_xminus[base_nbr + Bx * By],         // 顶点4: X-邻居 (Bx-1, y, z+1)
        blockData[base_curr + Bx * By],              // 顶点5: 当前块 (0, y, z+1)
        blockData[base_curr + Bx + Bx * By],         // 顶点6: 当前块 (0, y+1, z+1)
        neighbor_xminus[base_nbr + Bx + Bx * By]     // 顶点7: X-邻居 (Bx-1, y+1, z+1)
    }};
    
    int cellCaseId = util::findCaseId(cubeValues, isovalue);
    if (cellCaseId == 0 || cellCaseId == 255) {
        return;
    }
    
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // ========== 梯度计算 ==========
    std::array<std::array<float, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;  // true=在X-邻居块，false=在当前块
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {Bx-1, local_y, local_z, true},           // 0: X-邻居
        {0, local_y, local_z, false},             // 1: 当前块
        {0, local_y+1, local_z, false},           // 2: 当前块
        {Bx-1, local_y+1, local_z, true},         // 3: X-邻居
        {Bx-1, local_y, local_z+1, true},         // 4: X-邻居
        {0, local_y, local_z+1, false},           // 5: 当前块
        {0, local_y+1, local_z+1, false},         // 6: 当前块
        {Bx-1, local_y+1, local_z+1, true}        // 7: X-邻居
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const float* dataPtr = vi.in_neighbor ? neighbor_xminus : blockData;
        
        std::array<float, 3> grad;
        
        // === X方向梯度（跨块）===
        if (vi.in_neighbor) {
            // 在X-邻居（x=Bx-1）
            if (vi.px == Bx - 1) {
                float val_left = (vi.px > 0) ? 
                    neighbor_xminus[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] : 
                    neighbor_xminus[vi.px + vi.py * Bx + vi.pz * Bx * By];
                float val_right = blockData[0 + vi.py * Bx + vi.pz * Bx * By];
                grad[0] = (val_left - val_right) / 2.0f;
            } else {
                grad[0] = (neighbor_xminus[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                          neighbor_xminus[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
            }
        } else {
            // 在当前块（x=0）
            if (vi.px == 0) {
                float val_left = neighbor_xminus[(Bx-1) + vi.py * Bx + vi.pz * Bx * By];
                float val_right = (vi.px + 1 < Bx) ? 
                    blockData[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By] : 
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                grad[0] = (val_left - val_right) / 2.0f;
            } else {
                grad[0] = (blockData[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                          blockData[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
            }
        }
        
        // === Y方向梯度（块内）===
        if (vi.py == 0) {
            grad[1] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By];
        } else if (vi.py == By - 1) {
            grad[1] = dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[1] = (dataPtr[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // === Z方向梯度（块内）===
        if (vi.pz == 0) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By];
        } else if (vi.pz == Bx - 1) {
            grad[2] = dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[2] = (dataPtr[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
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

inline void processBoundaryYMinusCube(
    const float* blockData,
    const float* neighbor_yminus,
    size_t local_x, size_t local_z,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_y = 0;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr = local_x + (By-1) * Bx + local_z * Bx * By;
    
    std::array<float, 8> cubeValues = {{
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
    
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // 梯度计算（结构同X-）
    std::array<std::array<float, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x, By-1, local_z, true},
        {local_x+1, By-1, local_z, true},
        {local_x+1, 0, local_z, false},
        {local_x, 0, local_z, false},
        {local_x, By-1, local_z+1, true},
        {local_x+1, By-1, local_z+1, true},
        {local_x+1, 0, local_z+1, false},
        {local_x, 0, local_z+1, false}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const float* dataPtr = vi.in_neighbor ? neighbor_yminus : blockData;
        
        std::array<float, 3> grad;
        
        // X方向（块内）
        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // Y方向（跨块）
        if (vi.in_neighbor) {
            if (vi.py == By - 1) {
                float val_down = (vi.py > 0) ? 
                    neighbor_yminus[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] : 
                    neighbor_yminus[vi.px + vi.py * Bx + vi.pz * Bx * By];
                float val_up = blockData[vi.px + 0 * Bx + vi.pz * Bx * By];
                grad[1] = (val_down - val_up) / 2.0f;
            } else {
                grad[1] = (neighbor_yminus[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                          neighbor_yminus[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
            }
        } else {
            if (vi.py == 0) {
                float val_down = neighbor_yminus[vi.px + (By-1) * Bx + vi.pz * Bx * By];
                float val_up = (vi.py + 1 < By) ? 
                    blockData[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By] : 
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                grad[1] = (val_down - val_up) / 2.0f;
            } else {
                grad[1] = (blockData[vi.px + (vi.py-1) * Bx + vi.pz * Bx * By] - 
                          blockData[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
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
                      dataPtr[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
        }
        
        cubeGradients[i] = grad;
    }
    
    // 生成三角形（与X-相同）
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
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

inline void processBoundaryZMinusCube(
    const float* blockData,
    const float* neighbor_zminus,
    size_t local_x, size_t local_y,
    size_t Bx, size_t By,
    size_t global_x, size_t global_y, size_t global_z,
    float isovalue,
    std::vector<std::array<float, 3>>& localPoints,
    std::vector<std::array<float, 3>>& localNormals,
    std::vector<std::array<int, 3>>& localTriangles,
    std::unordered_map<size_t, int>& localPointMap,
    int& localPtIdx)
{
    size_t local_z = 0;
    
    size_t base_curr = local_x + local_y * Bx + local_z * Bx * By;
    size_t base_nbr = local_x + local_y * Bx + (Bx-1) * Bx * By;
    
    std::array<float, 8> cubeValues = {{
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
    
    std::array<std::array<float, 3>, 8> cubePositions = {{
        {float(global_x),   float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y),   float(global_z)},
        {float(global_x+1), float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y+1), float(global_z)},
        {float(global_x),   float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y),   float(global_z+1)},
        {float(global_x+1), float(global_y+1), float(global_z+1)},
        {float(global_x),   float(global_y+1), float(global_z+1)}
    }};
    
    // 梯度计算
    std::array<std::array<float, 3>, 8> cubeGradients;
    
    struct VertexInfo {
        size_t px, py, pz;
        bool in_neighbor;
    };
    
    std::array<VertexInfo, 8> vertexInfo = {{
        {local_x, local_y, Bx-1, true},
        {local_x+1, local_y, Bx-1, true},
        {local_x+1, local_y+1, Bx-1, true},
        {local_x, local_y+1, Bx-1, true},
        {local_x, local_y, 0, false},
        {local_x+1, local_y, 0, false},
        {local_x+1, local_y+1, 0, false},
        {local_x, local_y+1, 0, false}
    }};
    
    for (int i = 0; i < 8; ++i) {
        auto& vi = vertexInfo[i];
        const float* dataPtr = vi.in_neighbor ? neighbor_zminus : blockData;
        
        std::array<float, 3> grad;
        
        // X方向（块内）
        if (vi.px == 0) {
            grad[0] = dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By];
        } else if (vi.px == Bx - 1) {
            grad[0] = dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                     dataPtr[vi.px + vi.py * Bx + vi.pz * Bx * By];
        } else {
            grad[0] = (dataPtr[(vi.px-1) + vi.py * Bx + vi.pz * Bx * By] - 
                      dataPtr[(vi.px+1) + vi.py * Bx + vi.pz * Bx * By]) / 2.0f;
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
                      dataPtr[vi.px + (vi.py+1) * Bx + vi.pz * Bx * By]) / 2.0f;
        }
        
        // Z方向（跨块）
        if (vi.in_neighbor) {
            if (vi.pz == Bx - 1) {
                float val_back = (vi.pz > 0) ? 
                    neighbor_zminus[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] : 
                    neighbor_zminus[vi.px + vi.py * Bx + vi.pz * Bx * By];
                float val_front = blockData[vi.px + vi.py * Bx + 0 * Bx * By];
                grad[2] = (val_back - val_front) / 2.0f;
            } else {
                grad[2] = (neighbor_zminus[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                          neighbor_zminus[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
            }
        } else {
            if (vi.pz == 0) {
                float val_back = neighbor_zminus[vi.px + vi.py * Bx + (Bx-1) * Bx * By];
                float val_front = (vi.pz + 1 < Bx) ? 
                    blockData[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By] : 
                    blockData[vi.px + vi.py * Bx + vi.pz * Bx * By];
                grad[2] = (val_back - val_front) / 2.0f;
            } else {
                grad[2] = (blockData[vi.px + vi.py * Bx + (vi.pz-1) * Bx * By] - 
                          blockData[vi.px + vi.py * Bx + (vi.pz+1) * Bx * By]) / 2.0f;
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
                
                float w = (isovalue - cubeValues[v1]) / (cubeValues[v2] - cubeValues[v1]);
                
                std::array<float, 3> newPt = util::interpolate(cubePositions[v1], cubePositions[v2], w);
                std::array<float, 3> newNorm = util::interpolate(cubeGradients[v1], cubeGradients[v2], w);
                
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






// ============================================================================
// 主函数：新版 RunMarchingCubesOnDecompressedBlocks
// 采用方案1：分区域处理，追求最高效率
// ============================================================================
util::TriangleMesh<float> RunMarchingCubesOnDecompressedBlocks(
    const std::vector<std::vector<float>>& decompressedBlocks,
    const std::vector<size_t>& expandGlobalSmallBlockIds,
    const std::vector<size_t>& localGlobalSmallBlockIds,
    float isovalue,
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
    std::unordered_map<size_t, const float*> blockMap;
    blockMap.reserve(expandGlobalSmallBlockIds.size());
    for (size_t i = 0; i < expandGlobalSmallBlockIds.size(); i++) {
        blockMap[expandGlobalSmallBlockIds[i]] = decompressedBlocks[i].data();
    }

    // ========== 步骤2：排序块ID以提升Cache命中率 ==========
    // 相邻块连续处理，减少Cache miss
    std::vector<size_t> sortedBlockIds = expandGlobalSmallBlockIds;
    std::sort(sortedBlockIds.begin(), sortedBlockIds.end());

    // ========== 全局结果容器 ==========
    std::vector<std::array<float, 3>> globalPoints;
    std::vector<std::array<float, 3>> globalNormals;
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
        const float* blockData = blockMap[globalBlockId];
        
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
        const float* neighbor_xplus = getNeighborBlockPointer(globalBlockId, 1, 0, 0, blockMap, blockCountOnEachDim);
        const float* neighbor_yplus = getNeighborBlockPointer(globalBlockId, 0, 1, 0, blockMap, blockCountOnEachDim);
        const float* neighbor_zplus = getNeighborBlockPointer(globalBlockId, 0, 0, 1, blockMap, blockCountOnEachDim);


        //minus方向
        //const float* neighbor_xminus = getNeighborBlockPointer(globalBlockId, -1, 0, 0, blockMap, blockCountOnEachDim);
        //const float* neighbor_yminus = getNeighborBlockPointer(globalBlockId, 0, -1, 0, blockMap, blockCountOnEachDim);
        //const float* neighbor_zminus = getNeighborBlockPointer(globalBlockId, 0, 0, -1, blockMap, blockCountOnEachDim);



        // === 3.4 计算块的有效范围（考虑数据边界）===
        size_t max_x = std::min(Bx, dataShape[0] - global_x_start);
        size_t max_y = std::min(By, dataShape[1] - global_y_start);
        size_t max_z = std::min(Bz, dataShape[2] - global_z_start);

        // === 3.5 块内局部结果容器 ===
        std::vector<std::array<float, 3>> localPoints;
        std::vector<std::array<float, 3>> localNormals;
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
            const float* neighbor_xy = getNeighborBlockPointer(globalBlockId, 1, 1, 0, 
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
            
            const float* neighbor_xz = getNeighborBlockPointer(globalBlockId, 1, 0, 1, 
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
            
            const float* neighbor_yz = getNeighborBlockPointer(globalBlockId, 0, 1, 1, 
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
            const float* neighbor_xy = getNeighborBlockPointer(globalBlockId, 1, 1, 0, 
                                                               blockMap, blockCountOnEachDim);
            const float* neighbor_xz = getNeighborBlockPointer(globalBlockId, 1, 0, 1, 
                                                               blockMap, blockCountOnEachDim);
            const float* neighbor_yz = getNeighborBlockPointer(globalBlockId, 0, 1, 1, 
                                                               blockMap, blockCountOnEachDim);
            const float* neighbor_xyz = getNeighborBlockPointer(globalBlockId, 1, 1, 1, 
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

    return util::TriangleMesh<float>(globalPoints, globalNormals, globalTriangles);
}



void RunAndSaveIsosurfaceMesh(
    const std::vector<std::vector<float>>& decompressedBlocks,
    const std::vector<size_t>& expandGlobalSmallBlockIds,
    const std::vector<size_t>& localGlobalSmallBlockIds,
    float isovalue,
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


        //auto mappingStart = std::chrono::high_resolution_clock::now();
        
        //BlockDataAccessor accessor(decompressedBlocks, stepDataShape, smallBlockShape);
        
        //auto mappingEnd = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> mappingTime = mappingEnd - mappingStart;
        //std::cout << "[Info] Mapping table built in " << mappingTime.count() << " seconds.\n";


        // 启动计时器
        util::Timer timer;
        timer.start();

        // 执行 Marching Cubes 提取
        util::TriangleMesh<float> mesh = RunMarchingCubesOnDecompressedBlocks(
            
            decompressedBlocks,
            expandGlobalSmallBlockIds,
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

        // 写入 VTK
        util::saveTriangleMesh(mesh, outFile.c_str());

    }




int main(int argc, char* argv[]) {

    auto totalStart = std::chrono::high_resolution_clock::now(); 

    MPI_Init(&argc, &argv);

    int mpi_rank = 0;
    int mpi_size = 1;

    std::string inputFileName, variableName, indexDir;
    size_t nDim = 0;
    std::vector<size_t> stepDataShape, blockShape, smallBlockShape;
    size_t beginStepNum = 0, endStepNum = 0, bigBlocksPerStep = 1, smallBlocksPerBig = 1 ,  smallBlockSize = 1 , totalBlocksNumber = 1; 
    std::vector<float> queryRange;
    float relative_error_bound = 1E-3;
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

    for (size_t d = 0; d < nDim; d++) {
        bigBlocksPerStep *= stepDataShape[d] / blockShape[d];             // 每维大块数量
        smallBlocksPerBig *= blockShape[d] / smallBlockShape[d];          // 每维小块数量
        smallBlockSize *= smallBlockShape[d];                             // 每个小块的数据量
    }

    size_t nSteps = 1;
    //小块总数
    totalBlocksNumber = nSteps * bigBlocksPerStep*smallBlocksPerBig;



    if (queryRange.size() != 2) {
        //if (mpi_rank == 0) 
        std::cerr << "Error: Invalid query range. Use --query_range <low> <high>" << std::endl;
        //MPI_Finalize();
        return 1;
    }

    
    std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string subDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +  "_NYXSZ3DEf_index/";

    std::vector<size_t> selectedBlocks;

    auto afterProcessTime = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<float> preProcessTime = afterProcessTime - totalStart;

    std::cerr << "[Rank " << mpi_rank << "[Time1] : preProcessTime: " << preProcessTime.count() << " seconds" << std::endl;

    std::vector<ScidxInterval<float>> allBigBlockIndices = loadBigBlockIndexFile(subDir + "big_block_minmax");

    float globalMin = std::numeric_limits<float>::max();
    float globalMax = std::numeric_limits<float>::lowest();

    for (const auto& interval : allBigBlockIndices) {
        globalMin = std::min(globalMin, interval.low);
        globalMax = std::max(globalMax, interval.high);
    }

    //std::cout << "Global Min: " << globalMin << std::endl;
    //std::cout << "Global Max: " << globalMax << std::endl;

    float error_bound = relative_error_bound * (globalMax - globalMin);
    std::cout << "Computed error_bound: " << error_bound << std::endl;
    
   
    float isovalue = 3000.0f;

    //筛选大块。大块范围值无压缩，无误差，直接处理
    for (size_t i = 0; i < allBigBlockIndices.size(); i++) {
        if (isovalue >= allBigBlockIndices[i].low && isovalue <= allBigBlockIndices[i].high) {
            selectedBlocks.push_back(i);
        }
    }


    std::cout << "Selected Big Blocks Size: " << selectedBlocks.size() << std::endl;
        

    auto afterQueryBigBlock = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<float> readAndQueryBigBlock = afterQueryBigBlock - afterProcessTime;
    std::cerr << "[Rank " << mpi_rank << "[Time2] : read and query bigBlocks: " << readAndQueryBigBlock.count() << " seconds" << std::endl;


    std::vector<size_t> allGlobalSmallBlockIds;


    std::vector<size_t> localBlocks = selectedBlocks;


    //从小块中筛选overlap
    std::vector<size_t> localGlobalSmallBlockIds;
    for (size_t i = 0; i < localBlocks.size(); ++i) {
        std::vector<size_t> globalSmallBlockIDs = process_query_task(
            localBlocks[i], beginStepNum, bigBlocksPerStep,
            queryRange, subDir, error_bound,
            stepDataShape[0], blockShape[0], smallBlockShape[0]);

        localGlobalSmallBlockIds.insert(
            localGlobalSmallBlockIds.end(),
            globalSmallBlockIDs.begin(), globalSmallBlockIDs.end());
    }


   //通过index选出来的快个数
    std::cerr << "[CHECK] index filtered size = " << localGlobalSmallBlockIds.size() << std::endl;


    
    /*std::vector<size_t> allDecompressGlobalSmallBlockIds(totalBlocksNumber);  
    for (size_t i = 0; i < totalBlocksNumber; ++i) {
        allDecompressGlobalSmallBlockIds[i] = i;
    }*/





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

    





    std::vector<std::vector<float>> decompressedTargetOriginalBlockdata;

    //std::string inputFileBaseName = std::filesystem::path(inputFileName).filename().string();
    std::string mySubDir = "/home/nyan/scidx/scidx/" + inputFileBaseName + "_" + std::to_string(beginStepNum) + "_" + std::to_string(endStepNum) + "_" + std::to_string(extraValue) +   "_iso_fixlength_originalDataCompression/";


    /*if (allGlobalSmallBlockIds.size() > totalBlocksNumber * 0.9) {
        // 如果请求的块数比总块数0.9还多，使用 AllRead（更高效）
        decompressedTargetOriginalBlockdata = batchDecompressBlocksAllRead(
            mySubDir,
            allDecompressGlobalSmallBlockIds,
            smallBlockSize,
            totalBlocksNumber,
            error_bound
        );
    } else {*/


        // 否则使用普通版本
        decompressedTargetOriginalBlockdata = batchDecompressBlocks(
            mySubDir,
            expandedBlockIds,
            smallBlockSize,
            totalBlocksNumber,
            error_bound
        );
    //}


    //解压命中的小块中数据
    //std::vector<std::vector<float>> decompressedTargetOriginalBlockdata = batchDecompressBlocks(originalSubDir, allGlobalSmallBlockIds, smallBlockSize, totalBlocksNumber, error_bound);

    //auto finishDecompreeAllOverlapedSamllBlocks = std::chrono::high_resolution_clock::now();

        //计入iso surface
    RunAndSaveIsosurfaceMesh(
        decompressedTargetOriginalBlockdata,         //分块解压数据
        expandedBlockIds,
        localGlobalSmallBlockIds,
        3000.0f,                                     //  isovalue值
        smallBlockShape,                                // block 大小
        stepDataShape,                           // 总volume 尺寸
        "isosurface_gapless_double_1216_1.vtk"                  // 输出ISO surface文件名
    );
    
   
    return 0;
}
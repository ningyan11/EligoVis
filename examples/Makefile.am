AUTOMAKE_OPTIONS = foreign
CPPFLAGS += -I/home/nyan/adios2_gcc8_install/include
AM_CPPFLAGS = -I../scidx/include \
              -I/home/nyan/adios2_gcc8_install/include \
              -I/home/nyan/sz3_install/include \
              -I../miniIsosurface/marchingCubes/serial \
              -I../miniIsosurface/marchingCubes/util \
              -I../miniIsosurface/marchingCubes/mantevoCommon
AM_CXXFLAGS = -std=c++17 -fcommon -O3 -DNDEBUG 
AM_FCFLAGS = -I../scidx/include -I../scidx/

# Define AM_LDFLAGS globally, so it can be modified conditionally
AM_LDFLAGS = 

if OPENMP
AM_CPPFLAGS += -fopenmp
AM_LDFLAGS += -fopenmp
endif

# Add SZ3 library path to AM_LDFLAGS
AM_LDFLAGS += -L/home/nyan/sz3_install/lib64 -lSZ3c

bin_PROGRAMS = test_rb test_avl test_avl_adios test_avl_decompress test_readData test_compress originalData_compress originalData_szpCompress originalData_decompress avl_index_compress avl_index_optimized_compress avl_index_optimized_compress_default avl_2index_compress avl_2index_sz3_zfp_compress avl_1index_hdf5_compress avl_2index_hdf5_compress avl_2index_hdf5_sz3_zfp_compress octree_2index_compress octree_2index_sz3_zfp_compress octree_2index_hdf5_compress octree_2index_hdf5_sz3_zfp_compress compute_iso_distribute compute_iso_distribute_hdf5 avl_index_optimized_binary_compress avl_index_optimized_binary_compress_float originalData_binary_szpCompress  originalData_hdf5_szpCompress originalData_binary_szpDecompress originalData_binary_szpCompress_float originalData_binary_szpDecompress_float avl_query avl_query_default_iso queryRange_percentage avl_query_noIndex avl_query_noIndex_scatterplot avl_query_2index_iso avl_query_1index_iso avl_quey_1index_hdf5_iso avl_quey_2index_hdf5_iso octree_query_2index_iso octree_query_2index_hdf5_iso avl_query_2index_scatterplot avl_query_2index_hdf5_scatterplot octree_query_2index_scatterplot octree_query_2index_hdf5_scatterplot avl_original_scatter avl_query_noIndex_float avl_query_float originalData_szpDecompress originalData_sz3Decompress pearson_corr_adios originalData_zfp_block_Compress originalData_zfp_block_Decompress

test_rb_SOURCES = test_rb_interval_tree.cc
test_rb_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 

test_avl_SOURCES = test_avl_interval_tree.cc
test_avl_LDADD = ../scidx/.libs/libscidx.a -lm 

test_avl_adios_SOURCES = test_avl_interval_tree_adios.cc
test_avl_adios_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 

test_avl_decompress_SOURCES = test_avl_interval_tree_decompress.cc
test_avl_decompress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 

test_readData_SOURCES = test_readData.cc
test_readData_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 

test_compress_SOURCES = test_avl_compress.cc
test_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 

originalData_compress_SOURCES = originalData_compress.cc
originalData_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c

originalData_szpCompress_SOURCES = originalData_szpCompress.cc
originalData_szpCompress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c


originalData_zfp_block_Compress_SOURCES = originalData_zfp_block_Compress.cc
originalData_zfp_block_Compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c

originalData_decompress_SOURCES = originalData_decompress.cc
originalData_decompress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c

avl_index_compress_SOURCES = avl_index_compress.cc
avl_index_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c

avl_index_optimized_compress_SOURCES = avl_index_optimized_compress.cc
avl_index_optimized_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs

avl_index_optimized_compress_default_SOURCES = avl_index_optimized_compress_default.cc
avl_index_optimized_compress_default_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs


avl_2index_compress_SOURCES = avl_2index_compress.cc
avl_2index_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs


avl_2index_sz3_zfp_compress_SOURCES = avl_2index_sz3_zfp_compress.cc
avl_2index_sz3_zfp_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs


avl_1index_hdf5_compress_SOURCES = avl_1index_hdf5_compress.cc
avl_1index_hdf5_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs


avl_2index_hdf5_compress_SOURCES = avl_2index_hdf5_compress.cc
avl_2index_hdf5_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs



avl_2index_hdf5_sz3_zfp_compress_SOURCES = avl_2index_hdf5_sz3_zfp_compress.cc
avl_2index_hdf5_sz3_zfp_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs


octree_2index_compress_SOURCES = octree_2index_compress.cc
octree_2index_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs

octree_2index_sz3_zfp_compress_SOURCES = octree_2index_sz3_zfp_compress.cc
octree_2index_sz3_zfp_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs


octree_2index_hdf5_compress_SOURCES = octree_2index_hdf5_compress.cc
octree_2index_hdf5_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs


octree_2index_hdf5_sz3_zfp_compress_SOURCES = octree_2index_hdf5_sz3_zfp_compress.cc
octree_2index_hdf5_sz3_zfp_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs


compute_iso_distribute_SOURCES = compute_iso_distribute.cc
compute_iso_distribute_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs

compute_iso_distribute_hdf5_SOURCES = compute_iso_distribute_hdf5.cc
compute_iso_distribute_hdf5_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs


avl_index_optimized_binary_compress_SOURCES = avl_index_optimized_binary_compress.cc
avl_index_optimized_binary_compress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs -L/home/nyan/zfp_official_install/lib64 -lzfp

avl_index_optimized_binary_compress_float_SOURCES = avl_index_optimized_binary_compress_float.cc
avl_index_optimized_binary_compress_float_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -lstdc++fs -L/home/nyan/zfp_official_install/lib64 -lzfp


originalData_szpDecompress_SOURCES = originalData_szpDecompress.cc
originalData_szpDecompress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

originalData_sz3Decompress_SOURCES = originalData_sz3Decompress.cc
originalData_sz3Decompress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

originalData_zfp_block_Decompress_SOURCES = originalData_zfp_block_Decompress.cc
originalData_zfp_block_Decompress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp



originalData_binary_szpCompress_SOURCES = originalData_binary_szpCompress.cc
originalData_binary_szpCompress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

originalData_hdf5_szpCompress_SOURCES = originalData_hdf5_szpCompress.cc
originalData_hdf5_szpCompress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp


originalData_binary_szpDecompress_SOURCES = originalData_binary_szpDecompress.cc
originalData_binary_szpDecompress_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp


originalData_binary_szpCompress_float_SOURCES = originalData_binary_szpCompress_float.cc
originalData_binary_szpCompress_float_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

originalData_binary_szpDecompress_float_SOURCES = originalData_binary_szpDecompress_float.cc
originalData_binary_szpDecompress_float_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp


pearson_corr_adios_SOURCES = pearson_corr_adios.cc
pearson_corr_adios_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp





avl_query_SOURCES = \
    avl_query.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_query_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

avl_query_default_iso_SOURCES = \
    avl_query_default_iso.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_query_default_iso_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp


queryRange_percentage_SOURCES = queryRange_percentage.cc
queryRange_percentage_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

avl_query_noIndex_SOURCES = avl_query_noIndex.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_query_noIndex_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

avl_query_noIndex_scatterplot_SOURCES = avl_query_noIndex_scatterplot.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_query_noIndex_scatterplot_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp


avl_query_2index_iso_SOURCES = avl_query_2index_iso.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_query_2index_iso_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

avl_query_1index_iso_SOURCES = avl_query_1index_iso.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_query_1index_iso_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp



avl_quey_1index_hdf5_iso_SOURCES = avl_quey_1index_hdf5_iso.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_quey_1index_hdf5_iso_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

avl_quey_2index_hdf5_iso_SOURCES = avl_quey_2index_hdf5_iso.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_quey_2index_hdf5_iso_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp


octree_query_2index_iso_SOURCES = octree_query_2index_iso.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

octree_query_2index_iso_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

octree_query_2index_hdf5_iso_SOURCES = octree_query_2index_hdf5_iso.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

octree_query_2index_hdf5_iso_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp



avl_query_2index_scatterplot_SOURCES = avl_query_2index_scatterplot.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_query_2index_scatterplot_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

avl_query_2index_hdf5_scatterplot_SOURCES = avl_query_2index_hdf5_scatterplot.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_query_2index_hdf5_scatterplot_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp



octree_query_2index_scatterplot_SOURCES = octree_query_2index_scatterplot.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

octree_query_2index_scatterplot_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

octree_query_2index_hdf5_scatterplot_SOURCES = octree_query_2index_hdf5_scatterplot.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

octree_query_2index_hdf5_scatterplot_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp



avl_original_scatter_SOURCES = avl_original_scatter.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_original_scatter_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp




avl_query_noIndex_float_SOURCES = \
    avl_query_noIndex_float.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_query_noIndex_float_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp

avl_query_float_SOURCES = \
    avl_query_float.cc \
    ../miniIsosurface/marchingCubes/util/Image3D.cpp \
    ../miniIsosurface/marchingCubes/util/Timer.cpp 

avl_query_float_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan/adios2_gcc8_install/lib64 -ladios2_cxx11 -L/home/nyan/sz3_install/lib64 -lSZ3c -L/home/nyan/zfp_official_install/lib64 -lzfp








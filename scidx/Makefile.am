AUTOMAKE_OPTIONS = foreign
CPPFLAGS += -I/home/nyan/adios2_gcc8_install/include

AM_FCFLAGS = -I../scidx/include -I../scidx/

AM_LDFLAGS =
AM_LDFLAGS += -L/home/nyan/sz3_install/lib64 -lSZ3c
AM_LDFLAGS += -L/home/nyan/zfp_official_install/lib64 -lzfp

include_HEADERS = \
  include/scidx_rb_interval_tree.h \
  include/scidx_avl_interval_tree.h \
  include/scidx_octree_interval.h \
  include/scidx_defines.h \
  include/scidx_BytesToolkit.h \
  include/scidx_rw.h \
  include/scidx_utility.h \
  include/scidx.h \
  include/scidx_avl.h \
  include/scidx_octree.h \
  include/scidx_octree_hilbert.h \
  include/scidx_octree_hilbert2.h \
  include/scidx_Huffman.h \
  include/scidx_block_min_max.h

lib_LTLIBRARIES = libscidx.la

AM_CPPFLAGS = \
  -I./include \
  -I/home/nyan/adios2/include \
  -I/home/nyan/sz3_install/include \
  -I/home/nyan/zfp_official_install/include

AM_CXXFLAGS = -O3 -std=c++17 -Wall -fcommon

libscidx_la_LDFLAGS = -version-info 1:0:0 \
  -L/home/nyan/adios2/lib64 \
  -L/home/nyan/sz3_install/lib64 \
  -L/home/nyan/zfp_official_install/lib64 \
  -L../zstd/.libs \
  -lzstd -lm -ladios2_cxx11_mpi -lSZ3c

libscidx_la_LIBADD = \
  ../zstd/.libs/libzstd.a \
  -ladios2_cxx11_mpi -lSZ3c \
  -lzfp

if OPENMP
AM_CXXFLAGS += -fopenmp
libscidx_la_LDFLAGS += -fopenmp
endif

libscidx_la_SOURCES = \
  src/scidx_BytesToolkit.cc \
  src/scidx_rw.cc \
  src/scidx_utility.cc \
  src/scidx.cc \
  src/scidx_avl.cc \
  src/scidx_octree.cc \
  src/scidx_octree_hilbert.cc \
  src/scidx_octree_hilbert2.cc \
  src/scidx_defines.cc \
  src/scidx_Huffman.cc \
  src/scidx_block_min_max.cc
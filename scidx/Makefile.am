
AUTOMAKE_OPTIONS=foreign
include_HEADERS=include/scidx_rb_interval_tree.h include/scidx_defines.h include/scidx_BytesToolkit.h include/scidx_rw.h include/scidx_utility.h include/scidx.h include/scidx_Huffman.h include/scidx_block_min_max.h

lib_LTLIBRARIES=libscidx.la
libscidx_la_CXXFLAGS=-I./include -I../zstd/
libscidx_la_LDFLAGS = -version-info  1:0:0
if OPENMP
libscidx_la_CXXFLAGS+=-fopenmp
libscidx_la_LDFLAGS+=-fopenmp
endif
libscidx_la_SOURCES=src/scidx_BytesToolkit.cc src/scidx_rw.cc src/scidx_utility.cc src/scidx.cc src/scidx_defines.cc src/scidx_Huffman.cc src/scidx_block_min_max.cc


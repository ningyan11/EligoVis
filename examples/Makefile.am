AUTOMAKE_OPTIONS=foreign
AM_CPPFLAGS = -I../scidx/include -I/home/nyan1/ADIOS2/local/include
AM_FCFLAGS = -I../scidx/include -I../scidx/
if OPENMP
AM_CPPFLAGS+=-fopenmp
AM_LDFLAGS=-fopenmp
endif

bin_PROGRAMS=test_rb test_avl test_avl_adios test_avl_decompress


test_rb_SOURCES=test_rb_interval_tree.cc
test_rb_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan1/ADIOS2/local/lib -ladios2_cxx11

test_avl_SOURCES=test_avl_interval_tree.cc
test_avl_LDADD=../scidx/.libs/libscidx.a -lm

test_avl_adios_SOURCES=test_avl_interval_tree_adios.cc
test_avl_adios_LDADD=../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan1/ADIOS2/local/lib -ladios2_cxx11

test_avl_decompress_SOURCES=test_avl_interval_tree_decompress.cc
test_avl_decompress_LDADD=../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd -L/home/nyan1/ADIOS2/local/lib -ladios2_cxx11

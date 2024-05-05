AUTOMAKE_OPTIONS=foreign
AM_CPPFLAGS = -I../scidx/include -I/Users/lwan/Research/software/adios2/master/install/include/
AM_FCFLAGS = -I../scidx/include -I../scidx
if OPENMP
AM_CPPFLAGS+=-fopenmp
AM_LDFLAGS=-fopenmp
endif

bin_PROGRAMS=test_rb test_avl test_avl_adios

test_rb_SOURCES=test_rb_interval_tree.cc
test_rb_LDADD=../scidx/.libs/libscidx.a -lm

test_avl_SOURCES=test_avl_interval_tree.cc
test_avl_LDADD=../scidx/.libs/libscidx.a -lm

test_avl_adios_SOURCES=test_avl_interval_tree_adios.cc
test_avl_adios_LDADD=../scidx/.libs/libscidx.a /Users/lwan/Research/software/adios2/master/install/lib/libadios2_cxx11.dylib -lm
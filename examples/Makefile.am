AUTOMAKE_OPTIONS=foreign
AM_CPPFLAGS = -I../scidx/include
AM_FCFLAGS = -I../scidx/include -I../scidx
if OPENMP
AM_CPPFLAGS+=-fopenmp
AM_LDFLAGS=-fopenmp
endif

bin_PROGRAMS=test_rb test_avl

test_rb_SOURCES=test_rb_interval_tree.cc
test_rb_LDADD=../scidx/.libs/libscidx.a -lm

test_avl_SOURCES=test_avl_interval_tree.cc
test_avl_LDADD=../scidx/.libs/libscidx.a -lm

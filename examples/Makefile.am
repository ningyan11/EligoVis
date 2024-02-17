AUTOMAKE_OPTIONS=foreign
AM_CPPFLAGS = -I../scidx/include
AM_FCFLAGS = -I../scidx/include -I../scidx
if OPENMP
AM_CPPFLAGS+=-fopenmp
AM_LDFLAGS=-fopenmp
endif

bin_PROGRAMS=test

test_SOURCES=test.cc
test_LDADD = ../scidx/.libs/libscidx.a -lm -L../zstd/.libs -lzstd

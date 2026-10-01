module load cpu/0.17.3b
module load gcc/10.2.0/npcyll4
module load openmpi/4.1.1/ygduf2r

export C_INCLUDE_PATH=/home/nyan/zfp_official_install/include:/home/nyan/sz3_install/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/nyan/zfp_official_install/include:/home/nyan/sz3_install/include:$CPLUS_INCLUDE_PATH

./autogen.sh

./configure \
CC=mpicc \
CXX=mpicxx \
CXXFLAGS="-O3 -std=c++17 -Wall -fcommon \
-I/home/nyan/zfp_official_install/include \
-I/home/nyan/sz3_install/include \
-I/home/nyan/enter/include" \
LDFLAGS="-L/home/nyan/zfp_official_install/lib64 -lzfp \
-L/home/nyan/sz3_install/lib64 -lSZ3c \
-L/home/nyan/enter/lib -lhdf5_cpp -lhdf5 \
-Wl,-rpath,/home/nyan/zfp_official_install/lib64 \
-Wl,-rpath,/home/nyan/sz3_install/lib64 \
-Wl,-rpath,/home/nyan/adios2_gcc8_install/lib64 \
-Wl,-rpath,/home/nyan/scidx/scidx/zstd/.libs \
-Wl,-rpath,/home/nyan/enter/lib"

make clean
make -j1

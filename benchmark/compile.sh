export PATH=$PATH:/usr/local/cuda/bin

rm -rf executable
nvcc -std=c++11 gpu_tfm.cu main.cu -o executable -lfftw3 -lfftw3_threads -O3 -DHMC
./executable

export PATH=$PATH:/usr/local/cuda/bin
nvcc -std=c++11 gpu_tfm.cu main.cu -o executable
./executable

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <fftw3.h>
#include <iostream>

#define FLOAT_RAND ((float) rand())/((float) RAND_MAX)

const int combs = 2080;
const int times = 1000;
const int dims = 64;

const int rows = 394;
const int cols = 249;

int rx[combs] = {};
int tx[combs] = {};

float real_result[rows * cols] = {};
float imag_result[rows * cols] = {};

fftw_complex fft_in[combs * times] = {};
fftw_complex fft_out[combs * times] = {};

float real_exp[combs * times] = {};
float imag_exp[combs * times] = {};

float real_result_out[rows * cols] = {};
float imag_result_out[rows * cols] = {};

float tt_weight[combs] = {};

int lookup_idx[rows * cols * dims] = {};
float lookup_amp[rows * cols * dims] = {};

int grid_x = rows;
int grid_y = cols;
int grid_z = 1;
int thread_size = 128;

const int tot_pix = rows * combs;

__global__ void tfm_near_norm(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const int* lookup_ind, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp, const float* tt_weight);

float getElapsedTime(cudaEvent_t start, cudaEvent_t end) {
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    return elapsedTime;
}

void initialize(){
    for(int i = 0; i < combs; ++i){
        tx[i] = (i % 64) + 1;
        rx[i] = (i % 64) + 1;
        tt_weight[i] = FLOAT_RAND;
    }
    for(int i = 0; i < rows * cols; ++i){
        real_result[i] = FLOAT_RAND;
        imag_result[i] = FLOAT_RAND;
    }
    for(int i = 0; i < combs * times; ++i){
        real_exp[i] = FLOAT_RAND;
        imag_exp[i] = FLOAT_RAND;
        fft_in[i] = FLOAT_RAND, FLOAT_RAND};
    }
    for(int i = 0; i < rows * cols * dims; ++i){
        lookup_idx[i] = rows * cols * dims - i;
        lookup_amp[i] = FLOAT_RAND;
    }       
}


int main(){
    initialize();
    dim3 grid_size(ceil(rows * cols / thread_size), 1, 1);
    dim3 block_size(thread_size, 1, 1);

    int* rx_gpu;
    int* tx_gpu;
    float* real_result_gpu;
    float* imag_result_gpu;
    float* real_result_out_gpu;
    float* imag_result_out_gpu;
    float* real_exp_gpu;
    float* imag_exp_gpu;
    float* tt_weight_gpu;
    int* lookup_idx_gpu;
    float* lookup_amp_gpu;

    cudaMalloc((void**)&real_result_gpu, sizeof(real_result));
    cudaMalloc((void**)&imag_result_gpu, sizeof(imag_result));
    cudaMalloc((void**)&real_result_out_gpu, sizeof(real_result_out));
    cudaMalloc((void**)&imag_result_out_gpu, sizeof(imag_result_out));
    cudaMalloc((void**)&real_exp_gpu, sizeof(real_exp));
    cudaMalloc((void**)&imag_exp_gpu, sizeof(imag_exp));
    cudaMalloc((void**)&rx_gpu, sizeof(rx));
    cudaMalloc((void**)&tx_gpu, sizeof(tx));
    cudaMalloc((void**)&tt_weight_gpu, sizeof(tt_weight));
    cudaMalloc((void**)&lookup_idx_gpu, sizeof(lookup_idx));
    cudaMalloc((void**)&lookup_amp_gpu, sizeof(lookup_amp));

    int numFrames = 100;
    float totalElapsedTime = 0.0f;
    int numRuns = 1;  // run times
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    for(int frame = 0; frame < numFrames; ++frame){
        

        cudaEventRecord(start, 0);
        cudaMemcpy(real_result_gpu, real_result, sizeof(real_result), cudaMemcpyHostToDevice);
        cudaMemcpy(imag_result_gpu, imag_result, sizeof(imag_result), cudaMemcpyHostToDevice);
        cudaMemcpy(real_exp_gpu, real_exp, sizeof(real_exp), cudaMemcpyHostToDevice);
        cudaMemcpy(imag_exp_gpu, imag_exp, sizeof(imag_exp), cudaMemcpyHostToDevice);
        cudaMemcpy(rx_gpu, rx, sizeof(rx), cudaMemcpyHostToDevice);
        cudaMemcpy(tx_gpu, tx, sizeof(tx), cudaMemcpyHostToDevice);
        cudaMemcpy(tt_weight_gpu, tt_weight, sizeof(tt_weight), cudaMemcpyHostToDevice);
        cudaMemcpy(lookup_idx_gpu, lookup_idx, sizeof(lookup_idx), cudaMemcpyHostToDevice);
        cudaMemcpy(lookup_amp_gpu, lookup_amp, sizeof(lookup_amp), cudaMemcpyHostToDevice);

        tfm_near_norm<<<grid_size, block_size>>>(real_result_gpu, imag_result_gpu, times, combs, real_exp_gpu, imag_exp_gpu, tx_gpu, rx_gpu, lookup_idx_gpu, tot_pix, grid_x, grid_y, grid_z, lookup_amp_gpu, tt_weight_gpu);
        cudaDeviceSynchronize();
    
        cudaMemcpy(real_result_out, real_result_out_gpu, sizeof(real_result_out), cudaMemcpyDeviceToHost);
        cudaMemcpy(imag_result_out, imag_result_out_gpu, sizeof(imag_result_out), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float elapsedTime = getElapsedTime(start, end);
        totalElapsedTime += elapsedTime;    
    }
    float averageElapsedTime = totalElapsedTime / numRuns / numFrames;
    printf("Average time: %.4f ms\n", averageElapsedTime);
    return 0;
}

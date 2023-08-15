#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <fftw3.h>
#include <cmath>
#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#define FLOAT_RAND (float)((float) rand())/((float) RAND_MAX)


#if defined PREDEFINED
    const int times = 1000;
    const int dims = 64;
    const int combs = 4096;
    const int fft_combs = 4096;
    const int fft_parallel = 4;
    const int rows = 394;
    const int cols = 249;
    const int depths = 1;
#endif
#define LINEAR
#define FMC
const int times = 1400;
const int dims = 64;
const int rows = 1123;
const int cols = 197;
const int depths = 1;
const int combs = 4096;
const int fft_combs = 4096;
const int fft_parallel = 4;

int grid_x = rows;
int grid_y = cols;
int grid_z = depths;
int thread_size = 128;

const int tot_pix = rows * cols * depths;

fftw_complex fft_in[times][fft_combs] = {{{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}}, {{5.0, 0.0}, {6.0, 0.0}, {7.0, 0.0}, {8.0, 0.0}}, {{9.0, 0.0}, {10.0, 0.0}, {11.0, 0.0}, {12.0, 0.0}}};
fftw_complex fft_out[times][fft_combs];
fftw_complex fft_out2[times][fft_combs];

int rx[combs];
int tx[combs];

float real_result[rows * cols * depths];
float imag_result[rows * cols * depths];

float real_exp[combs * times];
float imag_exp[combs * times];

float _time[times];

float real_result_out[rows * cols * depths];
float imag_result_out[rows * cols * depths];

float tt_weight[combs];

float lookup_time[rows * cols * depths * dims];
int   lookup_idx[rows * cols * depths * dims];
float lookup_amp[rows * cols * depths * dims];

#if defined HMC
    float lookup_time_tx[rows * cols * depths * dims];
    int   lookup_idx_tx[rows * cols * depths * dims];
    float lookup_amp_tx[rows * cols * depths * dims];
#endif

#if defined HMC
    #if defined LINEAR
        __global__ void tfm_linear_hmc(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time_tx,const float* lookup_time_rx,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx, const float* lookup_amp_rx, const float* tt_weight);
    #else
        __global__ void tfm_near_hmc(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const int* lookup_ind_tx, const int* lookup_ind_rx,const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx,const float* lookup_amp_rx,const float* tt_weight);
    #endif
#else
    #if defined LINEAR
        __global__ void tfm_linear_norm(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp, const float* tt_weight);
    #else
        __global__ void tfm_near_norm(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const int* lookup_ind, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp, const float* tt_weight);
    #endif
#endif


double getElapsedTime(cudaEvent_t start, cudaEvent_t end) {
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    return elapsedTime;
}

void initialize(){
    for(int i = 0; i < combs; ++i){
        tx[i] = (i % dims) + 1;
        rx[i] = (i % dims) + 1;
        tt_weight[i] = FLOAT_RAND;
    }
    for(int i = 0; i < rows * cols * depths; ++i){
        real_result[i] = 0;
        imag_result[i] = 0;
    }
    for(int i = 0; i < combs * times; ++i){
        real_exp[i] = FLOAT_RAND;
        imag_exp[i] = FLOAT_RAND;
    }
    for(int i = 0; i < times; ++i){
        _time[i] = FLOAT_RAND;
    }
    for(int i = 0; i < rows * cols * depths * dims; ++i){
        lookup_idx[i] = rows * cols * depths * dims - i;
        lookup_amp[i] = FLOAT_RAND;
        lookup_time[i] = FLOAT_RAND;
        #if defined HMC
            lookup_idx_tx[i] = rows * cols * depths * dims - i;
            lookup_amp_tx[i] = FLOAT_RAND;
            lookup_time_tx[i] = FLOAT_RAND;
        #endif        
    }       
}

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
float* lookup_time_gpu;
float* _time_gpu;

#if defined HMC
    int* lookup_idx_tx_gpu;
    float* lookup_amp_tx_gpu;
    float* lookup_time_tx_gpu;
#endif


int main(){
    initialize();
    dim3 grid_size(ceil(rows * cols / thread_size), 1, 1);
    dim3 block_size(thread_size, 1, 1);



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
    cudaMalloc((void**)&lookup_time_gpu, sizeof(lookup_time));
    cudaMalloc((void**)&_time_gpu, sizeof(_time));

    #if defined HMC
        cudaMalloc((void**)&lookup_idx_tx_gpu, sizeof(lookup_idx_tx));
        cudaMalloc((void**)&lookup_amp_tx_gpu, sizeof(lookup_amp_tx));
        cudaMalloc((void**)&lookup_time_tx_gpu, sizeof(lookup_time_tx));
    #endif    

    int numFrames = 100;
    float totalElapsedTime = 0.0f;


    fftw_complex* in = reinterpret_cast<fftw_complex*>(fft_in);
    fftw_complex* out1 = reinterpret_cast<fftw_complex*>(fft_out);
    fftw_plan fft, ifft;
    // create fft plans
    fft  = fftw_plan_dft_1d(fft_combs, in, fft_out[0], FFTW_FORWARD, FFTW_ESTIMATE);
    ifft = fftw_plan_dft_1d(fft_combs, out1, fft_out2[0], FFTW_FORWARD, FFTW_ESTIMATE);


    std::chrono::steady_clock::time_point chrono_start = std::chrono::steady_clock::now();
    for(int frame = 0; frame < numFrames; ++frame){
	    for(int i = 0; i < times; ++i){
            fftw_execute_dft(fft, in + i * fft_combs, fft_out[i]);
        }
        for(int i = 0; i < times; ++i){
            fftw_execute_dft(ifft, out1 + i * fft_combs, fft_out2[i]);
        }
    }
    std::chrono::steady_clock::time_point chrono_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> chrono_duration = chrono_end - chrono_start;
    auto hilbert_per_frame = (chrono_duration.count() / (double) numFrames) / (double) fft_parallel;

    float ss = 0;

    for(int frame = 0; frame < numFrames; ++frame){
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);
        // common
        cudaMemcpy(real_result_gpu, real_result, sizeof(real_result), cudaMemcpyHostToDevice);
        cudaMemcpy(imag_result_gpu, imag_result, sizeof(imag_result), cudaMemcpyHostToDevice);
        cudaMemcpy(real_exp_gpu, real_exp, sizeof(real_exp), cudaMemcpyHostToDevice);
        cudaMemcpy(imag_exp_gpu, imag_exp, sizeof(imag_exp), cudaMemcpyHostToDevice);
        cudaMemcpy(rx_gpu, rx, sizeof(rx), cudaMemcpyHostToDevice);
        cudaMemcpy(tx_gpu, tx, sizeof(tx), cudaMemcpyHostToDevice);
        cudaMemcpy(tt_weight_gpu, tt_weight, sizeof(tt_weight), cudaMemcpyHostToDevice);
        cudaMemcpy(lookup_idx_gpu, lookup_idx, sizeof(lookup_idx), cudaMemcpyHostToDevice);
        cudaMemcpy(lookup_amp_gpu, lookup_amp, sizeof(lookup_amp), cudaMemcpyHostToDevice);
        #if defined LINEAR
            cudaMemcpy(lookup_time_gpu, lookup_time, sizeof(lookup_time), cudaMemcpyHostToDevice);
            cudaMemcpy(_time_gpu, _time, sizeof(_time), cudaMemcpyHostToDevice);
        #endif

        #if defined HMC
            cudaMemcpy(lookup_idx_tx_gpu, lookup_idx_tx, sizeof(lookup_idx_tx), cudaMemcpyHostToDevice);
            cudaMemcpy(lookup_amp_tx_gpu, lookup_amp_tx, sizeof(lookup_amp_tx), cudaMemcpyHostToDevice);
            #if defined LINEAR
                cudaMemcpy(lookup_time_tx_gpu, lookup_time_tx, sizeof(lookup_time_tx), cudaMemcpyHostToDevice);        
            #endif
        #endif
        
        #if defined HMC
            #if defined LINEAR
                // std::cout << "tfm_linear_hmc" << std::endl;
                tfm_linear_hmc<<<grid_size, block_size>>>(real_result_gpu, imag_result_gpu, 
                                                         times, combs, 
                                                         real_exp_gpu, imag_exp_gpu, 
                                                         tx_gpu, rx_gpu, 
                                                         lookup_time_tx_gpu, lookup_time_gpu, 
                                                         _time_gpu, 
                                                         tot_pix, grid_x, grid_y, grid_z, 
                                                         lookup_amp_tx_gpu, lookup_amp_gpu, 
                                                         tt_weight_gpu);
            #else
                // std::cout << "tfm_near_hmc" << std::endl;
                tfm_near_hmc<<<grid_size, block_size>>>(real_result_gpu, imag_result_gpu, 
                                                        times, combs, 
                                                        real_exp_gpu, imag_exp_gpu, 
                                                        tx_gpu, rx_gpu, 
                                                        lookup_idx_gpu, lookup_idx_tx_gpu, 
                                                        tot_pix, grid_x, grid_y, grid_z, 
                                                        lookup_amp_tx_gpu, lookup_amp_gpu, 
                                                        tt_weight_gpu);
            #endif
        #endif
        #if defined FMC
            #if defined LINEAR
                // std::cout << "tfm_linear_fmc" << std::endl;
                tfm_linear_norm<<<grid_size, block_size>>>(real_result_gpu, imag_result_gpu, 
                                                          times, combs, 
                                                          real_exp_gpu, imag_exp_gpu, 
                                                          tx_gpu, rx_gpu, 
                                                          lookup_time_gpu, 
                                                          _time_gpu, 
                                                          tot_pix, grid_x, grid_y, grid_z, 
                                                          lookup_amp_gpu, 
                                                          tt_weight_gpu);

            #else
                // std::cout << "tfm_near_fmc" << std::endl;
                tfm_near_norm<<<grid_size, block_size>>>(real_result_gpu, imag_result_gpu,
                                                        times, combs,
                                                        real_exp_gpu, imag_exp_gpu,
                                                        tx_gpu, rx_gpu,
                                                        lookup_idx_gpu,
                                                        tot_pix, grid_x, grid_y, grid_z,
                                                        lookup_amp_gpu,
                                                        tt_weight_gpu);
            #endif
        #endif
        cudaDeviceSynchronize();

        cudaMemcpy(real_result_out, real_result_out_gpu, sizeof(real_result_out), cudaMemcpyDeviceToHost);
        cudaMemcpy(imag_result_out, imag_result_out_gpu, sizeof(imag_result_out), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        for(int i = 0; i < rows * cols * depths; ++i){
            ss += real_result_out[i] + imag_result_out[i];
        }

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        double elapsedTime = getElapsedTime(start, end);
        totalElapsedTime += elapsedTime;
    }
    double inter_per_frame = totalElapsedTime / numFrames;
    #if defined HMC
        std::string data_format = "HMC";
    #else
        std::string data_format = "FMC";
    #endif
    #if defined LINEAR
        std::string method = "LINEAR";
    #else
        std::string method = "NEAREST";
    #endif
    std::cout << "================================================" << std::endl;
    std::cout << data_format << ' ' << method << ' ' << "Hilbert time (ms): " << hilbert_per_frame  << std::endl;
    std::cout << data_format << ' ' << method << ' ' << "Interploation time (ms): " << inter_per_frame << std::endl;
    std::cout << data_format << ' ' << method << ' ' << "Frame Rates (fps): " << 1000.0 / (hilbert_per_frame + inter_per_frame) << std::endl;
    std::cout << "================================================" << std::endl;
    
    std::ofstream outfile;
    outfile.open("results.txt");
    outfile << hilbert_per_frame << "\n"
            << inter_per_frame << "\n"
            << 1000.0 / (hilbert_per_frame + inter_per_frame) << "\n";
    outfile.close();

    return int(ss);
}


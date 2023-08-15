#include <hls_vector.h>
#include <hls_stream.h>
#include "hls_fft.h"

#include "assert.h"

using namespace hls;

#define DATA_SIZE 4096

const int N_ROWS = 394;
const int N_COLS = 249;
const int N_DIMS = 64;
const int N_COMBS = 2080;
const int N_TIMES = 1000;
const int N_PIXELS = N_ROWS * N_COLS;
const int PIXEL_BATCH = 64;

const int BATCH_SIZE = 16;
const int COMB_BATCH = 16;

typedef uint64_t IMAGE_TYPE;
typedef uint64_t FFT_TYPE;

typedef float real_t;
typedef std::complex<real_t> complex_t;



void read_time_data(const uint64_t* time_data, double time_batch[BATCH_SIZE][N_COMBS], const int offset){
    for(int i = 0; i < BATCH_SIZE; ++i){
        #pragma HLS pipeline
        for(int j = 0; j < N_COMBS; ++j){
            time_batch[i][j] = time_data[i * N_COMBS + j + offset * BATCH_SIZE * N_COMBS];
        }
    }
}



int calculate_rx(const int ii, const int offset){
#pragma HLS inline
    return ii + offset;
}

int calculate_tx(const int ii, const int offset){
#pragma HLS inline
    return ii + offset;    
}

int calculate_ind(const int ii, const int offset){
#pragma HLS inline
    return (ii + offset) & 0xffff;
}

double calculate_unit(const double real_exp[COMB_BATCH][N_TIMES], 
                      const double imag_exp[COMB_BATCH][N_TIMES],
                      const int offset, const int pixel_idx, 
                      const uint64_t el_c[N_DIMS * 3]){
#pragma HLS inline off
    ap_fixed<32, 16> real_buffer[COMB_BATCH];
    ap_fixed<32, 16> imag_buffer[COMB_BATCH];
    for(int ii = 0; ii < COMB_BATCH; ++ii){
#pragma HLS unroll
        int ii_idx = ii + offset;
        int tx_id = calculate_tx(ii, offset);
        int rx_id = calculate_rx(ii, offset);
        int tx_ind = calculate_ind(tx_id, pixel_idx);
        int rx_ind = calculate_ind(rx_id, pixel_idx);
        int index  = tx_ind + rx_ind + 1;
        ap_fixed<32, 16> amp = (el_c[ii] / el_c[tx_ind]) * (el_c[ii] / el_c[rx_ind]);
        // if(index >= 0 && index < N_TIMES){
            real_buffer[ii] = ap_fixed<32, 16>(real_exp[ii][index]) * amp;
            imag_buffer[ii] = ap_fixed<32, 16>(imag_exp[ii][index]) * amp;
        // }else{
        //     real_buffer[ii] = 0;
        //     imag_buffer[ii] = 0;
        // }
    }
    for(int i = 0; i < 8; ++i){
        #pragma HLS unroll
        real_buffer[i] += real_buffer[i+8];
        imag_buffer[i] += imag_buffer[i+8];
    }
    for(int i = 0; i < 4; ++i){
        #pragma HLS unroll
        real_buffer[i] += real_buffer[i+4];
        imag_buffer[i] += imag_buffer[i+4];
    }
    for(int i = 0; i < 2; ++i){
        #pragma HLS unroll
        real_buffer[i] += real_buffer[i+2];
        imag_buffer[i] += imag_buffer[i+2];
    }
    for(int i = 0; i < 1; ++i){
        #pragma HLS unroll
        real_buffer[i] += real_buffer[i+1];
        imag_buffer[i] += imag_buffer[i+1];
    }
    return real_buffer[0] + imag_buffer[0];
}

void process_accum(const double real_exp[COMB_BATCH][N_TIMES], 
                   const double imag_exp[COMB_BATCH][N_TIMES], 
                   IMAGE_TYPE* image_in, IMAGE_TYPE* image_out,
                   const uint64_t el_c[N_DIMS * 3], const int pixels, const int offset){

    for(int pix = 0; pix < N_PIXELS / 8; ++pix){
        IMAGE_TYPE tmp_in = image_in[pix];
        IMAGE_TYPE tmp_out = 0;
        for(int num = 0; num < 8; ++num){
            double res = calculate_unit(real_exp, imag_exp, offset, pix + num * 8, el_c);
            tmp_out = tmp_in + res;
        }
        image_out[pix] = tmp_out;
    }
}
void transpose_output(const double real_exp[BATCH_SIZE][N_COMBS], 
                      const double imag_exp[BATCH_SIZE][N_COMBS],
                      FFT_TYPE* fft_output,
                      const int offset){
    for(int i = 0; i < N_COMBS; ++i){
        #pragma HLS pipeline
        FFT_TYPE tmp;
        for(int j = 0; j < BATCH_SIZE; ++j){
            #pragma HLS unroll
            tmp = real_exp[i][j] + imag_exp[i][j];
        }
        fft_output[i * BATCH_SIZE + offset] = tmp;
    }
}

void read_fft_data(const FFT_TYPE* fft_output, 
                   double comb_batch_real[COMB_BATCH][N_TIMES],
                   double comb_batch_imag[COMB_BATCH][N_TIMES],
                   const int offset){
    for(int comb = 0; comb < COMB_BATCH; ++comb){
        for(int t = 0; t < N_TIMES / 8; ++t){
            #pragma HLS pipeline
            FFT_TYPE tmp = fft_output[comb * N_TIMES + t + offset];
            for(int k = 0; k < 8; ++k){
                #pragma HLS unroll
                comb_batch_real[comb][t * 8 + k] = tmp >> (k << 2);
                comb_batch_imag[comb][t * 8 + k] = tmp >> (k << 1);
            }
        }
    }
}

void parallel_fft(const double time_batch[BATCH_SIZE][N_COMBS], 
                  double real_exp[BATCH_SIZE][N_COMBS],
                  double imag_exp[BATCH_SIZE][N_COMBS]){
    for(int i = 0; i < BATCH_SIZE; ++i){
        for(int j = 0; j < N_COMBS; ++j){
            #pragma HLS pipeline
            real_exp[i][j] = time_batch[i][j] * 1.0;
            imag_exp[i][j] = time_batch[i][j] * 2.0;
        }
    }
}

void read_pixel_batch(const IMAGE_TYPE* image_in, IMAGE_TYPE pixel_buffer[PIXEL_BATCH], 
                      const uint64_t* amp, uint64_t amp_buffer[PIXEL_BATCH][N_DIMS], const int offset){
    for(int i = 0; i < PIXEL_BATCH / 8; ++i){
        #pragma HLS pipeline
        IMAGE_TYPE tmp = image_in[offset + i];
        for(int j = 0; j < 8; ++j){
            pixel_buffer[(i*8)+j] = tmp >> j;
        }
    }
    for(int i = 0; i < PIXEL_BATCH / 8; ++i){
        for(int j = 0; j < N_DIMS; ++j){
            #pragma HLS pipeline
            amp_buffer[i][j] = amp[offset+i + j]
        }
    }
}

void process_accum_pixel(const IMAGE_TYPE pixel_buffer[PIXEL_BATCH], const FFT_TYPE* fft_output){
    double real_exp[COMB_BATCH][N_TIMES]; 
    double imag_exp[COMB_BATCH][N_TIMES];

    for(int i = 0; i < N_COMBS / COMB_BATCH; ++i){
        
    }
}

extern "C" {
void tfm(IMAGE_TYPE* image_in, 
         IMAGE_TYPE* image_out, 
         const uint64_t* time_data,
         FFT_TYPE* fft_output,
         FFT_TYPE* fft_input,
         const uint64_t* config, 
         const int combs, const int dims, const int pixels){
#pragma HLS INTERFACE m_axi port=image_in offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=image_out offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=time_data offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=fft_output offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=fft_input offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=amp offset=slave bundle=gmem1

#pragma HLS INTERFACE s_axilite port=combs bundle=control
#pragma HLS INTERFACE s_axilite port=dims bundle=control
#pragma HLS INTERFACE s_axilite port=pixels bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    static double comb_batch_real[COMB_BATCH][N_TIMES];
    static double comb_batch_imag[COMB_BATCH][N_TIMES];
    static ap_fixed<32, 16> el_c_onchip[N_DIMS * 3];
    #pragma HLS array_partition variable=comb_batch_real type=complete dim=1
    #pragma HLS array_partition variable=comb_batch_imag type=complete dim=1
    #pragma HLS array_partition variable=comb_batch_real type=cyclic dim=2 factor=8
    #pragma HLS array_partition variable=comb_batch_imag type=cyclic dim=2 factor=8
    #pragma HLS array_partition variable=el_c_onchip type=complete dim=0

    for(int i = 0; i < N_DIMS; ++i){
        #pragma HLS pipeline
        uint64_t el_c_tmp = el_c[i];
        el_c_onchip[3 * i + 0] = el_c_tmp >> 64;
        el_c_onchip[3 * i + 1] = el_c_tmp >> 32;
        el_c_onchip[3 * i + 2] = el_c_tmp >> 16;
    }

    for(int idx = 0; idx < N_COMBS / COMB_BATCH; ++idx){
        #pragma HLS dataflow
        read_fft_data(fft_output, comb_batch_real, comb_batch_imag, idx);
        process_accum(comb_batch_real, comb_batch_imag, image_in, image_out, el_c_onchip, pixels, idx);
    }
    // static IMAGE_TYPE pixel_buf[PIXEL_BATCH];
    // static uint64_t amp_buf[PIXEL_BATCH][N_DIMs];

    // for(int idx = 0; idx < N_PIXELS/ PIXEL_BATCH/ 8; ++idx){
    //     //read pixels
    //     read_pixel_batch(image_in, pixel_buf, amp, amp_buf, idx);
    // }
}

}

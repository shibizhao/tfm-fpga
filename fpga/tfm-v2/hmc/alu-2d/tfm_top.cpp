#include "define.h"
#define CA
void read_config(const io_t* config, config_t config_buffer[N_DIMS][IMAGE_DIM]){
    for(int i = 0; i < N_DIMS; ++i){
        #pragma HLS pipeline
        io_t tmp = config[i];
        config_buffer[i][0] = tmp.range(31, 0);
        config_buffer[i][1] = tmp.range(63, 32);
        config_buffer[i][2] = tmp.range(95, 64);
    }
}

void read_batch_fft(const io_t* fft_read, 
                    real_t batch_fft_real[COMB_BATCH][N_TIMES], 
                    imag_t batch_fft_imag[COMB_BATCH][N_TIMES], 
                    const int idx){
    const int offset = idx * COMB_BATCH * N_TIMES / CONFIG_READ_BATCH;
    for(int i = 0; i < COMB_BATCH; ++i){
        for(int j = 0; j < N_TIMES; j += CONFIG_READ_BATCH){
            #pragma HLS pipeline
            io_t tmp = fft_read[offset + j];
            batch_fft_real[i][j]   = (real_t)(tmp.range(31, 0));
            batch_fft_imag[i][j]   = (imag_t)(tmp.range(63, 32));
            batch_fft_real[i][j+1] = (real_t)(tmp.range(95, 64));
            batch_fft_imag[i][j+1] = (imag_t)(tmp.range(127, 96));
        }
    }
}

void read_image(const io_t* image_in, real_t batch_image_real[2][PIXEL_BATCH / 2], imag_t batch_image_imag[2][PIXEL_BATCH / 2], const int idx){
#pragma HLS inline off
    const int offset = idx * PIXEL_BATCH;
    for(int i = 0; i < PIXEL_BATCH / 2; ++i){
        #pragma HLS pipeline
        io_t tmp = image_in[offset + i];
        batch_image_real[0][i] = (real_t)(tmp.range(31, 0));
        batch_image_imag[0][i] = (imag_t)(tmp.range(63, 32));
        batch_image_real[1][i] = (real_t)(tmp.range(95, 64));
        batch_image_imag[1][i] = (imag_t)(tmp.range(127, 96));
    }
}

void process_ca_old(const real_t batch_image_real[2][PIXEL_BATCH / 2], 
             const imag_t batch_image_imag[2][PIXEL_BATCH / 2],
             real_t batch_image_real_out[2][PIXEL_BATCH / 2], 
             imag_t batch_image_imag_out[2][PIXEL_BATCH / 2],
             const real_t batch_fft_real[COMB_BATCH][N_TIMES],
             const imag_t batch_fft_imag[COMB_BATCH][N_TIMES],
             const config_t config_buffer[N_DIMS][IMAGE_DIM],
             const int comb_idx){
#pragma HLS inline off
    static real_t real_buffer[2][COMB_BATCH];
    static imag_t imag_buffer[2][COMB_BATCH];
    #pragma HLS array_partition variable=real_buffer type=complete dim=0
    #pragma HLS array_partition variable=imag_buffer type=complete dim=0

    for(int pixel = 0; pixel < PIXEL_BATCH / 2; ++pixel){
        for(int par_pixel = 0; par_pixel < 2; ++par_pixel){
            #pragma HLS unroll
            for(int comb = 0; comb < COMB_BATCH; ++comb){
                #pragma HLS unroll
                int comb_offset = comb + comb_idx * COMB_BATCH;
                int tx = comb + comb_offset + par_pixel;
                int rx = comb + comb_offset + par_pixel + 1;
                int tx_index = tx & 0xffff;
                int rx_index = rx & 0xffff;
                real_t real_factor = (real_t) hls::log((real_t)((config_buffer[comb][0] / config_buffer[comb][1]) * (config_buffer[comb][2] / config_buffer[comb][3])));
                imag_t imag_factor = (imag_t) hls::sqrt((imag_t)((config_buffer[comb][0] / config_buffer[comb][1]) * (config_buffer[comb][2] / config_buffer[comb][3])));
                real_buffer[par_pixel][comb] = batch_fft_real[comb + comb_offset][tx_index] * real_factor;
                imag_buffer[par_pixel][comb] = batch_fft_imag[comb + comb_offset][rx_index] * imag_factor;
            }

            // for(int idx = 0; idx < 16; ++idx){
            //     #pragma HLS unroll
            //     real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 8];
            //     imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 8];
            // }

            for(int idx = 0; idx < 8; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 8];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 8];
            }
            for(int idx = 0; idx < 4; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 4];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 4];
            }
            for(int idx = 0; idx < 2; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 2];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 2];
            }
            for(int idx = 0; idx < 1; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 1];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 1];
            }
            batch_image_real_out[par_pixel][pixel] = real_buffer[0][par_pixel] + batch_image_real[par_pixel][pixel];
            batch_image_imag_out[par_pixel][pixel] = imag_buffer[0][par_pixel] + batch_image_imag[par_pixel][pixel];        
        }
    }
}


void process_ca(real_t batch_image_real_out[2][PIXEL_BATCH / 2], 
                imag_t batch_image_imag_out[2][PIXEL_BATCH / 2],
                const real_t batch_fft_real[COMB_BATCH][N_TIMES],
                const imag_t batch_fft_imag[COMB_BATCH][N_TIMES],
                const config_t config_buffer[N_DIMS][IMAGE_DIM],
                const int comb_idx){
#pragma HLS inline off
    static real_t real_buffer[2][COMB_BATCH];
    static imag_t imag_buffer[2][COMB_BATCH];
    #pragma HLS array_partition variable=real_buffer type=complete dim=0
    #pragma HLS array_partition variable=imag_buffer type=complete dim=0

    for(int pixel = 0; pixel < PIXEL_BATCH / 2; ++pixel){
        for(int par_pixel = 0; par_pixel < 2; ++par_pixel){
            #pragma HLS unroll
            for(int comb = 0; comb < COMB_BATCH; ++comb){
                #pragma HLS unroll
                int comb_offset = comb + comb_idx * COMB_BATCH;
                int tx = comb + comb_offset + par_pixel;
                int rx = comb + comb_offset + par_pixel + 1;
                int tx_index = tx & 0xffff;
                int rx_index = rx & 0xffff;
                real_t real_factor = (real_t) hls::log((real_t)((config_buffer[comb][0] / config_buffer[comb][1]) * (config_buffer[comb][2] / config_buffer[comb][3])));
                imag_t imag_factor = (imag_t) hls::sqrt((imag_t)((config_buffer[comb][0] / config_buffer[comb][1]) * (config_buffer[comb][2] / config_buffer[comb][3])));
                real_buffer[par_pixel][comb] = batch_fft_real[comb + comb_offset][tx_index] * real_factor;
                imag_buffer[par_pixel][comb] = batch_fft_imag[comb + comb_offset][rx_index] * imag_factor;
            }

            for(int idx = 0; idx < 16; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 8];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 8];
            }

            for(int idx = 0; idx < 8; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 8];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 8];
            }
            for(int idx = 0; idx < 4; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 4];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 4];
            }
            for(int idx = 0; idx < 2; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 2];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 2];
            }
            for(int idx = 0; idx < 1; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 1];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 1];
            }
            batch_image_real_out[par_pixel][pixel] = real_buffer[0][par_pixel];
            batch_image_imag_out[par_pixel][pixel] = imag_buffer[0][par_pixel];        
        }
    }
}



void process_pa(real_t batch_image_real_out[2][PIXEL_BATCH / 2], 
                imag_t batch_image_imag_out[2][PIXEL_BATCH / 2],
                const real_t batch_fft_real[COMB_BATCH][N_TIMES],
                const imag_t batch_fft_imag[COMB_BATCH][N_TIMES],
                const config_t config_buffer[N_DIMS][IMAGE_DIM],
                const int comb_idx){
#pragma HLS inline off
    static real_t real_buffer[2][COMB_BATCH];
    static imag_t imag_buffer[2][COMB_BATCH];
    #pragma HLS array_partition variable=real_buffer type=complete dim=0
    #pragma HLS array_partition variable=imag_buffer type=complete dim=0

    for(int pixel = 0; pixel < PIXEL_BATCH / 2; ++pixel){
        for(int par_pixel = 0; par_pixel < 2; ++par_pixel){
            #pragma HLS unroll
            for(int comb = 0; comb < COMB_BATCH; ++comb){
                #pragma HLS unroll
                int comb_offset = comb + comb_idx * COMB_BATCH;
                int tx = comb + comb_offset + par_pixel;
                int rx = comb + comb_offset + par_pixel + 1;
                int tx_index = tx & 0xffff;
                int rx_index = rx & 0xffff;
                real_t real_factor = (real_t) hls::log((real_t)((config_buffer[comb][0] / config_buffer[comb][1]) * (config_buffer[comb][2] / config_buffer[comb][3])));
                imag_t imag_factor = (imag_t) hls::sqrt((imag_t)((config_buffer[comb][0] / config_buffer[comb][1]) * (config_buffer[comb][2] / config_buffer[comb][3])));
                real_buffer[par_pixel][comb] = batch_fft_real[comb + comb_offset][tx_index] * real_factor;
                imag_buffer[par_pixel][comb] = batch_fft_imag[comb + comb_offset][rx_index] * imag_factor;
            }

            for(int idx = 0; idx < 32; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 32];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 32];
            }

            for(int idx = 0; idx < 16; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 16];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 16];
            }

            for(int idx = 0; idx < 8; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 8];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 8];
            }
            for(int idx = 0; idx < 4; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 4];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 4];
            }
            for(int idx = 0; idx < 2; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 2];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 2];
            }
            for(int idx = 0; idx < 1; ++idx){
                #pragma HLS unroll
                real_buffer[par_pixel][idx] += real_buffer[par_pixel][idx + 1];
                imag_buffer[par_pixel][idx] += imag_buffer[par_pixel][idx + 1];
            }
            batch_image_real_out[par_pixel][pixel] = real_buffer[0][par_pixel];
            batch_image_imag_out[par_pixel][pixel] = imag_buffer[0][par_pixel];        
        }
    }
}


void write_image(const real_t real_out_buffer[2][PIXEL_BATCH / 2],
                 const imag_t imag_out_buffer[2][PIXEL_BATCH / 2],
                 io_t* image_out, const int idx){
#pragma HLS inline off
    const int offset = idx * PIXEL_BATCH;
    for(int i = 0; i < PIXEL_BATCH / 2; ++i){
        #pragma HLS pipeline
        io_t tmp = (io_t)(0);
        tmp.range(31, 0) = real_out_buffer[0][i];
        tmp.range(63, 32) = imag_out_buffer[0][i];
        tmp.range(95, 64) = real_out_buffer[1][i];
        tmp.range(127, 96) = imag_out_buffer[1][i];
        image_out[offset + i] = tmp;
    }

}

void ca_interpolation_old(const real_t batch_fft_real[COMB_BATCH][N_TIMES], 
                      const imag_t batch_fft_imag[COMB_BATCH][N_TIMES], 
                      const io_t* image_in, io_t* image_out,
                      const config_t config_buffer[N_DIMS][IMAGE_DIM],
                      const int idx){
#pragma HLS inline off
    static real_t real_in_buffer[2][PIXEL_BATCH / 2];
    static imag_t imag_in_buffer[2][PIXEL_BATCH / 2];
    static real_t real_out_buffer[2][PIXEL_BATCH / 2];
    static imag_t imag_out_buffer[2][PIXEL_BATCH / 2];
    #pragma HLS array_partition variable=real_in_buffer type=complete dim=1
    #pragma HLS array_partition variable=imag_in_buffer type=complete dim=1
    #pragma HLS array_partition variable=real_out_buffer type=complete dim=1
    #pragma HLS array_partition variable=imag_out_buffer type=complete dim=1
    
    for(int i = 0; i < N_PIXELS / PIXEL_BATCH; ++i){
        #pragma HLS dataflow
        read_image(image_in, real_in_buffer, imag_in_buffer, i);
        process_ca_old(real_in_buffer, imag_in_buffer, real_out_buffer, imag_out_buffer, batch_fft_real, batch_fft_imag, config_buffer, idx);
        write_image(real_out_buffer, imag_out_buffer, image_out, i);
    }
}

void ca_interpolation(const real_t batch_fft_real[COMB_BATCH][N_TIMES], 
                      const imag_t batch_fft_imag[COMB_BATCH][N_TIMES], 
                      io_t* image_out,
                      const config_t config_buffer[N_DIMS][IMAGE_DIM],
                      const int idx){
#pragma HLS inline off
    static real_t real_in_buffer[2][PIXEL_BATCH / 2];
    static imag_t imag_in_buffer[2][PIXEL_BATCH / 2];
    static real_t real_out_buffer[2][PIXEL_BATCH / 2];
    static imag_t imag_out_buffer[2][PIXEL_BATCH / 2];
    #pragma HLS array_partition variable=real_in_buffer type=complete dim=1
    #pragma HLS array_partition variable=imag_in_buffer type=complete dim=1
    #pragma HLS array_partition variable=real_out_buffer type=complete dim=1
    #pragma HLS array_partition variable=imag_out_buffer type=complete dim=1
    
    for(int i = 0; i < N_PIXELS / PIXEL_BATCH; ++i){
        #pragma HLS dataflow
        process_ca(real_in_buffer, imag_in_buffer, batch_fft_real, batch_fft_imag, config_buffer, idx);
        write_image(real_in_buffer, imag_in_buffer, image_out, i);
    }
}


void pa_accumulate(const real_t real_image_buffer[2][PIXEL_BATCH / 2], 
                   const imag_t imag_image_buffer[2][PIXEL_BATCH / 2], 
                   io_t* image_out,
                   const int comb_idx,
                   const int pix_idx){
    static real_t internal_real[2][PIXEL_BATCH / 2];
    static imag_t internal_imag[2][PIXEL_BATCH / 2];
    #pragma HLS array_partition variable=internal_real type=complete dim=1
    #pragma HLS array_partition variable=internal_imag type=complete dim=1
    for(int j = 0; j < PIXEL_BATCH / 2; ++j){
        #pragma HLS pipeline
        if(comb_idx == 0){
            internal_real[0][j] = real_image_buffer[0][j];
            internal_imag[0][j] = imag_image_buffer[0][j];
            internal_real[1][j] = real_image_buffer[1][j];
            internal_imag[1][j] = imag_image_buffer[1][j];
        }else if(comb_idx == (N_COMBS / COMB_BATCH - 1)){
            io_t tmp = (io_t)(0);
            tmp.range(31, 0)   = real_image_buffer[0][j];
            tmp.range(63, 32)  = imag_image_buffer[0][j];
            tmp.range(95, 64)  = real_image_buffer[1][j];
            tmp.range(127, 96) = imag_image_buffer[1][j];
            image_out[j + pix_idx * PIXEL_BATCH] = tmp;
        }else{
            internal_real[0][j] += real_image_buffer[0][j];
            internal_imag[0][j] += imag_image_buffer[0][j];
            internal_real[1][j] += real_image_buffer[1][j];
            internal_imag[1][j] += imag_image_buffer[1][j];
        }
    }
}

void pa_interploation(const io_t* fft_read, 
                      io_t* image_out,
                      const config_t config_buffer[N_DIMS][IMAGE_DIM], 
                      const int idx){
#pragma HLS inline off
    static real_t batch_fft_real[COMB_BATCH][N_TIMES];
    static imag_t batch_fft_imag[COMB_BATCH][N_TIMES];
    static real_t real_image_buffer[2][PIXEL_BATCH/2];
    static imag_t imag_image_buffer[2][PIXEL_BATCH/2];
    #pragma HLS array_partition variable=batch_fft_real type=complete dim=1
    #pragma HLS array_partition variable=batch_fft_imag type=complete dim=1
    #pragma HLS array_partition variable=real_image_buffer type=complete dim=1
    #pragma HLS array_partition variable=imag_image_buffer type=complete dim=1

    for(int idx = 0; idx < N_PIXELS / PIXEL_BATCH; ++idx){
        for(int i = 0; i < N_COMBS / COMB_BATCH; ++i){
            #pragma HLS dataflow
            read_batch_fft(fft_read, batch_fft_real, batch_fft_imag, i);
            process_pa(real_image_buffer, imag_image_buffer, batch_fft_real, batch_fft_imag, config_buffer, i);
            pa_accumulate(real_image_buffer, imag_image_buffer, image_out, i, idx);
        }
    }
    
}

void sa_pit_wrapper(const io_t* fft_read, const io_t* image_in, io_t* image_out, 
                    const config_t config_buffer[N_DIMS][IMAGE_DIM]){
    static real_t batch_fft_real[COMB_BATCH][N_TIMES];
    static imag_t batch_fft_imag[COMB_BATCH][N_TIMES];
    #pragma HLS array_partition variable=batch_fft_real type=complete dim=1
    #pragma HLS array_partition variable=batch_fft_imag type=complete dim=1


    for(int idx = 0; idx < N_COMBS / COMB_BATCH; ++idx){
        #pragma HLS dataflow
        read_batch_fft(fft_read, batch_fft_real, batch_fft_imag, idx);
        ca_interpolation_old(batch_fft_real, batch_fft_imag, image_in, image_out, config_buffer, idx);
    }
}

void read_fmc(const io_t* data, real_t batch_fft_in[TIME_BATCH][N_COMBS], const int idx){
    int offset = idx * TIME_BATCH * N_COMBS / FMC_READ_BATCH;
    for(int j = 0; j < N_COMBS; ++j){
        for(int i = 0; i < TIME_BATCH / FMC_READ_BATCH; ++i){
            #pragma HLS pipeline
            io_t tmp = data[offset + j * N_COMBS / FMC_READ_BATCH + i];
            batch_fft_in[i * FMC_READ_BATCH][j]     = tmp.range(31, 0);
            batch_fft_in[i * FMC_READ_BATCH + 1][j] = tmp.range(63, 32);
            batch_fft_in[i * FMC_READ_BATCH + 2][j] = tmp.range(95, 64);
            batch_fft_in[i * FMC_READ_BATCH + 3][j] = tmp.range(127, 96);            
        }
    }
}

void write_fft(const real_t batch_fft_real[TIME_BATCH][N_COMBS], 
               const imag_t batch_fft_imag[TIME_BATCH][N_COMBS], 
               io_t* fft_write, const int idx){
    int offset = idx * TIME_BATCH * N_COMBS / FFT_WRITE_BATCH;
    for(int j = 0; j < N_COMBS; ++j){
        for(int i = 0; i < TIME_BATCH / FFT_WRITE_BATCH; ++i){
            #pragma HLS pipeline
            io_t tmp = (io_t)(0);
            tmp.range(31, 0)   = batch_fft_real[i * FFT_WRITE_BATCH][j];
            tmp.range(63, 32)  = batch_fft_imag[i * FFT_WRITE_BATCH][j];
            tmp.range(95, 64)  = batch_fft_real[i * FFT_WRITE_BATCH + 1][j];
            tmp.range(127, 96) = batch_fft_imag[i * FFT_WRITE_BATCH + 1][j];
            fft_write[offset + j * N_COMBS / FFT_WRITE_BATCH + i] = tmp;          
        }
    }
}

void parallel_fft(const real_t batch_fft_in[TIME_BATCH][N_COMBS], 
                //  real_t batch_fft_real[TIME_BATCH][N_COMBS], 
                //  imag_t batch_fft_imag[TIME_BATCH][N_COMBS]
                  ap_fixed<64, 32> batch_fft_uram[TIME_BATCH][N_COMBS]
                ){
    //const real_t cos_angle[N_COMBS] = {0, 1, 2, 3, 4, 5, 6, 7};
    //const real_t sin_angle[N_COMBS] = {0, 1, 2, 3, 4, 5, 6, 7};
    const ap_fixed<64,32> angle[N_COMBS] = {0, 1, 2, 3, 4, 5, 6, 7};
    #pragma HLS bind_storage variable=angle type=RAM_1P impl=URAM
    //static real_t batch_fft_real_buffer[TIME_BATCH][N_COMBS];
    //static imag_t batch_fft_imag_buffer[TIME_BATCH][N_COMBS];
    static ap_fixed<64, 32> batch_fft_uram_buffer[TIME_BATCH][N_COMBS];
    #pragma HLS bind_storage variable=batch_fft_uram_buffer type=RAM_1P impl=URAM
    #pragma HLS array_partition variable=batch_fft_uram_buffer type=complete dim=1
    //#pragma HLS array_partition variable=batch_fft_real_buffer type=complete dim=1
    //#pragma HLS array_partition variable=batch_fft_imag_buffer type=complete dim=1     
    for(int s = 1; s <= LOG_COMB; ++s){
        int m = 1 << s;
        for(int k = 0; k < N_COMBS; k += m){
            #pragma HLS loop_tripcount min=1 max=2048
            for(int j = 0; j < m / 2; ++j){
                #pragma HLS loop_tripcount min=1 max=2048
                #pragma HLS pipeline
                ap_fixed<64, 32> tmp_angle = angle[k];
                real_t sin_angle = tmp_angle.range(31, 0);
                imag_t cos_angle = tmp_angle.range(63, 32);
                for(int z = 0; z < TIME_BATCH; ++z){
                    #pragma HLS unroll
                    ap_fixed<64, 32> tmp_fft = batch_fft_uram_buffer[z][k];
                    real_t real_tmp = tmp_fft.range(31, 0);
                    imag_t imag_tmp = tmp_fft.range(63, 32);
                    real_tmp += sin_angle * batch_fft_in[z][k];
                    imag_tmp += cos_angle * batch_fft_in[z][k];
                    ap_fixed<64, 32> tmp_fft_res = 0;
                    tmp_fft_res.range(31, 0) = real_tmp;
                    tmp_fft_res.range(63, 32) = imag_tmp;
                    batch_fft_uram_buffer[z][k] = tmp_fft_res;
                }
            }
        }
    }
    for(int k = 0; k < N_COMBS; ++k){
        #pragma HLS pipeline
        for(int z = 0; z < TIME_BATCH; ++z){
            #pragma HLS unroll
            // ap_fixed<64, 32> tmp_fft = batch_fft_uram_buffer[z][k];
            // batch_fft_real[z][k] = tmp_fft.range(31, 0);
            // batch_fft_imag[z][k] = tmp_fft.range(63, 32);
            batch_fft_uram[z][k] = batch_fft_uram_buffer[z][k];
        }
    }
}

void parallel_ifft(
                //    const real_t batch_fft_real_in[TIME_BATCH][N_COMBS], 
                //    const real_t batch_fft_imag_in[TIME_BATCH][N_COMBS], 
                   const ap_fixed<64, 32> batch_fft_uram[TIME_BATCH][N_COMBS], 
                   real_t batch_fft_real[TIME_BATCH][N_COMBS], 
                   imag_t batch_fft_imag[TIME_BATCH][N_COMBS]){
    //const real_t cos_angle[N_COMBS] = {0, 1, 2, 3, 4, 5, 6, 7};
    //const real_t sin_angle[N_COMBS] = {0, 1, 2, 3, 4, 5, 6, 7};
    const ap_fixed<64,32> angle[N_COMBS] = {0, 1, 2, 3, 4, 5, 6, 7};
    #pragma HLS bind_storage variable=angle type=RAM_1P impl=URAM
    //static real_t batch_fft_real_buffer[TIME_BATCH][N_COMBS];
    //static imag_t batch_fft_imag_buffer[TIME_BATCH][N_COMBS];
    static ap_fixed<64, 32> batch_fft_uram_buffer[TIME_BATCH][N_COMBS];
    #pragma HLS bind_storage variable=batch_fft_uram_buffer type=RAM_1P impl=URAM
    #pragma HLS array_partition variable=batch_fft_uram_buffer type=complete dim=1
    //#pragma HLS array_partition variable=batch_fft_real_buffer type=complete dim=1
    //#pragma HLS array_partition variable=batch_fft_imag_buffer type=complete dim=1    
    for(int s = 1; s <= LOG_COMB; ++s){
        int m = 1 << s;
        for(int k = 0; k < N_COMBS; k += m){
            #pragma HLS loop_tripcount min=1 max=2048
            for(int j = 0; j < m / 2; ++j){
                #pragma HLS loop_tripcount min=1 max=2048
                #pragma HLS pipeline
                ap_fixed<64, 32> tmp_angle = angle[k];
                real_t sin_angle = tmp_angle.range(31, 0);
                imag_t cos_angle = tmp_angle.range(63, 32);
                for(int z = 0; z < TIME_BATCH; ++z){
                    #pragma HLS unroll
                    ap_fixed<64, 32> tmp_fft = batch_fft_uram_buffer[z][k];
                    ap_fixed<64, 32> tmp_fft_in = batch_fft_uram[z][k];
                    real_t real_tmp1 = tmp_fft.range(31, 0);
                    imag_t imag_tmp1 = tmp_fft.range(63, 32);
                    real_t real_tmp2 = tmp_fft_in.range(31, 0);
                    imag_t imag_tmp2 = tmp_fft_in.range(63, 32);                    
                    real_tmp1 += sin_angle * real_tmp2;
                    imag_tmp1 += cos_angle * imag_tmp2;
                    ap_fixed<64, 32> tmp_fft_res = 0;
                    tmp_fft_res.range(31, 0) = real_tmp1;
                    tmp_fft_res.range(63, 32) = imag_tmp1;
                    batch_fft_uram_buffer[z][k] = tmp_fft_res;
                }
            }
        }
    }
    for(int k = 0; k < N_COMBS; ++k){
        #pragma HLS pipeline
        for(int z = 0; z < TIME_BATCH; ++z){
            #pragma HLS unroll
            ap_fixed<64, 32> tmp_fft = batch_fft_uram_buffer[z][k];
            batch_fft_real[z][k] = tmp_fft.range(31, 0);
            batch_fft_imag[z][k] = tmp_fft.range(63, 32);
        }
    }
}


void sa_dpp(const io_t* data, io_t* fft_write){
    static real_t batch_fft_in[TIME_BATCH][N_COMBS];
    // static real_t batch_fft_real[TIME_BATCH][N_COMBS];
    // static imag_t batch_fft_imag[TIME_BATCH][N_COMBS];
    static real_t batch_fft_real_out[TIME_BATCH][N_COMBS];
    static imag_t batch_fft_imag_out[TIME_BATCH][N_COMBS];
    #pragma HLS array_partition variable=batch_fft_in type=complete dim=1
    // #pragma HLS array_partition variable=batch_fft_real type=complete dim=1
    // #pragma HLS array_partition variable=batch_fft_imag type=complete dim=1
    #pragma HLS array_partition variable=batch_fft_real_out type=complete dim=1
    #pragma HLS array_partition variable=batch_fft_imag_out type=complete dim=1

    static ap_fixed<64, 32> batch_fft_uram[TIME_BATCH][N_COMBS];
    #pragma HLS array_partition variable=batch_fft_uram type=complete dim=1
    #pragma HLS bind_storage variable=batch_fft_uram type=RAM_1P impl=URAM


    for(int idx = 0; idx < N_TIMES / TIME_BATCH; ++idx){
        #pragma HLS dataflow
        read_fmc(data, batch_fft_in, idx);
        parallel_fft(batch_fft_in, batch_fft_uram);
        parallel_ifft(batch_fft_uram, batch_fft_real_out, batch_fft_imag_out);
        write_fft(batch_fft_real_out, batch_fft_imag_out, fft_write, idx);
    }
}


extern "C"{
    void tfm(const io_t* data, io_t* fft_write, const io_t* fft_read, io_t* image_in, io_t* image_out, io_t* config){
        #pragma HLS interface m_axi port=data bundle=gmem0
        #pragma HLS interface m_axi port=fft_read bundle=gmem1
        #pragma HLS interface m_axi port=fft_write bundle=gmem3
        #pragma HLS interface m_axi port=image_in bundle=gmem2
        #pragma HLS interface m_axi port=image_out bundle=gmem2
        #pragma HLS interface m_axi port=config bundle=gmem1
        #pragma HLS interface s_axilite port=return bundle=control

        // pre-load
        static config_t config_buffer[N_DIMS][IMAGE_DIM];
        #pragma HLS array_partition variable=config_buffer type=complete dim=0

        read_config(fft_read, config_buffer);

        // static int mark[1];

        // fft-data in batch
        #ifdef CA
            for(int i = 0; i < 10; ++i){
                #pragma HLS dataflow
                sa_dpp(data, fft_write);
                sa_pit_wrapper(fft_read, image_in, image_out, config_buffer);
            }
            
        #endif
        #ifdef PA
            static real_t batch_fft_real[COMB_BATCH][N_TIMES];
            static imag_t batch_fft_imag[COMB_BATCH][N_TIMES];
            static real_t real_image_buffer[2][PIXEL_BATCH/2];
            static imag_t imag_image_buffer[2][PIXEL_BATCH/2];
            #pragma HLS array_partition variable=batch_fft_real type=complete dim=1
            #pragma HLS array_partition variable=batch_fft_imag type=complete dim=1
            #pragma HLS array_partition variable=real_image_buffer type=complete dim=1
            #pragma HLS array_partition variable=imag_image_buffer type=complete dim=1

            for(int cnt = 0; cnt < (N_PIXELS / PIXEL_BATCH) * (N_COMBS / COMB_BATCH); ++cnt){
                #pragma HLS dataflow
                int comb_idx = cnt % (N_COMBS / COMB_BATCH);
                int pix_idx  = cnt / (N_COMBS / COMB_BATCH);
                read_batch_fft(fft_read, batch_fft_real, batch_fft_imag, comb_idx);
                process_pa(real_image_buffer, imag_image_buffer, batch_fft_real, batch_fft_imag, config_buffer, comb_idx);
                pa_accumulate(real_image_buffer, imag_image_buffer, image_out, comb_idx, pix_idx);
            }
        #endif


    }
}

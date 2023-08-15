// #define N_COMBS 4096
// #define N_DIMS 64
// #define N_PIXELS 98106

// int generate_tx(int ii, int pix){
//     return (ii + pix) % N_DIMS;
// }

// int generate_rx(int ii, int pix){
//     return ii / N_DIMS;
// }

// void tfm_near_norm(unsigned long long* image_result, const int n, const int combs,
//                    const unsigned long long* exp_data, const unsigned long long* lookup_data, const int tot_pix) {
// #pragma HLS INTERFACE m_axi port=image_result offset=slave bundle=gmem0
// #pragma HLS INTERFACE m_axi port=exp_data offset=slave bundle=gmem1
// #pragma HLS INTERFACE m_axi port=lookup_data offset=slave bundle=gmem2
// #pragma HLS INTERFACE s_axilite port=n bundle=control
// #pragma HLS INTERFACE s_axilite port=combs bundle=control
// #pragma HLS INTERFACE s_axilite port=tot_pix bundle=control
// #pragma HLS INTERFACE s_axilite port=return bundle=control

    

//     for (int pix = 0; pix < tot_pix; pix++) {
// #pragma HLS LOOP_TRIPCOUNT min=98106 max=98106 avg=98106
//         float tot_real0, tot_real1, tot_real2 = 0;
//         for (int ii = 0; ii < combs; ii++) {
//             #pragma HLS LOOP_TRIPCOUNT min=2080 max=2080 avg=2080
//             #pragma HLS PIPELINE II=1
//             float real = 0;
//             float imag = 0;
//             int tx = generate_tx(ii, pix);
//             int rx = generate_rx(ii, pix);
//             int t_ind = tx * tot_pix + pix;
//             int r_ind = rx * tot_pix + pix;

//             unsigned long long tmp1 = lookup_data[t_ind];
//             unsigned long long tmp2 = lookup_data[r_ind];

//             int index = (tmp1 >> 31) + (tmp2 >> 31) + 1;
//             float amp_corr = (tmp1 & 0xffff) * (tmp2 & 0xffff);
//             if (index >= 0 && index < n) {
//                 int set_val = ii * n + index;
//                 unsigned long long real_imag_exp_tmp = exp_data[set_val];
//                 real = real_imag_exp_tmp * amp_corr;
//                 imag = real_imag_exp_tmp * amp_corr;
//             }
//             if(ii % combs == 0){
//                 tot_real0 += real;
//             }else if (ii % combs == 1){
//                 tot_real1 += real;
//             }else{
//                 tot_real2 += real;
//             }
//         }
//         image_result[pix] = tot_real0 + tot_real1 + tot_real2;// + tot_imag[0] + tot_imag[1] + tot_real[2] + tot_imag[2];
//     }
// }

#include <hls_math.h>

#define N_COMBS 4096
#define N_DIMS 64
#define N_PIXELS 98106

int generate_tx(int ii, int pix){
    return (ii + pix) % N_DIMS;
}

int generate_rx(int ii, int pix){
    return ii / N_DIMS;
}

void tfm_near_norm(unsigned long long* image_result, const int n, const int combs,
                   const unsigned long long* exp_data, const unsigned long long* lookup_data, const int tot_pix) {
#pragma HLS INTERFACE m_axi port=image_result offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=exp_data offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=lookup_data offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=combs bundle=control
#pragma HLS INTERFACE s_axilite port=tot_pix bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // 存储优化：使用 ap_fixed 数据类型代替浮点数类型
    ap_fixed<32, 16> tot_real0, tot_real1, tot_real2, tot_real3;
    tot_real0 = tot_real1 = tot_real2 = tot_real3 = 0;

    for (int pix = 0; pix < tot_pix; pix++) {
#pragma HLS LOOP_TRIPCOUNT min=98106 max=98106 avg=98106

        // 循环展开：将内部的 ii 循环展开为 4 个迭代
        for (int ii = 0; ii < combs; ii += 4) {
#pragma HLS LOOP_TRIPCOUNT min=2080 max=2080 avg=2080
#pragma HLS PIPELINE II=1

            // 存储优化：使用 ap_fixed 数据类型代替浮点数类型
            ap_fixed<32, 16> real0, real1, real2, real3, imag0, imag1, imag2, imag3;
            real0 = real1 = real2 = real3 = imag0 = imag1 = imag2 = imag3 = 0;

            // 计算 tx 和 rx
            int tx0 = generate_tx(ii, pix);
            int tx1 = generate_tx(ii + 1, pix);
            int tx2 = generate_tx(ii + 2, pix);
            int tx3 = generate_tx(ii + 3, pix);
            int rx0 = generate_rx(ii, pix);
            int rx1 = generate_rx(ii + 1, pix);
            int rx2 = generate_rx(ii + 2, pix);
            int rx3 = generate_rx(ii + 3, pix);

            // 计算 t_ind 和 r_ind
            int t_ind0 = tx0 * tot_pix + pix;
            int t_ind1 = tx1 * tot_pix + pix;
            int t_ind2 = tx2 * tot_pix + pix;
            int t_ind3 = tx3 * tot_pix + pix;
            int r_ind0 = rx0 * tot_pix + pix;
            int r_ind1 = rx1 * tot_pix + pix;
            int r_ind2 = rx2 * tot_pix + pix;
            int r_ind3 = rx3 * tot_pix + pix;

            // 优化内存访问：使用缓存
            unsigned long long tmp1_0 = lookup_data[t_ind0];
            unsigned long long tmp1_1 = lookup_data[t_ind1];
            unsigned long long tmp1_2 = lookup_data[t_ind2];
            unsigned long long tmp1_3 = lookup_data[t_ind3];
            unsigned long long tmp2_0 = lookup_data[r_ind0];
            unsigned long long tmp2_1 = lookup_data[r_ind1];
            unsigned long long tmp2_2 = lookup_data[r_ind2];
            unsigned long long tmp2_3 = lookup_data[r_ind3];

            // 计算 index 和 amp_corr
            int index0 = (tmp1_0 >> 31) + (tmp2_0 >> 31) + 1;
            int index1 = (tmp1_1 >> 31) + (tmp2_1 >> 31) + 1;
            int index2 = (tmp1_2 >> 31) + (tmp2_2 >> 31) + 1;
            int index3 = (tmp1_3 >> 31) + (tmp2_3 >> 31) + 1;
            float amp_corr0 = (tmp1_0 & 0xffff) * (tmp2_0 & 0xffff);
            float amp_corr1 = (tmp1_1 & 0xffff) * (tmp2_1 & 0xffff);
            float amp_corr2 = (tmp1_2 & 0xffff) * (tmp2_2 & 0xffff);
            float amp_corr3 = (tmp1_3 & 0xffff) * (tmp2_3 & 0xffff);

            // 访问 exp_data
            if (index0 >= 0 && index0 < n) {
                int set_val0 = ii * n + index0;
                unsigned long long real_imag_exp_tmp0 = exp_data[set_val0];
                real0 = real_imag_exp_tmp0 * amp_corr0;
                imag0 = real_imag_exp_tmp0 * amp_corr0;
            }
            if (index1 >= 0 && index1 < n) {
                int set_val1 = (ii + 1) * n + index1;
                unsigned long long real_imag_exp_tmp1 = exp_data[set_val1];
                real1 = real_imag_exp_tmp1 * amp_corr1;
                imag1 = real_imag_exp_tmp1 * amp_corr1;
            }
            if (index2 >= 0 && index2 < n) {
                int set_val2 = (ii + 2) * n + index2;
                unsigned long long real_imag_exp_tmp2 = exp_data[set_val2];
                real2 = real_imag_exp_tmp2 * amp_corr2;
                imag2 = real_imag_exp_tmp2 * amp_corr2;
            }
            if (index3 >= 0 && index3 < n) {
                int set_val3 = (ii + 3) * n + index3;
                unsigned long long real_imag_exp_tmp3 = exp_data[set_val3];
                real3 = real_imag_exp_tmp3 * amp_corr3;
                imag3 = real_imag_exp_tmp3 * amp_corr3;
            }

            // 累加结果
            tot_real0 += real0;
            tot_real1 += real1;
            tot_real2 += real2;
            tot_real3 += real3;
        }

        // 将结果写入输出数组
        image_result[pix] = tot_real0 + tot_real1 + tot_real2 + tot_real3;
    }
}
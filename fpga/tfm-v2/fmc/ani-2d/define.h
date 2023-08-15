#pragma once

#include "hls_stream.h"
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_x_complex.h"
#include "hls_half.h"

#include <hls_vector.h>
#include <hls_stream.h>
#include "hls_fft.h"
#include "hls_math.h"

#include "assert.h"


#define BITWIDTH 32
#define PORTWIDTH 128
#define FFT_READ_BATCH (PORTWIDTH / (BITWIDTH << 1))
#define CONFIG_READ_BATCH (PORTWIDTH / (BITWIDTH))
#define PIXEL_READ_BATCH (PORTWIDTH / (BITWIDTH << 1))
#define FMC_READ_BATCH (PORTWIDTH / (BITWIDTH))
#define FFT_WRITE_BATCH (PORTWIDTH / (BITWIDTH << 1))

typedef ap_uint<PORTWIDTH> io_t;
// fft-related typedef
typedef ap_fixed<BITWIDTH, (BITWIDTH >> 1)> real_t;
typedef real_t imag_t;
// config-related typedef
typedef ap_fixed<BITWIDTH, (BITWIDTH >> 1)> config_t;
// image-related typedef
// typedef ap_fixed<BITWIDTH, (BITWIDTH >> 1)> image_t;


const int N_ROWS = 400;
const int N_COLS = 33;
const int N_DIMS = 128;
const int N_COMBS = N_DIMS * N_DIMS;
const int N_TIMES = 600;
const int N_PIXELS = N_ROWS * N_COLS;
const int IMAGE_DIM = 3;
const int LOG_COMB = 16;

const int PIXEL_BATCH = 2048;
const int COMB_BATCH = 64;
const int TIME_BATCH = 14;




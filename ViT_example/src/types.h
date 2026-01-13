#pragma once
#include <hls_stream.h>
#include <ap_fixed.h>
#include "parameters_Conv_0.h"

// Linear layers weight dimensions
#define M1 16
#define N1 16

// Shared config constants (C++14 friendly)
static const int T = 49;   // sequence length / number of tokens
static const int E = 128;   // input dimension / embedded dimension
static const int O = 128;   // output dimension, should be equal to E
static const int H = 8;    // number of heads
static const int I = E/H;   // head dimension
static const int F = 2*E;   // hidden MLP size
static const int Channels = 1;
static const int Height = 28;
static const int Width = 28;
static const int Patch_size = 4;
static const int N_CLASSES = 10;


typedef ap_fixed<32,16,AP_RND, AP_SAT> act_0;
typedef ap_fixed<32,16,AP_RND, AP_SAT> weight_0;
typedef ap_fixed<32,16,AP_RND,AP_SAT> accumulation;
typedef struct kern_item_c0 {act_0 w[E];} KERN_ITEM_c0;

typedef short ITER;

static const act_0 Head_scale = (act_0)0.25; //sqrt(I)

// Weights and biases: define directly in header (single translation unit in HLS)
// NOTE: 'const' makes them link-safe; 'extern' is not needed if everything
// compiles as one TU (which HLS usually does).
const weight_0 W1[E][E] = { 0 };
const weight_0 B1[E]     = { 0 };

const weight_0 W2[E][F] = { 0 };
const weight_0 B2[F]     = { 0 };

const weight_0 W3[F][E] = { 0 };
const weight_0 B3[E]     = { 0 };

const weight_0 W4[E][N_CLASSES] = { 0 };
const weight_0 B4[N_CLASSES]     = { 0 };


#define in_s_d_c0 Channels
#define in_s_h_c0 Height
#define in_s_w_c0 Width
#define out_s_d_c0 E
#define out_s_h_c0 Height / Patch_size
#define out_s_w_c0 Width / Patch_size

#define kern_s_k_c0 E
#define kern_s_d_c0 Channels
#define kern_s_h_c0 Patch_size
#define kern_s_w_c0 Patch_size

#define stride_h_c0 Patch_size
#define stride_w_c0 Patch_size

#define pad_h_c0 0
#define pad_w_c0 0

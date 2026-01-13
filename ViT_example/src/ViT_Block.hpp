#include "types.h"
#include "activations.h"
#include "linear.h"
#include "ap_fixed.h"
#include "MultiHeadAttention.hpp"
#include "normalizations.hpp"
#include "pe_Conv_0.h"
#include "weight_Conv_0.h"
#include "bias_Conv_0.h"
#include "line_buffer_Conv_0.h"

#include "NORM_1.h"
#include "NORM_2.h"
#include "NORM.h"
#include "OUT_.h"
#include "PATCH.h"
#include "Q_.h"
#include "V_.h"
#include "K_.h"
#include "HEAD_.h"
#include "MLP1.h"
#include "MLP2.h"




void AddNorm(hls::stream<act_0> &residual,hls::stream<act_0> &in, hls::stream<act_0>&out, const act_0 mean[E],
        const act_0 var[E],
        const act_0 weight[E],
        const act_0 bias[E]);

void ViT_Block(hls::stream<act_0> &input, hls::stream<act_0> &output);

void PatchEmbedding(hls::stream<act_0> &input,hls::stream<accumulation>&out);

void FC1(hls::stream<act_0> &in, hls::stream<act_0>&out);

void FC2(hls::stream<act_0> &in, hls::stream<act_0> &out);

void Relu(hls::stream<act_0> &in, hls::stream<act_0> &out);

void MLP(hls::stream<act_0> &input, hls::stream<act_0> &output);

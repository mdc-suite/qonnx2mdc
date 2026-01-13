#include "types.h"
#include "activations.h"
#include "linear.h"
#include "ap_fixed.h"

void MultiHeadAttention(hls::stream<act_0> &Q, hls::stream<act_0> &K, hls::stream<act_0> &V, hls::stream<act_0> &output);

#include "MLP.hpp"
#include "linear.h"
#include "MLP1.h"
#include "MLP2.h"

void FC1(hls::stream<act_0> &in, hls::stream<act_0>&out) {
    printf("--------------MLP1--------------");
    LinearTokenBatch<T,E,F,act_0, weight_0>(in, out, MLP1_W, MLP1_B);
}

void FC2(hls::stream<act_0> &in, hls::stream<act_0> &out) {
    printf("--------------MLP2--------------");
	LinearTokenBatch<T,F,E,act_0, weight_0>(in, out, MLP2_W, MLP2_B);
}

void Relu(hls::stream<act_0> &in, hls::stream<act_0> &out) {
	RELU<T,F,act_0>(in, out);
}

void MLP(hls::stream<act_0> &input, hls::stream<act_0> &output) {
    // 1) Linear layers for Q, K, V
    hls::stream<act_0> fc1, relu;

    FC1(input, fc1);
    Relu(fc1, relu);
    FC2(relu,output);

}

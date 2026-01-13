/* The MultiHeadAttention layer is the dedicated module to
 * perform the Attention-mechanism which characterizes the
 * Transformers networks. It is composed of three Linear
 * layers (Q,K,V): the first two (Q,K) feed a Matmul+Scale+Softmax
 * block, which then feeds a Matmul layer, together with
 * the output of the remaining Linear layer (V).
 */


#include "MultiHeadAttention.hpp"
#include "MLP.hpp"
#include "linear.h"
#include "types.h"
#include "linear_opt.h"


#include "OUT_.h"
#include "Q_.h"
#include "V_.h"
#include "K_.h"

void V_linear(hls::stream<act_0> &in, hls::stream<act_0> (&out)[H]) {
	//printf("------------------V Linear-------------\n");
    LinearTokenBatchReshape<T+1,E,O,H, act_0, weight_0>(in, out, V__W, V__B);
    //printf("----------------------------------------\n");
}

void K_linear(hls::stream<act_0> &in, hls::stream<act_0>(&out)[H]) {
	//printf("------------------K Linear-------------\n");
    LinearTokenBatchReshapeTranspose<T+1,E,O,H, act_0, weight_0>(in, out, K__W, K__B);
    //printf("----------------------------------------\n");
}

void Q_linear(hls::stream<act_0> &in, hls::stream<act_0> (&out)[H]) {
	 //printf("------------------Q Linear-------------\n");
     LinearTokenBatchReshape<T+1,E,O,H, act_0, weight_0>(in, out, Q__W, Q__B);
}

void Out_linear(hls::stream<act_0> &in, hls::stream<act_0> &out) {
	LinearTokenBatch<T+1,E,O, act_0, weight_0>(in, out, OUT__W, OUT__B);
	//printf("----------------------------------------\n");
}





void MultiHeadAttention(hls::stream<act_0> &Q, hls::stream<act_0> &K, hls::stream<act_0> &V, hls::stream<act_0> &output) {
    // 1) Linear layers for Q, K, V
    #pragma HLS INLINE off
	#pragma HLS DATAFLOW
    hls::stream<act_0> Q_1[H];
    hls::stream<act_0> K_1[H];
    hls::stream<act_0> V_1[H];
    hls::stream<act_0> out_linear;
    #pragma HLS STREAM variable=Q_1 depth=150  // Minimum depth = T (65) + margin
    #pragma HLS STREAM variable=K_1 depth=150
    #pragma HLS STREAM variable=V_1 depth=150
    Q_linear(Q, Q_1);
    K_linear(K, K_1);
    V_linear(V, V_1);

    // T is number of tokens
    // E is the embedded dimension
    // H is the number of heads
    // I should be the size of each embedded dimension for each head (E/H)

    // 2) Separate output FIFOs for each head
    hls::stream<act_0> head_out[H];
    #pragma HLS STREAM variable=head_out depth=2100  // Depth >= T*I (65*32=2080)

    // 3) Process heads sequentially

    //Softmax_ITA_fixed
    //Softmax_LUT
    head_split: for (int h = 0; h < H; ++h) {
		#pragma HLS UNROLL
    	Softmax_LUT<T+1, I, T+1, act_0>(
            Q_1[h], K_1[h], V_1[h],
            head_out[h]  // Each head writes to its own FIFO
        );
    }


    // 4) Merge head outputs: token-major order
    head_merge: for (int t = 0; t < T+1; t++) {          // For each token
        for (int h = 0; h < H; h++) {      // For each head
            for (int i = 0; i < I; i++) {  // For each element in head
            	act_0 val = head_out[h].read();
            	out_linear.write(val);
                //out_linear.write(head_out[h].read());
                //printf("MHA -- HEAD %d -- VALUE %f \n", h, (float)val);
            }
        }
    }

    //Here I need the final linear layer
    Out_linear(out_linear, output);



}




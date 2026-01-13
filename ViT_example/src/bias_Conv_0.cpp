
#include <hls_stream.h>
#include <ap_fixed.h>
#include "PATCH.h"

#include "bias_Conv_0.h"
#include "parameters_Conv_0.h"
using namespace hls;

 
void bias_Conv_0(stream <KERN_ITEM_c0> &bias_Conv_0_r){
#pragma HLS AGGREGATE variable=bias_Conv_0_r
#pragma HLS INTERFACE ap_ctrl_none port=return
	
	ITER pout;
	ITER hout;
	ITER wout;
	
	const KERN_ITEM_c0 current_bias = BIAS_PATCH;
	#pragma HLS ARRAY_PARTITION variable=current_bias.w complete dim=1
	
	// Riprende l'ordine dei cicli for dell'attore che fa la convoluzione
	for(hout=0; hout<out_s_h_c0; hout++){
		for(wout=0; wout<out_s_w_c0; wout++){
			bias_Conv_0_r.write(current_bias);
		}
	}
}

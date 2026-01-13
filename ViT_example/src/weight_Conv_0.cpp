
#include <hls_stream.h>
#include <ap_fixed.h>
#include "PATCH.h"
#include "weight_Conv_0.h"
#include "parameters_Conv_0.h"
using namespace hls;

 
void weight_Conv_0(stream <KERN_ITEM_c0> &weight_Conv_0_r){
#pragma HLS AGGREGATE variable=weight_Conv_0_r
#pragma HLS INTERFACE ap_ctrl_none port=return
	
	ITER pout;
	ITER hout;
	ITER wout;
	
	ITER pkern;
	ITER hkern;
	ITER wkern;
	
	KERN_ITEM_c0 current_kern;
	#pragma HLS ARRAY_PARTITION variable=current_kern.w complete dim=1
	
	const KERN_ITEM_c0 weight[kern_s_d_c0][kern_s_h_c0][kern_s_w_c0] = WEIGHT_PATCH;
	
	// Riprende l'ordine dei cicli for dell'attore che fa la convoluzione
	for(hout=0; hout<out_s_h_c0; hout++){
		for(wout=0; wout<out_s_w_c0; wout++){	
			for(hkern=0; hkern < kern_s_h_c0 ; hkern++){
				for(wkern=0; wkern < kern_s_w_c0; wkern++){
					for(pkern=0; pkern < kern_s_d_c0; pkern++){
						#pragma HLS PIPELINE
						current_kern = weight[pkern][hkern][wkern];
						weight_Conv_0_r.write(current_kern);
					}
				}
			}
		}
	}
}

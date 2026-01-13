
#include <hls_stream.h>
#include <ap_fixed.h>
#include "parameters_Conv_0.h"
#include "pe_Conv_0.h"
using namespace hls;

void mac_Conv_0(accumulation* out_val, act_0* current, act_0* kern);

 
void pe_Conv_0(
		stream<act_0> &input_0,
		stream <accumulation> &output_0,
		stream <KERN_ITEM_c0> &weight_Conv_0_r,
		stream <KERN_ITEM_c0> &bias_Conv_0_r		){
	#pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS AGGREGATE variable=weight_Conv_0_r
    #pragma HLS AGGREGATE variable=bias_Conv_0_r
	
	ITER pout;
	ITER hout;
	ITER wout;
	
	ITER pkern;
	ITER hkern;
	ITER wkern;

	ITER init_idx;
	ITER wr_idx;
	
	act_0 current;
	KERN_ITEM_c0 current_bias;
	#pragma HLS ARRAY_PARTITION variable=current_bias.w complete dim=1

	KERN_ITEM_c0 current_kern;
	#pragma HLS ARRAY_PARTITION variable=current_kern.w complete dim=1

	accumulation out_val[out_s_d_c0];
    #pragma HLS ARRAY_PARTITION variable=out_val dim=1 complete
	
	const accumulation cls_token[E] = CLS_PARAM;
	#pragma HLS ARRAY_PARTITION variable=cls_token dim=1 complete

	const accumulation positional[T+1][E] = POS_ENC_PARAM;
	#pragma HLS ARRAY_PARTITION variable=positional dim=2 complete

	Loop_CLS_Token: for (ITER cls_idx = 0; cls_idx < out_s_d_c0; cls_idx++) {
	        #pragma HLS PIPELINE II=1
	        output_0.write(cls_token[cls_idx] + positional[0][cls_idx] );
	    }


	for(hout=0; hout<out_s_h_c0*out_s_w_c0; hout++){
			
			bias_Conv_0_r.read(current_bias);
			Loop_init:for(init_idx=0; init_idx < out_s_d_c0; init_idx++){
				#pragma HLS UNROLL
				out_val[init_idx] = (accumulation) current_bias.w[init_idx];
			};
			
			Loop_conv:for(hkern=0; hkern < kern_s_h_c0 ; hkern++){
				for(wkern=0; wkern < kern_s_w_c0; wkern++){
					Loop_read:for(pkern=0; pkern < kern_s_d_c0; pkern++){
						input_0.read(current);
						weight_Conv_0_r.read(current_kern);	//current_kern = weight[:][pkern][hkern][wkern];
						mac_Conv_0(out_val, &current, current_kern.w);

					}
				}
			}
			
			Loop_wr:for(wr_idx=0; wr_idx < out_s_d_c0; wr_idx++){
				output_0.write(out_val[wr_idx] + positional[hout + 1][wr_idx]);
			}
		}
	}

	
void mac_Conv_0(accumulation* out_val, act_0* current, act_0* kern){
	Inner_loop:for(ITER pout=0; pout < out_s_d_c0; pout++){
			#pragma HLS UNROLL
			out_val[pout] += (accumulation)*current * (accumulation)kern[pout];
	}
}


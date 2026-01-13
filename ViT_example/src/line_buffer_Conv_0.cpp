
#include <hls_stream.h>
//#include <hls_video.h>
#include "my_hls_video.h"
#include <ap_fixed.h>



#include "line_buffer_Conv_0.h"
using namespace hls;



void line_buffer_Conv_0(stream<act_0> &input_0, stream <act_0> &output_0) {
#pragma HLS INTERFACE ap_ctrl_none port=return
	ITER pout;
	ITER hout;
	ITER wout;
	
	ITER pin;	
	ITER hin;
	ITER win;
	
	ITER pkern;
	ITER hkern;
	ITER wkern;

	act_0 in_val;
	act_0 out_val;
	bool out_of_bounds;
	
	LineBuffer<kern_s_h_c0,in_s_w_c0+2*pad_w_c0, act_0> buffer[in_s_d_c0];

	hin = 0;
	win = 0;
	
	for(hout = 0; hout < out_s_h_c0; hout++) {		
		for(wout = 0; wout < out_s_w_c0; wout++) {
Loop_while:while( (win <= (wout * stride_w_c0 + kern_s_w_c0-1)) || (hin < (hout * stride_h_c0 + kern_s_h_c0-1) ) ){
				out_of_bounds = ((hin<pad_h_c0) || (hin>pad_h_c0+in_s_h_c0-1) || (win<pad_w_c0) || (win>pad_w_c0+in_s_w_c0-1))? true : false;
Loop_lettura:for (pin = 0; pin < in_s_d_c0; pin++) {
					if(out_of_bounds){
						in_val=0;
					} else{
						input_0.read(in_val);
					}
					buffer[pin].shift_pixels_up(win);
					buffer[pin].insert_bottom_row(in_val,win);
					}
				// Update input indexes
				if(win < in_s_w_c0-1+2*pad_w_c0){ 
					win++;
					}
				else {
					win = 0; 
					hin++;
					if(hin > (hout * stride_h_c0 + kern_s_h_c0-1) ){
					break;
					}
				}
			}
	
			

		//Now it can write a submatrix
Loop_scrittura:for(hkern=0; hkern < kern_s_h_c0 ; hkern++){
				for(wkern=0; wkern < kern_s_w_c0; wkern++){
	Loop_interno: for(pkern=0; pkern < kern_s_d_c0; pkern++){
					#pragma HLS DEPENDENCE variable=buffer array inter false
					#pragma HLS PIPELINE rewind
						out_val = buffer[pkern].getval(hkern, wout*stride_w_c0 + wkern);
						output_0.write(out_val);
					}
				}
			}
		}
	}
}	
			

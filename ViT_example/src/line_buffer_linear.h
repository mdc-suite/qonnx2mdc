#pragma once
#include <hls_stream.h>
#include <cassert>
#include <limits>
#include "my_hls_video.h"


template<
	int H_size,                    //Matrix1 [A,L]
    int M,                    //Matrix2 [L,C]
	int pad_h,
	int pad_w,
    typename ActType         // Activation (input/output) type
>
void LineBufferLinear(
		hls::stream<ActType> &input,
	    hls::stream<ActType> &output){


	LineBuffer<M,H_size + 2*pad_w, ActType> buffer;








}

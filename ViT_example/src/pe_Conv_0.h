
#ifndef Conv_0_H
#define Conv_0_H
    #include <hls_stream.h>
    #include "types.h"
    using namespace hls;
    void pe_Conv_0(stream<act_0> &input_0,
    		stream <accumulation> &output_0,
			stream <KERN_ITEM_c0> &weight_Conv_0_r,
			stream <KERN_ITEM_c0> &bias_Conv_0_r);
#endif

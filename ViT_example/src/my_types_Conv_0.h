
#ifndef MY_TYPES_Conv_0_S
#define MY_TYPES_Conv_0_S
    #include <ap_fixed.h>
    #include "types.h"
    // types of this layer
    typedef ap_fixed< 32, 16, AP_RND, AP_SAT>  ACT_mac;
    typedef ap_fixed< 32, 16, AP_RND, AP_SAT>  ACT_c0;
    typedef ap_fixed< 8, 4, AP_RND, AP_SAT>  COEFF_c0;
    typedef struct kern_item_c0 {COEFF_c0 w[kern_s_k_c0];} KERN_ITEM_c0;

    // types of previous layer
    typedef ap_fixed<8,4,AP_RND, AP_SAT>  ACT_in;
    typedef short ITER;
#endif

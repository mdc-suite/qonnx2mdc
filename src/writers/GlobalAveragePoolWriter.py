# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

import os
import re
from onnx import helper


from .HLSWriter import  HLSWriter

#buffers
class GlobalAveragePoolWriter(HLSWriter):

    def __init__(self, node, model, init, json_file, file):

        # recover data from reader node
        self.recover_data_from_reader(node, model, init, json_file)

        self.file = file

# -----------------------------------------------------
# METHODS FOR GENERATING CAL FILES

    # write CAL files for this layer
    def write_CAL(self,path):

        # lists containing inputs and output of the layer
        input_list = []
        output_list = []

        input_list.append("input_0")
        output_list.append("output_0")

        self.write_id_file_CAL( input_list, output_list, path)

        '''
        # buffer CAL file creation
        actor_name = "line_buffer_" + self.name
        actor_name = actor_name.replace("GlobalAvgPool","gap")

        input_buffer = "input_0"
        output_buffer = "output_0"

        self.write_line_buffer_file_CAL(actor_name, input_buffer, output_buffer, path, )
        '''
    # write a CAL file related to the layer
    def write_id_file_CAL(self, input_list, output_list, path):

        ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()

        node_name = self.name
        # initializations of the variables that will contain
        # inputs and outputs templates of the layer
        inputs_actor = ""
        outputs_actor = ""

        # template to be filled
        template = \
"""package actors;

actor {}()
    {}

    ==>
    {} :
end"""

        # input template to be filled
        template_input = \
"""
    int(size={}) {}"""

        # output template to be filled
        template_output = \
"""     
    int(size={}) {}"""

        # initialization
        n_bits = ""

        # add all the inputs of the layer
        for elem in input_list:

            

            inputs_actor += template_input.format(ap_fixed_INP_tot, elem)
        
        # add the output size from the input one
        for elem in output_list:

            outputs_actor += template_output.format(ap_fixed_OUT_tot, elem)

        # template merging and creation of the content file
        content_file = template.format(node_name, inputs_actor, outputs_actor)

        # file name
        name_file = self.name + ".cal"

        # file creation
        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # write a CAL file for the line buffer of the layer
    def write_line_buffer_file_CAL(self, buff_name, input_name, output_name, path):

        ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()


        

        template = \
"""package actors;

actor {}()

    int(size={}) {}

    ==>

    int(size={}) {} :
end"""

        # contenuto del file
        content_file = template.format(buff_name ,
                                    ap_fixed_INP_tot, input_name,
                                    ap_fixed_INP_tot, output_name)
        name_file = buff_name + ".cal"

        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

#-----------------------------------------------------
# METHODS FOR GENERATING HLS FILES

    # generate HLS files
    def write_HLS(self, path):

        self.generate_layer_sizes_h_HLS(path)

        self.generate_globalavgpool_h_HLS(path)

        self.generate_globalavgpool_ccp_HLS(path)

        #self.generate_line_buffer_h_HLS(path)

        #self.generate_line_buffer_ccp_HLS(path)

        self.generate_my_types_h(path)

        self.generate_my_hls_video_h(path)

        #self.generate_my_Input_types_h(path)

    #generate layer_size_X.h file
    def generate_layer_sizes_h_HLS(self,path):

        content_file = \
"""
#ifndef LAYER_SIZES_AAA_H
#define LAYER_SIZES_AAA_H

	#define in_s_d_BBB {}
	#define in_s_h_BBB {}
	#define in_s_w_BBB {}
	#define out_s_d_BBB {}
	#define out_s_h_BBB {}
	#define out_s_w_BBB {}
	

#endif     
"""
        number = ''.join(filter(str.isdigit, self.name))
        content_file = content_file.replace("BBB", "gap"+number)
        in_d, in_h, in_w = self.isizes[1:]
        out_d, out_h, out_w, = self.osizes[1:]
      

        content_file = content_file.format(
                                  in_d,in_h,in_w,
                                  out_d, out_h, out_w,
                                  )
        content_file = content_file.replace("AAA", self.name)
        name_file = "layer_sizes_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    #generate X.h file
    def generate_globalavgpool_h_HLS(self,path):

        content_file = \
"""
#ifndef AAA_H
#define AAA_H
	#include <hls_stream.h>
	#include "my_types_AAA.h"
	using namespace hls;
	void AAA(stream<ACT_CCC> &input_0, stream <ACT_BBB> &output_0);
#endif
"""

        content_file = content_file.replace("AAA", self.name)
        number = ''.join(filter(str.isdigit, self.name))
        content_file = content_file.replace("BBB", "gap"+number)

        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")   

        name_file = "{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    #generate X.ccp file
    def generate_globalavgpool_ccp_HLS(self,path):

 
        content_file = \
"""
#include <hls_stream.h>
//#include <hls_video.h>
#include "my_hls_video.h"
#include <ap_fixed.h>

#include "layer_sizes_AAA.h"	
#include "my_types_AAA.h"
using namespace hls;

void AAA(stream<ACT_CCC> &input_0, stream <ACT_BBB> &output_0) {
#pragma HLS INTERFACE ap_ctrl_none port=return
	ITER pout;
	ITER hout;
	ITER wout;
    ITER count;
	
	ITER pin;	
	ITER hin;
	ITER win;
	
	ITER hkern;
	ITER wkern;

	ACT_CCC in_val;
	ACT_BBB out_val;
	
	ACT_mac sum[out_s_d_BBB];  // Ensure initialization happens only once

    for(count = 0; count < out_s_d_BBB; count++){
        sum[count] = 0;
    }

	for(hout = 0; hout < in_s_h_BBB; hout++) {
		for(wout = 0; wout < in_s_w_BBB; wout++) {
			for (pin = 0; pin < in_s_d_BBB; pin++) {
            #pragma HLS PIPELINE rewind
					input_0.read(in_val);

					sum[pin] = sum[pin] + in_val;
					}
				}
			}
	
			

		//Now it can write a submatrix
Loop_scrittura:for(pout=0; pout < out_s_d_BBB ; pout++){

						out_val = (ACT_BBB)(sum[pout]/(in_s_h_BBB * in_s_w_BBB));
						output_0.write(out_val);
					}
    }
"""
     

        content_file = content_file.replace("AAA", self.name)

        number = ''.join(filter(str.isdigit, self.name))
        content_file = content_file.replace("BBB", "gap"+number)

        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}") 

        name_file = "{}.cpp".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    #generate line_buffer_X.h file
    def generate_line_buffer_h_HLS(self,path):

        content_file = \
"""
#ifndef LINE_BUFFER_AAA_H
#define LINE_BUFFER_AAA_H
	#include "my_types_AAA.h"
	#include <hls_stream.h>
	using namespace hls;
	
	void line_buffer_AAA(stream<ACT_CCC> &input_0, stream <ACT_BBB> &output_0);
#endif
"""
        
        content_file = content_file.replace("AAA",self.name)
        content_file = content_file.replace("line_buffer_GlobalAvgPool","line_buffer_gap")
        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}") 


        number = ''.join(filter(str.isdigit, self.name))
        content_file = content_file.replace("BBB", "gap"+number)

        name_file = "line_buffer_{}.h".format(self.name)
        name_file = name_file.replace("GlobalAvgPool","gap")

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

        pass

    #generate line_buffer_X.ccp file
    def generate_line_buffer_ccp_HLS(self, path):

        content_file = \
"""
#include <hls_stream.h>
//#include <hls_video.h>
#include "my_hls_video.h"
#include <ap_fixed.h>

#include "layer_sizes_AAA.h"	
#include "my_types_AAA.h"
#include "line_buffer_AAA.h"
using namespace hls;

void line_buffer_AAA(stream<ACT_CCC> &input_0, stream <ACT_BBB> &output_0) {
#pragma HLS INTERFACE ap_ctrl_none port=return
	ITER pout;
	ITER hout;
	ITER wout;
	
	ITER pin;	
	ITER hin;
	ITER win;
	
	ITER hkern;
	ITER wkern;

	ACT_CCC in_val;
	ACT_BBB out_val;
	
	
    for (pin = 0; pin < in_s_d_BBB; pin++) {
	#pragma HLS PIPELINE rewind
						sum[pin] = 0;
						}

	for(hout = 0; hout < out_s_h_BBB; hout++) {
		for(wout = 0; wout < out_s_w_BBB; wout++) {
			for (pin = 0; pin < in_s_d_BBB; pin++) {
            #pragma HLS PIPELINE rewind
					input_0.read(in_val);

					sum[pin] = sum[pin] + in_val;
					}
				}
			}
	
			

		//Now it can write a submatrix
Loop_scrittura:for(pout=0; pout < out_s_d_BBB ; pout++){

						out_val = sum[pout]/out_s_d_BBB;
						count = count + 1;
						output_0.write(out_val);
					}
    }
"""

        content_file = content_file.replace("AAA",self.name)
        number = ''.join(filter(str.isdigit, self.name))
        content_file = content_file.replace("BBB", "gap"+number)
        content_file = content_file.replace("line_buffer_GlobalAvgPool","line_buffer_gap")
        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}") 


        name_file = "line_buffer_{}.cpp".format(self.name)
        name_file = name_file.replace("GlobalAvgPool","gap")

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)
        pass

    # generate a "my tipes" file
    def generate_my_types_h(self,  path):


        template = \
"""
#ifndef MY_TYPES_AAA_S
#define MY_TYPES_AAA_S
    #include <ap_fixed.h>
    #include "layer_sizes_AAA.h"
    // types of this layer
    typedef ZZZ ACT_mac;
    typedef XXX ACT_BBB;
"""
        template_previous= \
"""
    // types of previous layer
    typedef WWW ACT_CCC;
    typedef short ITER;
#endif
"""
        template_previous_coeff= \
"""
    // types of previous layer
    typedef WWW ACT_CCC;
    typedef JJJ COEFF_CCC;
    typedef short ITER;
#endif
"""
         
        # fill the template
        content_file = template.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        number = "gap"+ number

        content_file = content_file.replace("BBB", number)
        #initialization
        ap_fixed_DATA_tot = ""
        ap_fixed_DATA_int = ""
        ap_fixed_COEFF_tot = ""
        ap_fixed_COEFF_int = ""


        template_ap_fixed = \
"""ap_fixed< BBB, CCC, AP_RND, AP_SAT> """


       #Extract Previous Layer's Types.
        #ap_fixed_DATA_int, ap_fixed_DATA_tot, selector_DATA, ap_fixed_COEFF_int, ap_fixed_COEFF_tot, selector_COEFF = self.get_my_size(bit_size_directives)
        if self.is_Input_prev():
            ap_fixed_INP_int_P , ap_fixed_INP_tot_P,ap_fixed_OUT_int_P , ap_fixed_OUT_tot_P, ap_fixed_COEFF_int_P , ap_fixed_COEFF_tot_P = self.get_my_size(spec_node=str(self.init.net_input))
        else:
            ap_fixed_INP_int_P , ap_fixed_INP_tot_P,ap_fixed_OUT_int_P , ap_fixed_OUT_tot_P, ap_fixed_COEFF_int_P , ap_fixed_COEFF_tot_P = self.get_my_size(spec_node=self.prev_layers[0].name)


        #ap_fixed_DATA_int_prev, ap_fixed_DATA_tot_prev, selector_DATA_prev, ap_fixed_COEFF_int_prev, ap_fixed_COEFF_tot_prev, selector_COEFF_prev = self.get_my_size(bit_size_directives, prev_layer.operation)
        ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()

        # create content file
        content_file = content_file.replace("AAA", self.name)

        
        tmp = template_ap_fixed.replace("BBB", str(ap_fixed_INP_tot)).replace("CCC", str(ap_fixed_INP_int))
        content_file = content_file.replace("XXX",tmp)
        mac_value_tot,mac_value_int = self.get_MAC_size()
        tmp = template_ap_fixed.replace("BBB", str(mac_value_tot)).replace("CCC", str(mac_value_int))
        content_file = content_file.replace("ZZZ",tmp)

        

        

        ##############--PREVIOUS LAYER--############################

        

        #check if previous layer has coeff type
        if  ap_fixed_COEFF_tot_P:

            if self.is_Input_prev() :
                template_previous_coeff = template_previous_coeff.replace("CCC", "in")
            else:
                match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
                first_letters, last_number = match.groups()
                template_previous_coeff = template_previous_coeff.replace("CCC", f"{first_letters[0].lower()}{last_number}")

            tmp = template_ap_fixed.replace("BBB", str(ap_fixed_OUT_tot_P)).replace("CCC", str(ap_fixed_OUT_int_P))
            template_previous_coeff = template_previous_coeff.replace("WWW",tmp)
            tmp = template_ap_fixed.replace("BBB", str(ap_fixed_COEFF_tot_P)).replace("CCC", str(ap_fixed_COEFF_int_P))
            template_previous_coeff = template_previous_coeff.replace("JJJ", tmp)
            content_file = content_file + template_previous_coeff
        else:

            if self.is_Input_prev() :
                template_previous = template_previous.replace("CCC", "in")
            else:
                match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
                first_letters, last_number = match.groups()
                template_previous = template_previous.replace("CCC", f"{first_letters[0].lower()}{last_number}")

            tmp = template_ap_fixed.replace("BBB", str(ap_fixed_OUT_tot_P)).replace("CCC", str(ap_fixed_OUT_int_P))
            template_previous = template_previous.replace("WWW",tmp)
            content_file= content_file + template_previous


        # name of the file
        name_file = "my_types_" + self.name + ".h"

        # file creation
        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

        with open(os.path.join(path, self.file), "a") as new_file:
            new_file.write(content_file)

    def generate_my_hls_video_h(self, path):
        content_file = \
"""


#include <cassert>

#ifndef MY_LB_H
#define MY_LB_H

//#define _HLSCLIB_DEBUG_

#ifdef AESL_SYN
#undef _HLSCLIB_DEBUG_
#endif

typedef  unsigned int HLS_SIZE_T;

namespace hls {


/* Template class of Line Buffer */
template<int ROWS, int COLS, typename T, int RESHAPE=0>
class LineBuffer;

template<int ROWS, int COLS, typename T>
class LineBuffer<ROWS, COLS, T, 0> {
public:
    LineBuffer() {
#pragma HLS array_partition variable=val dim=1 complete
#pragma HLS dependence variable=val inter false
#pragma HLS dependence variable=val intra false
    };
    /* LineBuffer main APIs */
    void shift_pixels_up(int col);
    void insert_bottom_row(T value, int col);

    T& getval(int row, int col);
    T& operator ()(int row, int col);

    T val[ROWS][COLS];
#ifdef _HLSCLIB_DEBUG_
    void restore_val();
    void linebuffer_print(int col);
    T val_t[ROWS][COLS];
#endif
};


/* Member functions of LineBuffer class */

/*
 * LineBuffer content shift down
 * Assumes new values will be placed in top row = 0
 */

/*
 * LineBuffer content shift up
 * Assumes new values will be placed in top row = ROWS-1
 */
template<int ROWS, int COLS, typename T> void LineBuffer<ROWS, COLS, T>::shift_pixels_up(int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);

#ifdef _HLSCLIB_DEBUG_
    std::cout << "LineBuffer Elements in col=" << col << ":";
    linebuffer_print(col);
    restore_val();
#endif

    HLS_SIZE_T i;
    for(i = 0; i < ROWS-1; i++) {
#pragma HLS unroll
        val[i][col] = val[i+1][col];
    }

#ifdef _HLSCLIB_DEBUG_
    std::cout << "===  After " << __FUNCTION__ << ":  ===";
    std::cout << "LineBuffer Elements Update in col=" << col << ":";
    linebuffer_print(col);
    HLS_SIZE_T j;
    for(i = 0; i < ROWS; i++) {
        for(j = 0; j < COLS; j++) {
            if(j==col)
                if(i==ROWS-1)
                    assert(val_t[i][j] == val[i][j] && "*** window shift_pixels_up mismatch! ***");
                else
                    assert(val_t[i+1][j] == val[i][j] && "*** window shift_pixels_up mismatch! ***");
            else
                assert(val_t[i][j] == val[i][j] && "*** window shift_pixels_up mismatch! ***");
        }
    }
#endif

}

/* LineBuffer insert bottom row
 * Inserts a new value in bottom row= ROWS-1 of the linebuffer
 */
template<int ROWS, int COLS, typename T> void LineBuffer<ROWS, COLS, T>::insert_bottom_row(T value, int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);

#ifdef _HLSCLIB_DEBUG_
    std::cout << "LineBuffer Elements in col=" << col << ":";
    linebuffer_print(col);
    restore_val();
#endif

    val[ROWS-1][col] = value;

#ifdef _HLSCLIB_DEBUG_
    std::cout << "===  After " << __FUNCTION__ << ":  ===";
    std::cout << "LineBuffer Elements Update in col=" << col << ":";
    linebuffer_print(col);
    HLS_SIZE_T i, j;
    for(i = 0; i < ROWS; i++) {
        for(j = 0; j < COLS; j++) {
            if(j==col && i==ROWS-1)
                assert(val[i][j] == value && "*** window insert_bottom_row mismatch! ***");
            else
                assert(val_t[i][j] == val[i][j] && "*** window insert_bottom_row mismatch! ***");
        }
    }
#endif

}

/* Line buffer getval
 * Returns the data value in the line buffer at position row, col
 */
template <int ROWS, int COLS, typename T> T& LineBuffer<ROWS, COLS, T>::getval(int row, int col) {
#pragma HLS inline
    assert(row >= 0 && row < ROWS && col >= 0 && col < COLS);
    return val[row][col];
}

/* Line buffer getval
 * Returns the data value in the line buffer at position row, col
 */
template<int ROWS, int COLS, typename T> T& LineBuffer<ROWS, COLS, T>::operator ()(int row, int col) {
#pragma HLS inline
    return getval(row, col);
}


} // namespace hls

#endif

"""
        name_file = "my_hls_video.h"
        # file creation
        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

    def generate_my_Input_types_h(self,path):


            template = \
    """
    #ifndef MY_TYPES_Input0
    #define MY_TYPES_Input0
        #include <ap_fixed.h>
        
        typedef ap_fixed< 16, 6, AP_RND, AP_SAT>  ACT_in;
        typedef short ITER;
    #endif
    """
            
            # name of the file
            name_file = "my_types_Input0.h"

            # file creation
            with open(os.path.join(path, name_file), "w") as new_file:
                new_file.write(template)

   


   
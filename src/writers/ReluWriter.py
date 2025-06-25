# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

import os
import re

from .HLSWriter import  HLSWriter


class ReluWriter(HLSWriter):

    def __init__(self, node, model, init, json_file):

        # recover data from reader node
        self.recover_data_from_reader(node, model, init, json_file)

        
        
        

        # recover hyperparameters
        # save all these hyperparameters
        #self.relu_alpha = self.relu_params()
        #self.leaky_relu_alpha = self.leaky_relu_alpha()

# -----------------------------------------------------
# METHODS FOR GENERATING CAL FILES
# return alpha parameter from "Relu" node
    def relu_params (self):

          # default value of alpha set to zero
          alpha = 0

          # if the ONNX node exist:
          if self.node is not None:

            # if the node operator is "Relu"
            if self.node.op_type == "Relu":

                # for every attribute of the node
                for attribute in self.node.attribute:

                    # if the attribute "alpha" is found
                    if attribute.name =="alpha":

                        # save it
                        alpha = attribute.f

            # return the value of alpha
            return alpha

          # if the node ONNX does not exist
          else:

            # return None
            return None

    # return alpha parameter from "LeakyRelu" node
    def leaky_relu_alpha(self):

          alpha = 0

          # if the node exists
          if self.node is not None:

            # if the node has a LeakyRelu operator
            if self.node.op_type =="LeakyRelu":

                # for every attribute of the node
                for attribute in self.node.attribute:

                    # if the attribute "alpha" is found
                    if attribute.name == "alpha":

                        # save it
                        alpha = attribute.f

            # return alpha
            return alpha

          # if the node does not exist, return None
          else:
            return None
    # write CAL files for the actual lauer
    def write_CAL(self, path):

        # list that will contains input and outputs of the layer
        input_list = []
        output_list = []

        input_list.append("input_0")
        output_list.append("output_0")

        # write the CAL file corrisponding to this layer
        self.write_id_file_CAL( input_list, output_list, path)

    # write a CAL file related to the layer
    def write_id_file_CAL(self, input_list, output_list, path):

        # initializations of the variables that will contain
        # inputs and outputs templates of the layer

        node_name = self.name


        ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()

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

        # initialized for avoiding errors
        n_bits = ap_fixed_OUT_tot

        # add all the inputs of the layer
        for elem in input_list:
                    
            size_MAC, _ = self.get_MAC_size()
            inputs_actor += template_input.format(size_MAC, elem)

        # add the output size from the input one
        for elem in output_list:

            outputs_actor += template_output.format(n_bits, elem)

        # template merging and creation of the content file
        content_file = template.format(node_name, inputs_actor, outputs_actor)

        # file name
        name_file = self.name + ".cal"

        # file creation
        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

#-----------------------------------------------------
# METHODS FOR GENERATING HLS FILES

    # generate HLS files
    def write_HLS(self, path):

        self.generate_layer_sizes_h_HLS(path)

        self.generate_relu_h_HLS(path)

        self.generate_relu_ccp_HLS(path)

        self.generate_my_types_h(path)

        self.generate_my_hls_video_h(path)

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
        #CHW
        if (len(self.isizes)) == 2 :
            if self.isizes[1] == 1 or self.isizes[1] == -1:
                in_d = 1
                in_w = self.isizes[2]
                in_h = 1
            else:
                in_d,in_h,in_w = self.isizes[1:]
        else:
            in_d,in_h,in_w = self.isizes[1:]

        if (len(self.osizes)) == 2 :
            if self.osizes[1] == 1 or self.osizes[1] == -1:
                out_d = 1
                out_w = self.osizes[2]
                out_h = 1
            else:
                out_d, out_h, out_w, = self.osizes[1:]
        else:
            out_d, out_h, out_w, = self.osizes[1:]
        

        content_file = content_file.format(
                                  in_d,in_h,in_w,
                                  out_d, out_h, out_w,
                                  )
        content_file = content_file.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        number = "r"+ number



        content_file = content_file.replace("BBB", number)
        name_file = "layer_sizes_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    #generate X.h file
    def generate_relu_h_HLS(self,path):

        content_file = \
"""
#ifndef AAA_H
#define AAA_H
	#include <hls_stream.h>
	#include "my_types_AAA.h"
	using namespace hls;
	void AAA(stream<ACT_mac> &input_0, stream <ACT_BBB> &output_0);
#endif
"""

        content_file = content_file.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        content_file = content_file.replace("CCC", number)
        number = "r"+ number
        content_file = content_file.replace("BBB", number)

        name_file = "{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    #generate X.ccp file
    def generate_relu_ccp_HLS(self,path):

        content_file = \
"""
#include <hls_stream.h>
#include <ap_fixed.h>
#include "my_types_AAA.h"
#include "layer_sizes_AAA.h"
#include "AAA.h"
using namespace hls;

void AAA(stream< ACT_mac > &input_0, stream< ACT_BBB > &output_0) {
#pragma HLS INTERFACE ap_ctrl_none port=return
	ITER i, j, k;
	ACT_mac in_mac;
	ACT_BBB	out_act;
	for(i=0; i<out_s_h_BBB; i++){
		for(j=0; j<out_s_w_BBB; j++){
			for(k=0; k<out_s_d_BBB; k++){
				input_0.read(in_mac);
				out_act = in_mac > 0 ? (ACT_BBB) in_mac : (ACT_BBB) 0;
				output_0.write(out_act);
			}
		}
	}
}
"""


        content_file = content_file.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        content_file = content_file.replace("CCC", number)
        content_file = content_file.replace("BBB", "r"+number)

        name_file = "{}.cpp".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # generate a "my tipes" file
    def generate_my_types_h(self,path):


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
        number = "r"+ number

        if self.is_Input_prev():
            template = template.replace("CCC", "in")
        else:
            #match = re.match(r'([a-zA-Z])(\d+)', self.prev_layers[0].name)
            print("In relu! What is self.prev_layers[0].name:",self.prev_layers[0].name)
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            template = template.replace("CCC", f"{first_letters[0].lower()}{last_number}")

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
            ap_fixed_INP_int_P , ap_fixed_INP_tot_P,ap_fixed_OUT_int_P , ap_fixed_OUT_tot, ap_fixed_COEFF_int_P , ap_fixed_COEFF_tot_P = self.get_my_size(spec_node=str(self.init.net_input))
        else:
            ap_fixed_INP_int_P , ap_fixed_INP_tot_P,ap_fixed_OUT_int_P , ap_fixed_OUT_tot, ap_fixed_COEFF_int_P , ap_fixed_COEFF_tot_P = self.get_my_size(spec_node=self.prev_layers[0].name)

        #ap_fixed_DATA_int_prev, ap_fixed_DATA_tot_prev, selector_DATA_prev, ap_fixed_COEFF_int_prev, ap_fixed_COEFF_tot_prev, selector_COEFF_prev = self.get_my_size(bit_size_directives, prev_layer.operation)
        ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()


        
        tmp = template_ap_fixed.replace("BBB", str(ap_fixed_OUT_tot)).replace("CCC", str(ap_fixed_OUT_int))
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

            tmp = template_ap_fixed.replace("BBB", str(ap_fixed_INP_tot_P)).replace("CCC", str(ap_fixed_INP_int_P))
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

            tmp = template_ap_fixed.replace("BBB", str(ap_fixed_INP_tot_P)).replace("CCC", str(ap_fixed_INP_int_P))
            template_previous = template_previous.replace("WWW",tmp)
            content_file= content_file + template_previous


        # name of the file
        name_file = "my_types_" + self.name + ".h"

        # file creation
        with open(os.path.join(path, name_file), "w") as new_file:
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

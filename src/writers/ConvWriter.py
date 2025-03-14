# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

import re
import os
import numpy as np
from onnx import helper


from .HLSWriter import  HLSWriter

#buffers
class ConvWriter(HLSWriter):

    def __init__(self, node, model, init, json_file, types_file, sizes_file):

        # recover data from reader node
        self.recover_data_from_reader(node, model, init, json_file)

        
        pads = [0, 0, 0, 0]  # Default value for pads

        for attr in node.attribute:
            if attr.name == "pads":
                pads = helper.get_attribute_value(attr)
            elif attr.name == "strides":
                stride = helper.get_attribute_value(attr)
            elif attr.name == "kernel_shape":
                kernel_shape = helper.get_attribute_value(attr)
            elif attr.name == "group":
                group = helper.get_attribute_value(attr)
            elif attr.name == "dilations":
                dilations = helper.get_attribute_value(attr)


        # recover hyperparameters
        self.stride = stride
        self.kernel = kernel_shape
        self.padding = pads
        self.group = group
        self.dilation = dilations
        self.types_file = types_file
        self.sizes_file = sizes_file
        

        # recover parameters
        #self.parameters = self.node.parameters
        

# -----------------------------------------------------
# METHODS FOR GENERATING CAL FILES
    
    # write CAL files for this layer
    def write_CAL(self, path):

        # lists containing inputs and output of the layer
        input_list = []
        output_list = []

        

        input_list.append("input_0")
        output_list.append("output_0")
        

        # add conv parameters
        input_list.append("weight_" + self.name)
        input_list.append("bias_" + self.name)

        self.write_id_file_CAL( input_list, output_list, path)

        # buffer creation:
        actor_name = "line_buffer_" + self.name

        input_buffer = "input_0"
        output_buffer = "output_0"

        self.write_line_buffer_file_CAL(actor_name, input_buffer, output_buffer, path )

        # weight CAL creation:
        
        actor_name = "weight_" + self.name 
        output_actor_name = "weight_" + self.name + "_r"

        self.write_parameter_file_CAL(actor_name, output_actor_name, path)

        # bias CAL creation:
        actor_name = "bias_" + self.name 
        output_actor_name = "bias_" + self.name +"_r"

        self.write_parameter_file_CAL(actor_name, output_actor_name, path)

    # write a CAL file related to the layer
    def write_id_file_CAL(self, input_list, output_list, path):
    
        sufix = "_V"

        node_name = self.name

        
        ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()


        # initializations of the variables that will contain
        # inputs and outputs templates of the layer
        inputs_actor = ""
        outputs_actor = ""

        # template to be filled
        template = \
"""package actors;

actor pe_{}()
    {}

    ==>
    {} :
end"""

        
        # input template to be filled
        template_input = \
"""
    int(size={}) {}"""
        
        # input template to be filled for weights and biases
        template_input_wb = \
"""
    int(size={}) {}_r"""

        # output template to be filled
        template_output = \
"""     
    int(size={}) {}"""

        # initialized for avoiding errors:

        

        # recover n_filters of Conv
        n_filters = self.init.parameters_values[self.parameters[0]].shape[0]

        # add all the inputs of the layer
        #   "replace" is used to delete the "_ConvX" after the name of weight and, eventually, bias
        for index,elem in enumerate(input_list):
            
            # input
            if(elem.replace("_" + node_name,"") != "weight" and elem.replace("_" + node_name,"") != "bias"):

                inputs_actor += template_input.format(ap_fixed_INP_tot, elem.replace("_" + node_name,""))

                # add comma if not last element
                if(index != len(input_list)-1):

                    inputs_actor += ","

            # parameters
            elif(elem.replace("_" + node_name,"") == "weight" or elem.replace("_" + node_name,"") == "bias"):

                
                n_bits =  ap_fixed_COEFF_tot * n_filters

                inputs_actor += template_input_wb.format(n_bits, elem)


                if (index != len(input_list) - 1):
                    inputs_actor += ","


        # add the output size with mac parameters
        size_MAC, _ = self.get_MAC_size()
        for elem in output_list:

            outputs_actor += template_output.format(size_MAC, elem)


        # template merging and creation of the content file
        content_file = template.format(node_name, inputs_actor, outputs_actor)

        # file name
        name_file = "pe_" +self.name + ".cal"

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

        # nome del file
        name_file = buff_name + ".cal"

        # creazione del file
        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)


    # write a CAL file for every parameter found in the layer
    def write_parameter_file_CAL(self, actor_name, output_name, path):

        template = \
"""package actors;

actor {}()

        ==>

        int(size={}) {} :
end"""

        ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()

        # recover number of filters from weights shape
        n_filters = self.init.parameters_values[self.parameters[0]].shape[0]


        n_bits = ap_fixed_COEFF_tot * n_filters


        # contenuto del file
        content_file = template.format(actor_name, n_bits, output_name)

        # nome del file
        name_file = actor_name + ".cal"

        # creazione del file
        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

#-----------------------------------------------------
# METHODS FOR GENERATING HLS FILES

    # generate HLS files
    def write_HLS(self, path):

        self.generate_layer_sizes_h_HLS(path)

        self.generate_conv_h_HLS(path)

        self.generate_conv_ccp_HLS(path)

        self.generate_parameters_h_HLS(path)

        self.generate_weight_h_HLS(path)

        self.generate_weight_ccp_HLS(path)

        self.generate_bias_h_HLS(path)

        self.generate_bias_ccp_HLS(path)

        self.generate_line_buffer_h_HLS(path)

        self.generate_line_buffer_ccp_HLS(path)

        self.generate_my_types_h(path)

        self.generate_my_hls_video_h(path)

        #self.generate_my_Input_types_h(path)


        
    # generate layer_size_X.h file
    def generate_layer_sizes_h_HLS(self, path):

        content_file = \
"""
#ifndef LAYER_SIZES_AAA_H
#define LAYER_SIZES_AAA_H

    #define in_s_d_AAA {}
    #define in_s_h_AAA {}
    #define in_s_w_AAA {}
    #define out_s_d_AAA {}
    #define out_s_h_AAA {}
    #define out_s_w_AAA {}

    #define kern_s_k_AAA {}
    #define kern_s_d_AAA {}
    #define kern_s_h_AAA {}
    #define kern_s_w_AAA {}

    #define stride_h_AAA {}
    #define stride_w_AAA {}

    #define pad_h_AAA {}
    #define pad_w_AAA {}      
#endif     
"""
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        content_file = content_file.replace("AAA", "c"+number)

        
        in_d, in_h, in_w = self.isizes[1:]
        print("self.osizes conv", self.osizes)
        out_d, out_h, out_w, = self.osizes[1:]

        # k is depth of the weights, d is channels of the weights
        kern_k, kern_d = out_d, in_d

        kern_h, kern_w = self.kernel
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding[0:2]

        content_file = content_file.format(
            in_d, in_h, in_w,
            out_d, out_h, out_w,
            kern_k, kern_d, kern_h, kern_w,
            stride_h, stride_w,
            pad_h, pad_w
        )

        name_file = "layer_sizes_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

    # generate X.h file
    def generate_conv_h_HLS(self, path):

        content_file = \
"""
#ifndef AAA_H
#define AAA_H
    #include <hls_stream.h>
    #include "my_types_AAA.h"
    #include "layer_sizes_AAA.h"
    using namespace hls;
    void pe_AAA(stream<ACT_CCC> &input_0, stream <ACT_mac> &output_0, stream <KERN_ITEM_BBB> &weight_AAA_r, stream <KERN_ITEM_BBB> &bias_AAA_r);
#endif
"""

        # fill the template
        content_file = content_file.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))

        content_file = content_file.replace("BBB", "c" + number)

        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")                
            

        # create file
        name_file = "pe_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # generate X.ccp file
    def generate_conv_ccp_HLS(self, path):

        content_file = \
"""
#include <hls_stream.h>
#include <ap_fixed.h>

#include "my_types_AAA.h"
#include "layer_sizes_AAA.h"
#include "pe_AAA.h"
using namespace hls;

void mac_AAA(ACT_mac* out_val, ACT_CCC* current, COEFF_BBB* kern);

 
void pe_AAA(stream<ACT_CCC> &input_0, stream <ACT_mac> &output_0, stream <KERN_ITEM_BBB> &weight_AAA_r, stream <KERN_ITEM_BBB> &bias_AAA_r){
	#pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS DATA_PACK variable=weight_AAA_r
    #pragma HLS DATA_PACK variable=bias_AAA_r
	
	ITER pout;
	ITER hout;
	ITER wout;
	
	ITER pkern;
	ITER hkern;
	ITER wkern;

	ITER init_idx;
	ITER wr_idx;
	
	ACT_CCC current;
	KERN_ITEM_BBB current_bias;
	#pragma HLS ARRAY_PARTITION variable=current_bias.w complete dim=1
	KERN_ITEM_BBB current_kern;
	#pragma HLS ARRAY_PARTITION variable=current_kern.w complete dim=1
	ACT_mac out_val[out_s_d_BBB];
    #pragma HLS ARRAY_PARTITION variable=out_val dim=1 complete
	

	for(hout=0; hout<out_s_h_BBB; hout++){
		for(wout=0; wout<out_s_w_BBB; wout++){
			
			bias_AAA_r.read(current_bias);
			Loop_init:for(init_idx=0; init_idx < out_s_d_BBB; init_idx++){
				#pragma HLS UNROLL
				out_val[init_idx] = (ACT_mac) current_bias.w[init_idx];
			};
			
			Loop_conv:for(hkern=0; hkern < kern_s_h_BBB ; hkern++){
				for(wkern=0; wkern < kern_s_w_BBB; wkern++){
					Loop_read:for(pkern=0; pkern < kern_s_d_BBB; pkern++){
						input_0.read(current);
						weight_AAA_r.read(current_kern);	//current_kern = weight[:][pkern][hkern][wkern];
						mac_AAA(out_val, &current, current_kern.w);

					}
				}
			}
			
			Loop_wr:for(wr_idx=0; wr_idx < out_s_d_BBB; wr_idx++){
				output_0.write(out_val[wr_idx]);
			}
		}
	}
}
	
void mac_AAA(ACT_mac* out_val, ACT_CCC* current, COEFF_BBB* kern){
	Inner_loop:for(ITER pout=0; pout < out_s_d_BBB; pout++){
			#pragma HLS UNROLL
			out_val[pout] += (ACT_mac)*current * (ACT_mac)kern[pout];
	}
}

"""

        # fill the template
        content_file = content_file.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))

        content_file = content_file.replace("BBB", "c" + number)

        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")                
            
        name_file = "pe_{}.cpp".format(self.name)

        # generate file
        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # generate line_buffer_X.h file
    def generate_line_buffer_h_HLS(self, path):

        content_file = \
"""
#ifndef LINE_BUFFER_AAA_H
#define LINE_BUFFER_AAA_H
    #include "my_types_AAA.h"
    #include <hls_stream.h>
    #include "layer_sizes_AAA.h"
    using namespace hls;

    void line_buffer_AAA(stream<ACT_CCC> &input_0, stream <ACT_CCC> &output_0);
#endif
"""

        # fill the template
        content_file = content_file.replace("AAA", self.name)
        
        if  self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")     
        

        # create file
        name_file = "line_buffer_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

    # generate line_buffer_X.ccp file
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



void line_buffer_AAA(stream<ACT_CCC> &input_0, stream <ACT_CCC> &output_0) {
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

	ACT_CCC in_val;
	ACT_CCC out_val;
	bool out_of_bounds;
	
	LineBuffer<kern_s_h_BBB,in_s_w_BBB+2*pad_w_BBB, ACT_CCC> buffer[in_s_d_BBB];

	hin = 0;
	win = 0;
	
	for(hout = 0; hout < out_s_h_BBB; hout++) {		
		for(wout = 0; wout < out_s_w_BBB; wout++) {
Loop_while:while( (win <= (wout * stride_w_BBB + kern_s_w_BBB-1)) || (hin < (hout * stride_h_BBB + kern_s_h_BBB-1) ) ){
				out_of_bounds = ((hin<pad_h_BBB) || (hin>pad_h_BBB+in_s_h_BBB-1) || (win<pad_w_BBB) || (win>pad_w_BBB+in_s_w_BBB-1))? true : false;
Loop_lettura:for (pin = 0; pin < in_s_d_BBB; pin++) {
					if(out_of_bounds){
						in_val=0;
					} else{
						input_0.read(in_val);
					}
					buffer[pin].shift_pixels_up(win);
					buffer[pin].insert_bottom_row(in_val,win);
					}
				// Update input indexes
				if(win < in_s_w_BBB-1+2*pad_w_BBB){ 
					win++;
					}
				else {
					win = 0; 
					hin++;
					if(hin > (hout * stride_h_BBB + kern_s_h_BBB-1) ){
					break;
					}
				}
			}
	
			

		//Now it can write a submatrix
Loop_scrittura:for(hkern=0; hkern < kern_s_h_BBB ; hkern++){
				for(wkern=0; wkern < kern_s_w_BBB; wkern++){
	Loop_interno: for(pkern=0; pkern < kern_s_d_BBB; pkern++){
					#pragma HLS DEPENDENCE variable=buffer array inter false
					#pragma HLS PIPELINE rewind
						out_val = buffer[pkern].getval(hkern, wout*stride_w_BBB + wkern);
						output_0.write(out_val);
					}
				}
			}
		}
	}
}	
			
"""

        # fill the template
        content_file = content_file.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))

        content_file = content_file.replace("BBB", "c" + number)

        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")   

        # create file
        name_file = "line_buffer_{}.cpp".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

    #generate parameters_X.h file
    def generate_parameters_h_HLS(self, path):

        content_file = \
"""
#ifndef AAA_PARAMS
#define AAA_PARAMS
    {}
	{}
#endif
"""
        # fill the template
        content_file = content_file.replace("AAA", self.name)

        #enter_id = self.map_of_in_elements["weight"]

        enter_id = "_"+self.node.input[1]

        print(enter_id)
        print(self.init.parameters_values[enter_id].shape)
        # start reshaping:
        k, d, h, w = self.init.parameters_values[enter_id].shape

        wanted_shape = (d,h,w,k)
        weight_values = np.zeros(shape = wanted_shape)

        # custom reshaping of values
        # filters
        for index_k, k1 in enumerate(self.init.parameters_values[enter_id]):

            # channels
            for index_d, d1 in enumerate(k1):

                # height
                for index_h, h1 in enumerate(d1):

                    # width
                    for index_w, value in enumerate(h1):

                        # reordering
                        weight_values[index_d][index_h][index_w][index_k] = value

        weight_values = str(weight_values.tolist())

        weight_values = weight_values.replace("[","{")
        weight_values = weight_values.replace("]", "}")

        weight = \
"""
#define WEIGHT_AAA   """ + weight_values

        # fill the template
        weight = weight.replace("AAA", self.name)

        #enter_id = self.map_of_in_elements["bias"]
        enter_id = "_"+self.node.input[2]

        bias_values = str(self.init.parameters_values[enter_id].tolist())
        bias_values = bias_values.replace("[", "{")
        bias_values = bias_values.replace("]", "}")

        bias = \
"""
#define BIAS_AAA   """ + bias_values
        # fill the template
        bias = bias.replace("AAA", self.name)
        content_file = content_file.format(weight, bias)

        name_file = "parameters_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

    #generate weight_X.h file
    def generate_weight_h_HLS(self, path):

        content_file = \
"""
#ifndef WEIGHT_AAA_H
#define WEIGHT_AAA_H
    #include <hls_stream.h>
    #include "my_types_AAA.h"
    #include "layer_sizes_AAA.h"
    using namespace hls;
    void weight_AAA(stream <KERN_ITEM_BBB> &weight_AAA_r);
#endif
"""

        content_file = content_file.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))

        number = "c"+ number



        content_file = content_file.replace("BBB", number)

        name_file = "weight_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # generate weight_X.ccp file
    def generate_weight_ccp_HLS(self, path):

        content_file = \
"""
#include <hls_stream.h>
#include <ap_fixed.h>

#include "my_types_AAA.h"
#include "layer_sizes_AAA.h"
#include "weight_AAA.h"
#include "parameters_AAA.h"
using namespace hls;

 
void weight_AAA(stream <KERN_ITEM_BBB> &weight_AAA_r){
#pragma HLS DATA_PACK variable=weight_AAA_r
#pragma HLS INTERFACE ap_ctrl_none port=return
	
	ITER pout;
	ITER hout;
	ITER wout;
	
	ITER pkern;
	ITER hkern;
	ITER wkern;
	
	KERN_ITEM_BBB current_kern;
	#pragma HLS ARRAY_PARTITION variable=current_kern.w complete dim=1
	
	const KERN_ITEM_BBB weight[kern_s_d_BBB][kern_s_h_BBB][kern_s_w_BBB] = WEIGHT_AAA;
	
	// Riprende l'ordine dei cicli for dell'attore che fa la convoluzione
	for(hout=0; hout<out_s_h_BBB; hout++){
		for(wout=0; wout<out_s_w_BBB; wout++){	
			for(hkern=0; hkern < kern_s_h_BBB ; hkern++){
				for(wkern=0; wkern < kern_s_w_BBB; wkern++){
					for(pkern=0; pkern < kern_s_d_BBB; pkern++){
						#pragma HLS PIPELINE
						current_kern = weight[pkern][hkern][wkern];
						weight_AAA_r.write(current_kern);
					}
				}
			}
		}
	}
}
"""

        content_file = content_file.replace("AAA",self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        number = "c"+ number



        content_file = content_file.replace("BBB", number)

        name_file = "weight_{}.cpp".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # generate bias_X.h file
    def generate_bias_h_HLS(self, path):

        content_file = \
"""
#ifndef BIAS_AAA_H
#define BIAS_AAA_H
    #include <hls_stream.h>
    #include "my_types_AAA.h"
    #include "layer_sizes_AAA.h"
    using namespace hls;
    void bias_AAA(stream <KERN_ITEM_BBB> &bias_AAA_r);
#endif
"""

        content_file = content_file.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        number = "c"+ number

        content_file = content_file.replace("BBB", number)

        name_file = "bias_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

    # generate bias_X.ccp file
    def generate_bias_ccp_HLS(self,path):

        content_file = \
"""
#include <hls_stream.h>
#include <ap_fixed.h>

#include "my_types_AAA.h"
#include "layer_sizes_AAA.h"
#include "bias_AAA.h"
#include "parameters_AAA.h"
using namespace hls;

 
void bias_AAA(stream <KERN_ITEM_BBB> &bias_AAA_r){
#pragma HLS DATA_PACK variable=bias_AAA_r
#pragma HLS INTERFACE ap_ctrl_none port=return
	
	ITER pout;
	ITER hout;
	ITER wout;
	
	const KERN_ITEM_BBB current_bias = BIAS_AAA;
	#pragma HLS ARRAY_PARTITION variable=current_bias.w complete dim=1
	
	// Riprende l'ordine dei cicli for dell'attore che fa la convoluzione
	for(hout=0; hout<out_s_h_BBB; hout++){
		for(wout=0; wout<out_s_w_BBB; wout++){
			bias_AAA_r.write(current_bias);
		}
	}
}
"""

        content_file = content_file.replace("AAA",self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        number = "c"+ number



        content_file = content_file.replace("BBB", number)

        name_file = "bias_{}.cpp".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

   # generate a "my tipes" file
    def generate_my_types_h(self , path ):


        template_types = \
"""
// AAA types
    typedef XXX ACT_BBB;
    typedef YYY COEFF_BBB;
    typedef struct kern_item_BBB {COEFF_BBB w[kern_s_k_BBB];} KERN_ITEM_BBB;
"""


        template = \
"""
#ifndef MY_TYPES_AAA_S
#define MY_TYPES_AAA_S
    #include <ap_fixed.h>
    #include "layer_sizes_AAA.h"
    // types of this layer
    typedef ZZZ ACT_mac;
    typedef XXX ACT_BBB;
    typedef YYY COEFF_BBB;
    typedef struct kern_item_BBB {COEFF_BBB w[kern_s_k_BBB];} KERN_ITEM_BBB;
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
        number = ''.join(filter(str.isdigit, self.name))
        number = "c"+ number
         
        # fill the template
        template_types = template_types.replace("AAA", self.name)
        template_types = template_types.replace("BBB", number)
        template = template.replace("AAA", self.name)
        template = template.replace("BBB", number)
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

        # create content file
        content_file = template.replace("AAA", self.name)

        
        tmp = template_ap_fixed.replace("BBB", str(ap_fixed_INP_tot)).replace("CCC", str(ap_fixed_INP_int))
        content_file = content_file.replace("XXX",tmp)
        template_types = template_types.replace("XXX", tmp)
        mac_value_tot,mac_value_int = self.get_MAC_size()
        tmp = template_ap_fixed.replace("BBB", str(mac_value_tot)).replace("CCC", str(mac_value_int))
        content_file = content_file.replace("ZZZ",tmp)

        

        tmp = template_ap_fixed.replace("BBB", str(ap_fixed_COEFF_tot)).replace("CCC", str(ap_fixed_COEFF_int))
        content_file = content_file.replace("YYY", tmp)
        template_types = template_types.replace("YYY", tmp)

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
        
        with open(os.path.join(path, self.types_file), "a") as new_file:
            new_file.write(template_types)

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



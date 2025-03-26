# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

import os

from .HLSWriter import  HLSWriter

# no buffers
class ConcatWriter(HLSWriter):

    def __init__(self, node, model, init, json_file, types_file, size_file):

        # recover data from reader node
        self.recover_data_from_reader(node, model, init, json_file)

        print("ConcatWriter")
        print("self.node",self.node)
        print(self.prev_layers)

# -----------------------------------------------------
# METHODS FOR GENERATING CAL FILES

    # write CAL files for the actual lauer
    def write_CAL(self, path):

        # list that will contains inputs and output of the layer
        input_list = []
        output_list = []

        input_tmp = "input_{}"


        for i in range(len(self.prev_layers)):
            input_list.append(input_tmp.format(i))

        output_list.append("output_0")

        # write the CAL file corrisponding to this layer
        self.write_id_file_CAL(self.name, input_list, output_list, path)

    # write a CAL file related to the layer
    def write_id_file_CAL(self, node_name, input_list, output_list, path):

        # initializations of the variables that will contain
        # inputs and outputs templates of the layer
        inputs_actor = ""
        outputs_actor = ""

        ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()

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
        n_bits = ""

        print("input_list",input_list)
        # add all the inputs of the layer
        for index,elem in enumerate(input_list):

            n_bits = ap_fixed_INP_tot

            inputs_actor += template_input.format(n_bits, elem)

            # add a comma if it is not the last one
            if(index != len(input_list)-1):

                inputs_actor += ","

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
    def write_HLS(self,path):

        self.generate_layer_sizes_h_HLS(path)

        self.generate_concat_h_HLS(path)

        self.generate_concat_ccp_HLS(path)

        self.generate_my_types_h(path)

    #generate layer_size_X.h file
    def generate_layer_sizes_h_HLS(self,path):
        
        in_size = {}
        in_d = {}
        in_h = {}
        in_w = {}

        in_size_tmp = \
"""
        #define in_s_d_{} {}
        #define in_s_h_{} {}
        #define in_s_w_{} {}
"""

        content_file = \
"""
#ifndef LAYER_SIZES_H
#define LAYER_SIZES_H

"""

        output_tmp = \
"""
        #define in_s_h in_s_h_0
        #define in_s_w in_s_w_0
        #define out_s_d {}
        #define out_s_h {}
        #define out_s_w {}        
#endif     
"""
        for i in range(len(self.prev_layers)):
            in_size[i] = self.model.get_tensor_shape(self.node.input[i])
            in_d[i], in_h[i], in_w[i] = in_size[i][1:]
            content_file += in_size_tmp.format(i, in_d[i],i, in_h[i],i, in_w[i])
        #CHW
        

        
        out_d, out_h, out_w, = self.osizes[1:]

        
        output_tmp = output_tmp.format(
                                  out_d, out_h, out_w,
                                  )

        content_file +=  output_tmp
        name_file = "layer_sizes_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    #generate X.h file
    def generate_concat_h_HLS(self,path):

        content_file = \
"""
#ifndef CONCAT_H
#define CONCAT_H
	#include <hls_stream.h>
	#include "my_types_AAA.h"
	#include "layer_sizes_AAA.h"
	
	using namespace hls;
	void AAA(BBB, stream <ACT_mac> &output_0);
#endif
"""
        input_list = []
        output_list = []

        input_tmp = "input_{}"


        for i in range(len(self.prev_layers)):
            input_list.append(input_tmp.format(i))

        output_list.append("output_0")


    

        input_template = \
""" stream<ACT_mac> &DDD"""

        function_input = ""

        num_of_input = len(input_list)

        for index,input_element in enumerate(input_list):

            function_input += input_template.replace("DDD", input_element)


            if(index+1 != num_of_input):

                function_input += ","


        content_file = content_file.replace("AAA", self.name)
        content_file = content_file.replace("BBB", function_input)

        name_file = "{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    #generate X.ccp file
    def generate_concat_ccp_HLS(self,path):

        content_file = \
"""
#include <hls_stream.h>
#include <ap_fixed.h>
#include "my_types_AAA.h"
#include "layer_sizes_AAA.h"
#include "AAA.h"
using namespace hls;

void AAA(BBB, stream <ACT_mac> &output_0){
#pragma HLS INTERFACE ap_ctrl_none port=return

	ACT_mac current;
	ITER i, j, k;

	for(k=0; k<in_s_h; k++){
		for(j=0; j<in_s_w; j++){
        DDD
		}
	}
}
"""


        input_list = []
        output_list = []

        input_tmp = "input_{}"


        for i in range(len(self.prev_layers)):
            input_list.append(input_tmp.format(i))

        output_list.append("output_0")


        input_template = \
""" stream<ACT_mac> &EEE"""

        function_input = ""

        num_of_input = len(input_list)

        for index,input_element in enumerate(input_list):

            function_input += input_template.replace("EEE", input_element)

            if(index+1 != num_of_input):

                function_input += ","


        code_template = \
"""
			for(i=0; i<in_s_d_XXX; i++){
				current = FFF.read();
				output_0.write(current);
			}
"""

        function_code = ""

        for index,input_element in enumerate(input_list):

            code_tmp = code_template.replace("XXX", str(index))
            function_code += code_tmp.replace("FFF", input_element)


        content_file = content_file.replace("AAA", self.name)
        content_file = content_file.replace("BBB", function_input)
        content_file = content_file.replace("DDD", function_code)


        name_file = "{}.cpp".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # generate a "my tipes" file
    def generate_my_types_h(self,path):

        template = \
"""
#ifndef MY_TYPES_AAA
#define MY_TYPES_AAA
    #include <ap_fixed.h>
    #include "layer_sizes_AAA.h"
    typedef XXX ACT_mac;
    typedef short ITER;
#endif
"""
        # initialization
        ap_fixed_DATA_tot = ""
        ap_fixed_DATA_int = ""
        ap_fixed_COEFF_tot = ""
        ap_fixed_COEFF_int = ""

        template_ap_fixed = \
"""ap_fixed< BBB, CCC, AP_RND, AP_SAT> """


        ap_fixed_DATA_tot, ap_fixed_DATA_int = self.get_MAC_size()
       
        template = template.replace("AAA", self.name)
        template_ap_fixed = template_ap_fixed.replace("BBB", str(ap_fixed_DATA_tot))
        template_ap_fixed = template_ap_fixed.replace("CCC", str(ap_fixed_DATA_int))

        content_file = template.replace("XXX", template_ap_fixed)
        # name of the file
        name_file = "my_types_" + self.name + ".h"

        # file creation
        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)
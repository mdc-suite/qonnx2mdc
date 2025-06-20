# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

import os
import re

from .HLSWriter import  HLSWriter

class SigmoidWriter(HLSWriter):

    def __init__(self, node, model, init, json_file, types_file, sizes_file):

        # recover data from reader node
        self.recover_data_from_reader(node, model, init, json_file)

        self.types_file = types_file
        self.sizes_file = sizes_file
# -----------------------------------------------------
# METHODS FOR GENERATING CAL FILES

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

        node_name = self.name

        
        ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()

        
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

        # initialized for avoiding errors
        n_bits = ""

        # add all the inputs of the layer
        for elem in input_list:

            inputs_actor += template_input.format(ap_fixed_INP_tot, elem)

        # add the output of the layer
        for elem in output_list:

            outputs_actor += template_output.format(ap_fixed_OUT_tot, elem)

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

        self.generate_sigmoid_h_HLS(path)

        self.generate_sigmoid_ccp_HLS(path)

        self.generate_my_types_h(path)

        self.generate_sigmoid_table_h(path)

    #generate layer_size_X.h file
    def generate_layer_sizes_h_HLS(self,path):

        template = \
"""
    #define in_s_d_AAA {}
    #define in_s_h_AAA {}
    #define in_s_w_AAA {}
    #define in_s_AAA in_s_d_AAA*in_s_h_AAA*in_s_w_AAA  
"""

        content_file = \
"""
#ifndef LAYER_SIZES_AAA_H
#define LAYER_SIZES_AAA_H

        #define in_s_d_AAA {}
        #define in_s_h_AAA {}
        #define in_s_w_AAA {}
        #define in_s_AAA in_s_d_AAA*in_s_h_AAA*in_s_w_AAA       
#endif     
"""
        number = ''.join(filter(str.isdigit, self.name))

        number = "s"+ number

        content_file = content_file.replace("AAA", number)
        template = template.replace("AAA", number)
        #CHW
        if(len(self.isizes) == 4):
            in_d,in_h,in_w = self.isizes[1:]

        elif (len(self.isizes) == 2):
            in_d = 1
            in_h, in_w = self.isizes
        
        if in_d == -1:
            in_d = 1
        elif in_h == -1:
            in_h = 1
        elif in_w == -1:
            in_w = 1

        content_file = content_file.format(
                                  in_d,in_h,in_w
                                  )
        template = template.format(
                                  in_d,in_h,in_w
                                  ) 
        name_file = "layer_sizes_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

        with open(os.path.join(path, self.sizes_file), "a") as new_file:
            new_file.write(template)

    #generate X.h file
    def generate_sigmoid_h_HLS(self,path):

        content_file = \
"""
#ifndef SIGMOID_AAA_H
#define SIGMOID_AAA_H
	#include <hls_stream.h>
	#include "my_types_AAA.h"
	using namespace hls;
	void AAA(stream<ACT_BBB> &input_0, stream <ACT_CCC> &output_0);
#endif
"""

        content_file = content_file.replace("AAA", self.name)
        number = ''.join(filter(str.isdigit, self.name))

        number = "s"+ number

        content_file = content_file.replace("CCC", number)


        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        elif "Conv" in self.prev_layers[0].name or "Gemm" in self.prev_layers[0].name:
            content_file = content_file.replace("BBB", "mac")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("BBB", f"{first_letters[0].lower()}{last_number}")    

        name_file = "{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    def generate_sigmoid_table_h(self,path):
        
        # Example usage
        table_size = 1024
        name_file = "AAA_table.h"
        name_file = name_file.replace("AAA", self.name)
        

        # Write to the file with ap_fixed<32, 16, AP_RND, AP_SAT>
        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(f"#ifndef {self.name}_Table \n")
            new_file.write(f"#define {self.name}_Table \n")
            new_file.write(f"#define TABLE_SIZE {table_size} \n")
            new_file.write("#include <ap_fixed.h>\n\n")
            new_file.write(f"#include \"my_types_{self.name}.h\" \n\n")
            new_file.write("const ACT_s0 sigmoid_table[] = {")
            
            for i in range(table_size):
                x_value = 32.0 * (i - table_size / 2.0) / table_size
                sigmoid_value = 1 / (1 + 2.71828 ** (-x_value))  # Sigmoid approximation using ap_fixed
                new_file.write(f" ACT_s0({sigmoid_value:.6f})")
                
                if i < table_size - 1:
                    new_file.write(",")
                    if (i + 1) % 10 == 0:  # Insert newline after every 10 values
                        new_file.write("\n    ")
            
            new_file.write("};\n\n")
            new_file.write("#endif  \n\n")


    #generate X.ccp file
    def generate_sigmoid_ccp_HLS(self,path):

        #WE ARE STILL LOOKING FOR A WAY TO IMPLEMENT THE SIGMOID FUNCTION: FOR NOW, WE MAKE IT DO NOTHING

        do_nothing = False
        lut_or_discrete = False

        '''
        input.to_double(): This converts the fixed-point input value (ap_fixed<32,16>) to a double-precision floating-point value. This conversion is necessary because the original calculation involved floating-point values, and we want to maintain compatibility.

        (input.to_double() + 8.0): This shifts the input values by 8 units to the right. The goal is to map the original range of input values, which is assumed to be within the range [-8.0, 8.0], to a positive range [0.0, 16.0]. Adding 8.0 achieves this shift.

        * TABLE_SIZE: This scales the shifted values by the TABLE_SIZE. The result is that the shifted values now span the range [0.0, TABLE_SIZE * 16.0].

        / 16.0: This further scales the values by 16.0. This division ensures that the final values are in the range [0.0, TABLE_SIZE].

        (int): This converts the result to an integer, rounding down to the nearest integer. This integer value is then used as the index for the sigmoid_table.

        '''

        if do_nothing:
            content_file = \
"""

#include <hls_stream.h>
#include "AAA_table.h"
#include "ap_fixed.h"
#include "hls_math.h"
#include "my_types_AAA.h"
#include "layer_sizes_AAA.h"
#include "AAA.h"
using namespace hls;

void AAA(stream<ACT_BBB> &input_0, stream <ACT_CCC> &output_0){
#pragma HLS INTERFACE ap_ctrl_none port=return
	ITER i;
	ACT_BBB v;
    ACT_CCC result;
	for(i=0; i<in_s_CCC; i++){
					input_0.read(v);
					result = (ACT_s0)v;
					output_0.write(result);
			}
}
"""

        elif lut_or_discrete:

            content_file = \
"""

#include <hls_stream.h>
#include "AAA_table.h"
#include "ap_fixed.h"
#include "hls_math.h"
#include "my_types_AAA.h"
#include "layer_sizes_AAA.h"
#include "AAA.h"
using namespace hls;

void AAA(stream<ACT_BBB> &input_0, stream <ACT_CCC> &output_0){
#pragma HLS INTERFACE ap_ctrl_none port=return
	ITER i;
	ACT_BBB v;
    ACT_CCC result;
	for(i=0; i<in_s_CCC; i++){
					input_0.read(v);
					if (v <= -8) {
						result =  ACT_CCC(0.0);
						} else if (v > -8 && v <= -2) {
							result =  ACT_CCC(0.01511) * v + ACT_CCC(0.09783);
						} else if (v > -2 && v <= 2) {
							result =  ACT_CCC(0.21619) * v + ACT_CCC(0.5);
						} else if (v > 2 && v <= 8) {
							result = ACT_CCC(0.01511) * v + ACT_CCC(0.90217);
						} else {
							result = ACT_CCC(1.0);
						}
					output_0.write(result);
			}
}
"""

        else:

            content_file = \
"""
#include <hls_stream.h>
#include "AAA_table.h"
#include "ap_fixed.h"
#include "hls_math.h"
#include "my_types_AAA.h"
#include "layer_sizes_AAA.h"
#include "AAA.h"
using namespace hls;

void AAA(stream<ACT_BBB> &input_0, stream <ACT_CCC> &output_0){
#pragma HLS INTERFACE ap_ctrl_none port=return
	ITER i;
	ACT_BBB v;
        ACT_CCC result;
	for(i=0; i<in_s_CCC; i++){
		input_0.read(v);
                int index = (int)((v.to_double() + 32.0) * TABLE_SIZE / 64.0);
                // Ensure the index is within bounds
                if (index < 0) {
                    index = 0;
                } else if (index >= TABLE_SIZE) {
                    index = TABLE_SIZE - 1;
                }
                // Retrieve and return the sigmoid value from the lookup table
                result = (ACT_CCC)sigmoid_table[index];
                
                output_0.write(result);
            }
}
"""

        content_file = content_file.replace("AAA", self.name)
        
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        number = "s"+ number
        content_file = content_file.replace("CCC", number)

        if self.is_Input_prev():
            content_file = content_file.replace("BBB", "in")
        elif "Conv" in self.prev_layers[0].name or "Gemm" in self.prev_layers[0].name:
            content_file = content_file.replace("BBB", "mac")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("BBB", f"{first_letters[0].lower()}{last_number}")  

        name_file = "{}.cpp".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # generate a "my tipes" file
    def generate_my_types_h(self,path):

        template_types = \
"""
// AAA types
    typedef XXX ACT_CCC;
"""

        template = \
        """
#ifndef MY_TYPES_AAA_S
#define MY_TYPES_AAA_S
    #include <ap_fixed.h>
    #include "layer_sizes_AAA.h"
    //types of this layer
    typedef XXX ACT_CCC;
    // types of previous layer
    typedef WWW ACT_BBB;
    typedef short ITER;
#endif
        """

         # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        number = "s"+ number
        content_file = template.replace("CCC", number)
        template_types = template_types.replace("AAA", self.name)
        template_types = template_types.replace("CCC", number)

        if self.is_Input_prev():
            content_file = content_file.replace("BBB", "in")
        elif "Conv" in self.prev_layers[0].name or "Gemm" in self.prev_layers[0].name:
            content_file = content_file.replace("BBB", "mac")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("BBB", f"{first_letters[0].lower()}{last_number}")    


        # initialization
        ap_fixed_DATA_tot = ""
        ap_fixed_DATA_int = ""
        ap_fixed_COEFF_tot = ""
        ap_fixed_COEFF_int = ""

        template_ap_fixed = \
            """ap_fixed< BBB, CCC, AP_RND, AP_SAT> """
        
        if self.is_Input_prev():
            ap_fixed_INP_int_P , ap_fixed_INP_tot_P,ap_fixed_OUT_int_P , ap_fixed_OUT_tot, ap_fixed_COEFF_int_P , ap_fixed_COEFF_tot_P = self.get_my_size(spec_node=str(self.init.net_input))
        else:
            ap_fixed_INP_int_P , ap_fixed_INP_tot_P,ap_fixed_OUT_int_P , ap_fixed_OUT_tot, ap_fixed_COEFF_int_P , ap_fixed_COEFF_tot_P = self.get_my_size(spec_node=self.prev_layers[0].name)

        ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()

        # create content file
        content_file = content_file.replace("AAA", self.name)

        tmp = template_ap_fixed.replace("BBB", str(ap_fixed_INP_tot)).replace("CCC", str(ap_fixed_INP_int))
        content_file = content_file.replace("XXX",tmp)

        if "Conv" in self.prev_layers[0].name or "Gemm" in self.prev_layers[0].name:
            mac_value_tot,mac_value_int = self.get_MAC_size()
            tmp = template_ap_fixed.replace("BBB", str(mac_value_tot)).replace("CCC", str(mac_value_int))
            content_file = content_file.replace("WWW",tmp)
            
        else:
            mac_value_tot,mac_value_int = self.get_MAC_size()
            tmp = template_ap_fixed.replace("BBB", str(mac_value_tot)).replace("CCC", str(mac_value_int))
            content_file = content_file.replace("WWW",tmp)

        template_types = template_types.replace("XXX",tmp)

        # name of the file
        name_file = "my_types_" + self.name + ".h"

        # file creation
        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)
        
        with open(os.path.join(path, self.types_file), "a") as new_file:
            new_file.write(template_types)

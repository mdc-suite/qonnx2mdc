# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

import os

from .HLSWriter import  HLSWriter

# no buffers
class ConcatWriter(HLSWriter):

    def __init__(self, reader):

        # recover data from reader node
        self.recover_data_from_reader(reader)

        # recover hyperparameters
        self.axis = self.reader.axis

# -----------------------------------------------------
# METHODS FOR GENERATING CAL FILES

    # write CAL files for the actual lauer
    def write_CAL(self, bit_size_directives, path):

        # list that will contains inputs and output of the layer
        input_list = []
        output_list = []


        for input_name in self.map_of_in_elements.keys():
            input_list.append(input_name)

        output_list.append("output_0")

        # write the CAL file corrisponding to this layer
        self.write_id_file_CAL(self.name, input_list, output_list, bit_size_directives, path)

    # write a CAL file related to the layer
    def write_id_file_CAL(self, node_name, input_list, output_list, bit_size_directives, path):

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
        for index,elem in enumerate(input_list):

            # default value
            if (bit_size_directives["DEFAULT_BIT_SIZES"]["DATATYPE"] == "float"):

                n_bits = 32

            else:

                n_bits = bit_size_directives["DEFAULT_BIT_SIZES"]["DATA"][0]

            for prim_key in bit_size_directives.keys():

                if prim_key == "Concat":

                    if (bit_size_directives["Concat"]["DATATYPE"] == "float"):

                        n_bits = 32

                    else:

                        n_bits = bit_size_directives["Concat"]["DATA"][0]

                    break

            # do it for all the inputs
            searched = self.input_

            if "Concat" in bit_size_directives.keys():

                for searched_element in searched:

                    if searched_element in bit_size_directives["Concat"].keys():

                        if (bit_size_directives["Concat"][searched_element]["DATATYPE"] == "float"):

                            n_bits = 32

                        else:

                            n_bits = bit_size_directives["Concat"][searched_element]["DATA"][0]

                        break

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
    def write_HLS(self, bit_size_directives, path):

        self.generate_layer_sizes_h_HLS(path)

        self.generate_concat_h_HLS(path)

        self.generate_concat_ccp_HLS(path)

        self.generate_my_types_h(bit_size_directives,path)

    #generate layer_size_X.h file
    def generate_layer_sizes_h_HLS(self,path):

        content_file = \
"""
#ifndef LAYER_SIZES_H
#define LAYER_SIZES_H

        #define in_s_d {}
        #define in_s_h {}
        #define in_s_w {}
        #define out_s_d {}
        #define out_s_h {}
        #define out_s_w {}        
#endif     
"""
        #CHW
        in_d,in_h,in_w = self.isizes
        out_d, out_h, out_w, = self.osizes

        content_file = content_file.format(
                                  in_d,in_h,in_w,
                                  out_d, out_h, out_w,
                                  )

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
	void AAA(BBB, stream <DATA> &output_0);
#endif
"""


        input_list = []

        for input_name in self.map_of_in_elements.keys():

            input_list.append(input_name)



        input_template = \
""" stream<DATA> &DDD"""

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

void AAA(BBB, stream <DATA> &output_0){
#pragma HLS INTERFACE ap_ctrl_none port=return

	DATA current;
	ITER i, j, k;

	for(k=0; k<in_s_h; k++){
		for(j=0; j<in_s_w; j++){
        DDD
		}
	}
}
"""


        input_list = []

        for input_name in self.map_of_in_elements.keys():

            input_list.append(input_name)

        input_template = \
""" stream<DATA> &EEE"""

        function_input = ""

        num_of_input = len(input_list)

        for index,input_element in enumerate(input_list):

            function_input += input_template.replace("EEE", input_element)

            if(index+1 != num_of_input):

                function_input += ","


        code_template = \
"""
			for(i=0; i<in_s_d; i++){
				current = FFF.read();
				output_0.write(current);
			}
"""

        function_code = ""

        for index,input_element in enumerate(input_list):

            function_code += code_template.replace("FFF", input_element)


        content_file = content_file.replace("AAA", self.name)
        content_file = content_file.replace("BBB", function_input)
        content_file = content_file.replace("DDD", function_code)


        name_file = "{}.cpp".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # generate a "my tipes" file
    def generate_my_types_h(self,bit_size_directives,path):

        template = \
"""
#ifndef MY_TYPES_S
#define MY_TYPES_S
    #include <ap_fixed.h>
    #include "layer_sizes_AAA.h"
    typedef XXX DATA;
    typedef YYY COEFF;
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

        # data
        if (bit_size_directives["DEFAULT_BIT_SIZES"]["DATATYPE"] == "float"):

            selector_DATA = "float"

        else:

            ap_fixed_DATA_tot = bit_size_directives["DEFAULT_BIT_SIZES"]["DATA"][0]
            ap_fixed_DATA_int = bit_size_directives["DEFAULT_BIT_SIZES"]["DATA"][1]
            selector_DATA = "ap_fixed"

        for prim_key in bit_size_directives.keys():

            if prim_key == "Concat":

                if (bit_size_directives["Concat"]["DATATYPE"] == "float"):

                    selector_DATA = "float"

                else:

                    ap_fixed_DATA_tot = bit_size_directives["Concat"]["DATA"][0]
                    ap_fixed_DATA_int = bit_size_directives["Concat"]["DATA"][1]
                    selector_DATA = "ap_fixed"

                break

        searched = self.input_

        if "Concat" in bit_size_directives.keys():

            for searched_element in searched:

                if searched_element in bit_size_directives["Concat"].keys():

                    if (bit_size_directives["Concat"][searched_element]["DATATYPE"] == "float"):

                        selector_DATA = "float"

                    else:

                        ap_fixed_DATA_tot = bit_size_directives["Concat"][searched_element]["DATA"][0]
                        ap_fixed_DATA_int = bit_size_directives["Concat"][searched_element]["DATA"][1]
                        selector_DATA = "ap_fixed"

                    break

        # coeff
        if (bit_size_directives["DEFAULT_BIT_SIZES"]["DATATYPE"] == "float"):

            selector_COEFF = "float"

        else:

            ap_fixed_COEFF_tot = bit_size_directives["DEFAULT_BIT_SIZES"]["COEFF"][0]
            ap_fixed_COEFF_int = bit_size_directives["DEFAULT_BIT_SIZES"]["COEFF"][1]
            selector_COEFF = "ap_fixed"

        for prim_key in bit_size_directives.keys():

            if prim_key == "Concat":

                if (bit_size_directives["Concat"]["DATATYPE"] == "float"):

                    selector_COEFF = "float"

                else:

                    ap_fixed_COEFF_tot = bit_size_directives["Concat"]["COEFF"][0]
                    ap_fixed_COEFF_int = bit_size_directives["Concat"]["COEFF"][1]
                    selector_COEFF = "ap_fixed"

                break

        # create content file
        content_file = template.replace("AAA", self.name)

        if (selector_DATA == "float"):

            content_file = content_file.replace("XXX", "float")

        else:

            tmp = template_ap_fixed.replace("BBB", str(ap_fixed_DATA_tot)).replace("CCC",
                                                                                   str(ap_fixed_DATA_int))
            content_file = content_file.replace("XXX", tmp)

        if (selector_COEFF == "float"):

            content_file = content_file.replace("YYY", "float")

        else:

            tmp = template_ap_fixed.replace("BBB", str(ap_fixed_COEFF_tot)).replace("CCC",
                                                                                    str(ap_fixed_COEFF_int))
            content_file = content_file.replace("YYY", tmp)

        # name of the file
        name_file = "my_types_" + self.name + ".h"

        # file creation
        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)
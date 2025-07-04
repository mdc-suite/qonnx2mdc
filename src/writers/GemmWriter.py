# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

import re
import os

from .HLSWriter import  HLSWriter

#no buffers
class GemmWriter(HLSWriter):

    def __init__(self, node, model, init, json_file, types_file, size_file):

        # recover data from reader node
        self.recover_data_from_reader(node, model, init, json_file)

        # recover hyperparameters
        self.gemm_transA = self.gemm_transA()
        self.gemm_transB = self.gemm_transB()
        self.pool_mode = 0
        self.types_file = types_file
        self.size_file = size_file

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
        size_MAC, _ = self.get_MAC_size()
        # add the output of the layer
        for elem in output_list:

            outputs_actor += template_output.format(size_MAC, elem)

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

        self.generate_gemm_h_HLS(path)

        self.generate_parameters_h_HLS(path)

        self.generate_gemm_ccp_HLS(path)

        self.generate_my_types_h(path)

        #self.generate_my_Input_types_h(path)

    #generate layer_size_X.h file
    def generate_layer_sizes_h_HLS(self,path):

        template = \
"""
    #define in_s_d_BBB {}
    #define in_s_h_BBB {}
    #define in_s_w_BBB {}
    #define in_s_BBB {}
    #define out_s_BBB {}    
"""

        content_file = \
"""
#ifndef LAYER_SIZES_AAA_H
#define LAYER_SIZES_AAA_H

        #define in_s_d_BBB {}
        #define in_s_h_BBB {}
        #define in_s_w_BBB {}
        #define in_s_BBB {}
        #define out_s_BBB {}        
#endif     
"""
        content_file = content_file.replace("AAA",self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        number = "g"+ number
        content_file = content_file.replace("BBB", number)
        template = template.replace("BBB", number)

        
        if (len(self.isizes)) == 2 :
            if self.isizes[1] == 1 or self.isizes[1] == -1:
                in_d = 1
                in_w = self.isizes[2]
                in_h = 1
            elif self.isizes[0] == 1 or self.isizes[0] == -1:
                in_d = 1
                in_w = self.isizes[1]
                in_h = 1
            else:
                in_d,in_h,in_w = self.isizes[1:]
        else:
            in_d,in_h,in_w = self.isizes[1:]


        # recover number_of_classes
        out_s = self.osizes[1]

        content_file = content_file.format(
                                  in_d,in_h,in_w,
                                  "in_s_d_BBB*in_s_h_BBB*in_s_w_BBB",
                                  out_s
                                  )
        template = template.format(
                                  in_d,in_h,in_w,
                                  "in_s_d_BBB*in_s_h_BBB*in_s_w_BBB",
                                  out_s
                                  )   
        content_file = content_file.replace("BBB", number)

        name_file = "layer_sizes_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

        with open(os.path.join(path, self.size_file), "a") as new_file:

            new_file.write(template)

    # generate X.h file
    def generate_gemm_h_HLS(self, path):

        content_file = \
"""
#ifndef AAA_H
#define AAA_H
	#include <hls_stream.h>
	#include "my_types_AAA.h"
	using namespace hls;
	void AAA (stream<ACT_CCC> &input_0, stream <ACT_mac> &output_0);
#endif
"""

        content_file = content_file.replace("AAA", self.name)
        
        
        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")     

        name_file = "{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    #generate parameters_X.h file
    def generate_parameters_h_HLS(self, path):

        content_file = \
"""
#ifndef GEMM_AAA_PARAMS
#define GEMM_AAA_PARAMS
    {}
	{}
#endif
"""

        content_file = content_file.replace("AAA", self.name)
        enter_id = "_"+self.node.input[1]

        # check
        #print("Shape pesi: ", self.init.parameters_values[enter_id].shape)
        #print("Valori dei pesi: ", self.init.parameters_values[enter_id])

        #bring shape and swap shape (A is height, B is width); must be swapped in order
        # to fit inside gemm layer
        A, B= self.init.parameters_values[enter_id].shape
        #weight_values = str(self.init.parameters_values[enter_id].reshape((B,A)).tolist())
        weight_values = str(self.init.parameters_values[enter_id].tolist())

        weight_values = weight_values.replace("[", "{")
        weight_values = weight_values.replace("]", "}")

        weight = \
"""
#define WEIGHT_AAA   """ + weight_values
        weight = weight.replace("AAA", self.name)
        enter_id = "_"+self.node.input[2]
        bias_values = str(self.init.parameters_values[enter_id].tolist())
        bias_values = bias_values.replace("[", "{")
        bias_values = bias_values.replace("]", "}")

        bias = \
"""
#define BIAS_AAA   """ + bias_values
        bias = bias.replace("AAA", self.name)
        content_file = content_file.format(weight, bias)

        name_file = "parameters_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

    #generate X.ccp file
    def generate_gemm_ccp_HLS(self,path):

        content_file = \
"""
#include <hls_stream.h>
#include <ap_fixed.h>

#include "my_types_AAA.h"
#include "AAA.h"
#include "parameters_AAA.h"
#include "layer_sizes_AAA.h"
using namespace hls;


void AAA(stream<ACT_CCC> &input_0, stream <ACT_mac> &output_0){
#pragma HLS INTERFACE ap_ctrl_none port=return		

const COEFF_BBB weight[in_s_BBB][out_s_BBB]	= WEIGHT_AAA;
const COEFF_BBB bias[out_s_BBB] = BIAS_AAA;
	ITER in;
	ITER hkern;
	ITER wkern;
	
	ACT_CCC in_val;
	ACT_mac curr_weight;
	ACT_mac out_val[out_s_BBB];
	
	//Init out_val
	for(wkern=0; wkern<out_s_BBB; wkern++){
		out_val[wkern] = bias[wkern];
	}
	
	//for each input data
	for(in=0; in<in_s_BBB; in++){
		hkern = ((in%in_s_d_BBB)*(in_s_h_BBB*in_s_w_BBB)) + in/in_s_d_BBB;
		//NOTA: C tra interi arrotonda per difetto in quanto scarta la parte decimale
		//In generale si potrebbe pensare di riordinare weights in funzione di come arrivano i dati
		input_0.read(in_val);
		for(wkern=0; wkern<out_s_BBB; wkern++){					
			curr_weight = weight[in][wkern];
			out_val[wkern] += curr_weight*in_val;
		}
	}
	
	for(wkern=0; wkern<out_s_BBB; wkern++){
		output_0.write(out_val[wkern]);
	}
}
"""

        content_file = content_file.replace("AAA", self.name)
        number = ''.join(filter(str.isdigit, self.name))
        number = "g"+ number
        content_file = content_file.replace("BBB", number)
        
        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")    

        name_file = "{}.cpp".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # generate a "my tipes" file
    def generate_my_types_h(self, path):


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
        number = "g"+ number

        
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
        mac_value_tot,mac_value_int = self.get_MAC_size()
        tmp = template_ap_fixed.replace("BBB", str(mac_value_tot)).replace("CCC", str(mac_value_int))
        content_file = content_file.replace("ZZZ",tmp)

        

        tmp = template_ap_fixed.replace("BBB", str(ap_fixed_COEFF_tot)).replace("CCC", str(ap_fixed_COEFF_int))
        content_file = content_file.replace("YYY", tmp)
        content_file = content_file.replace("BBB", number)
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
    

    # return hyperparameter "transA" of the "Gemm"
    def gemm_transA(self):

        # default value of "transA" as None
        transA = None

        # if the node has a "Gemm" operator
        if self.node.op_type == "Gemm":

            # for every attribute of the node
            for attr in self.node.attribute:

                # if attribute "transA" is found:
                if attr.name == "transA":

                    # save it
                    transA = attr.i

        # return "transA" node
        return transA

    # return hyperparameter "transB" of the "Gemm"
    def gemm_transB(self):

        # default value of "transA" as None
        transB = None

        # if the node has a "Gemm" operator
        if self.node.op_type == "Gemm":

            # for every attribute of the node
            for attr in self.node.attribute:

                # if attribute "transB" is found:
                if attr.name == "transB":
                    # save it
                    transB = attr.i

        # return "transB" node
        return transB
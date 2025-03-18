# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

import os

from .HLSWriter import  HLSWriter



class BatchNormalizationWriter(HLSWriter):

    def __init__(self, node, model, init, json_file):

        # recover data from reader node
        self.recover_data_from_reader(node, model, init, json_file)

        
        # recover hyperparameters
        self.Batchnormalization_params = self.batchnorm_params()
        self.batch_eps = self.Batchnormalization_params[0]
        self.batch_spatial = self.Batchnormalization_params[1]
        self.batch_momentum = self.Batchnormalization_params[2]


        

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

        
            #ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot = self.get_my_size()
            ap_fixed_INP_tot, _ = self.get_MAC_size()
            ap_fixed_OUT_tot, _ = self.get_MAC_size()
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

#-----------------------------------------------------
# METHODS FOR GENERATING HLS FILES

    # write HLS files
    def write_HLS(self, path):

        self.generate_layer_sizes_h_HLS(path)

        self.generate_batch_h_HLS(path)

        self.generate_parameters_h_HLS(path)

        self.generate_batch_ccp_HLS(path)

        self.generate_my_types_h(path)

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
        number = "b"+ number



        content_file = content_file.replace("BBB", number)
        name_file = "layer_sizes_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)
    #generate X.h file
    def generate_batch_h_HLS(self,path):

        import re

        content_file = \
"""
#ifndef BATCH_AAA_H
#define BATCH_AAA_H
	#include <hls_stream.h>
	#include "my_types_AAA.h"
	using namespace hls;
void AAA(stream< ACT_CCC > &input_0, stream <ACT_BBB> &output_0);
#endif
"""

        content_file = content_file.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        
        number = "b"+ number
        content_file = content_file.replace("BBB", number)

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
#ifndef BATCH_AAA_PARAMS
#define BATCH_AAA_PARAMS
    {}
	{}
	{}
	{}
#endif
"""


        content_file = content_file.replace("AAA", self.name)
        enter_id = "_"+self.node.input[1]
        weight_values = str(self.init.parameters_values[enter_id].tolist())

        weight_values = weight_values.replace("[","{")
        weight_values = weight_values.replace("]", "}")

        weight = \
"""
#define WEIGHT_AAA   """ + weight_values

        weight = weight.replace("AAA", self.name)
        enter_id = "_"+self.node.input[2]
        bias_values = str(self.init.parameters_values[enter_id].tolist())
        bias_values = bias_values.replace("[","{")
        bias_values = bias_values.replace("]", "}")

        bias = \
"""
#define BIAS_AAA   """ + bias_values

        bias = bias.replace("AAA", self.name)
        enter_id = "_"+self.node.input[3]
        runn_mean_values = str(self.init.parameters_values[enter_id].tolist())
        runn_mean_values = runn_mean_values.replace("[", "{")
        runn_mean_values = runn_mean_values.replace("]", "}")

        runn_mean = \
"""
#define RUNNING_MEAN_AAA   """ + runn_mean_values

        eps = 0.1
        runn_mean = runn_mean.replace("AAA", self.name)
        enter_id = "_"+self.node.input[4]
        self.init.parameters_values[enter_id] = self.init.parameters_values[enter_id] +  self.batch_eps + eps
        runn_var_values = str(self.init.parameters_values[enter_id].tolist())
        
        runn_var_values = runn_var_values.replace("[", "{")
        runn_var_values = runn_var_values.replace("]", "}")

        runn_var = \
"""
#define RUNNING_VAR_AAA   """ + runn_var_values

        runn_var = runn_var.replace("AAA", self.name)
        content_file = content_file.format(weight, bias,runn_mean,runn_var)

        name_file = "parameters_{}.h".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:
            new_file.write(content_file)

    #generate X.ccp file
    def generate_batch_ccp_HLS(self,path):

        import re

        content_file = \
"""
#include <hls_stream.h>
#include <ap_fixed.h>

#include "my_types_AAA.h"
#include "AAA.h"
#include "layer_sizes_AAA.h"
#include "parameters_AAA.h"
using namespace hls;

void AAA(stream<ACT_CCC> &input_0, stream <ACT_BBB> &output_0){
	#pragma HLS INTERFACE ap_ctrl_none port=return

	const COEFF_BBB running_mean[in_s_d_BBB] = RUNNING_MEAN_AAA;
	const COEFF_BBB running_std[in_s_d_BBB] = RUNNING_VAR_AAA; // I stored std instead of var
	const COEFF_BBB weight[in_s_d_BBB] = WEIGHT_AAA;
	const COEFF_BBB bias[in_s_d_BBB] = BIAS_AAA;
	
	COEFF_BBB mean, std, scale, B; 
    COEFF_BBB current;
    ACT_BBB out;
	ITER hin, win, pin;

	for (hin = 0; hin < in_s_h_BBB; hin++){
		for (win = 0; win < in_s_w_BBB; win++){
Loop_interno:for (pin = 0; pin < in_s_d_BBB; pin++){
				mean = running_mean[pin];
				std = running_std[pin];
				scale = weight[pin];
				B = bias[pin];
				
				current = (COEFF_BBB)input_0.read(); //in[pin][hin][win];

				out = (ACT_BBB)(((current - mean) / std) * scale + B);
				output_0.write(out);
			}
		}
	}
}
"""

        content_file = content_file.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        number = "b"+ number

        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")



        content_file = content_file.replace("BBB", number)

        name_file = "{}.cpp".format(self.name)

        with open(os.path.join(path, name_file), "w") as new_file:

            new_file.write(content_file)

    # generate a "my tipes" file
    def generate_my_types_h(self, path):

        import re

        template = \
"""
#ifndef MY_TYPES_AAA_S
#define MY_TYPES_AAA_S
    #include <ap_fixed.h>
    #include "layer_sizes_AAA.h"
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
        # initialization
        ap_fixed_DATA_tot = ""
        ap_fixed_DATA_int = ""
        ap_fixed_COEFF_tot = ""
        ap_fixed_COEFF_int = ""

        # fill the template
        content_file = template.replace("AAA", self.name)
        # fill the template
        number = ''.join(filter(str.isdigit, self.name))
        number = "b"+ number

        if self.is_Input_prev():
            content_file = content_file.replace("CCC", "in")
        else:
            
            match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
            first_letters, last_number = match.groups()
            content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")

        content_file = content_file.replace("BBB", number)

        

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

        #output
        tmp = template_ap_fixed.replace("BBB", str(ap_fixed_OUT_tot_P)).replace("CCC", str(ap_fixed_OUT_int_P))
        content_file = content_file.replace("XXX",tmp)
        
        tmp = template_ap_fixed.replace("BBB", str(ap_fixed_COEFF_tot)).replace("CCC", str(ap_fixed_COEFF_int))
        content_file = content_file.replace("YYY", tmp)

        ##############--PREVIOUS LAYER--############################

        

        #check if previous layer has coeff type
        if  ap_fixed_COEFF_tot_P:

            if self.is_Input_prev() :
                template_previous_coeff = template_previous_coeff.replace("CCC", "in")
            else:
                match = re.match(r'([a-zA-Z]+)_([0-9]+)', self.prev_layers[0].name)        
                first_letters, last_number = match.groups()
                template_previous_coeff = template_previous_coeff.replace("CCC", f"{first_letters[0].lower()}{last_number}")

            tmp = template_ap_fixed.replace("BBB", str(ap_fixed_OUT_tot_P)).replace("CCC", str(ap_fixed_OUT_tot_P))
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

# return hyperparameters: eps, spatial, momentum / of the BatchNormalization layer

    def batchnorm_params(self):

        # if ONNX node exists
        if self.node is not None:

            # initially set attributes to None
            epsilon = None
            spatial = None
            momentum = None

            # if the node is not a BatchNorm type
            if self.node.op_type == "BatchNormalization":

                # for all the attribute of the node
                for attribute in self.node.attribute:

                    # if attribute "eps" is found
                    if attribute.name == "epsilon":

                        # save it and set flag as True
                        epsilon = attribute.f

                    # otherwise set "1e-05" as default value of "eps"
                    else:
                        epsilon = 1e-05

                    # if attribute "spatial" is found
                    if attribute.name == "spatial":

                        # save it and set flag as True
                        spatial = attribute.i

                    # otherwise set "1" as default value of "spatial"
                    else: spatial = 1

                    # if attribute "momentum" is found
                    if attribute.name == "momentum":

                        # sace it and set flag as True
                        momentum = attribute.f

                    # otherwise set "0.9" as default value of "momentum"
                    else: momentum = 0.9

                #restituisci i valori dei tre iperparametri
                return epsilon, spatial, momentum

            # if ONNX node do not exists, return None for all the attributes
            else:

                return None, None, None

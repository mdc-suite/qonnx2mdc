# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

import os
import re
#--------------------------------------------------------------------------
# HLS WRITER OBJECT CLASS

class TestBenchCppWriter():

    def __init__(self, writer_node_list, path):

        self.node_list = writer_node_list
        self.path = path

        # recover data from reader node
        self.mapping_names_to_ids()
        self.generate_TB_file()
   
    # ----------------------------------------------------------------
    # METHODS FOR GENERATING XDF FILE

    # generates the XDF file
    def generate_TB_file(self):

        # return the prefix of the XDF file
        prefix_file = self.get_TB_prefix()

        # return the instances of the XDF file
        instances_file = self.get_TB_streams()

        # return connections of the XDF file
        connections_file = self.get_TB_connections()

        # return the ending of the xDF file
        ending_file = self.get_XDF_ending()

        # compose the XDF file
        XDF_file = prefix_file \
                    + instances_file \
                    + connections_file \
                    + ending_file

        # name of the directory where save the file
        #dir_name = "/actors"

        # path where put create a directory which
        # will contain XDF file
        #path = self.path + dir_name
        path = self.path 

        # create a directory for XDF file (if not existing)
        if os.path.exists(path) is not True:
            os.makedirs(path)

        # create XDF file
        with open(os.path.join(path, "Testbench_vitis_hls.cpp"), "w") as the_XDF_file:
            the_XDF_file.write(XDF_file)

    # generate XDF prefix
    def get_TB_prefix(self):

        # template to be filled
        prefix = """
        #include <hls_stream.h>
        #include <ap_fixed.h>
        #include "my_types.h"
        #include "my_sizes.h"
        #include "my_hls_video.h"
        #include <fstream>
        #include <iostream>
        #define INPUT_SIZE 28*28 //MNIST default
        #define OUTPUT_SIZE 10 //MNIST default
        using namespace hls;
        void initialize_stream_array(ACT_in* array, stream<ACT_in> &in_stream);
        void initialize_stream_file(const char* file_name, stream<ACT_in> &in_stream);
        """


        instances = ""

        instances += prefix
        # for every writer node in the net
        for writer_node in self.node_list:

            # particular care for Conv layer
            if (writer_node.operation == "Conv"):

                template_conv = """
        #include "weight_{}.h"
        #include "parameters_{}.h
        #include "pe_{}.h"
        #include "bias_{}.h
        #include "line_buffer_{}.h"
                """

                id = writer_node.name 
                instances += template_conv.format(id, id, id, id, id)

            # for other layers it is only needed the input
            elif (writer_node.operation == "BatchNormalization" or writer_node.operation == "Gemm"):

                template_ = """
        #include "parameters_{}.h"
        #include "{}.h"
                """
                id = writer_node.name 
                instances += template_.format(id, id)


            elif (writer_node.operation == "MaxPool"):

                template_mp = """
        #include "{}.h"
        #include "line_buffer_{}.h"
                """

                id = writer_node.name 
                id1 = id.replace("MaxPool","mp")
                instances += template_mp.format(id, id1)

            else:

                template = """
        #include {}.h
                """

                # input
                id = writer_node.name
                instances += template.format(writer_node.name, id)

        instances += """
        void main() {
        """
        
        # return XDF file
        return instances

    # generate XDF instances
    def get_TB_streams(self):


        # initialize variable for containing instances
        instances = ""

        template_in = """
        // Streams
        ACT_{} in_stream ("in_stream");

        """
        template_in = template_in.format("in")


        template = ""

        # template to be filled for every instance

        # for every writer node in the net
        for writer_node in self.node_list:

            var = writer_node.is_Input_prev()
             

            
            if (writer_node.operation == "Conv"):
                
                if var: 
                    type_prev = "in"
                else:
                    prev = writer_node.prev_layers 
                    match = re.match(r'([a-zA-Z]+)_([0-9]+)', prev[0].name)        
                    first_letters, last_number = match.groups()
                    content_file = "CCC"
                    content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")                
                    type_prev = content_file

                # fill the template
                number = ''.join(filter(str.isdigit, writer_node.name))
                type_stream = "c" + number
                    

                template_conv = """
            KERN_ITEM_{} weight_{}_stream ("weight_{}_stream");
            KERN_ITEM_{} bias_{}_stream ("bias_{}_stream");
            ACT_{} line_buffer_{}_stream ("line_buffer_{}_stream");
            ACT_mac pe_{}_stream ("pe_{}_stream");
"""

                # input
                id = writer_node.name

                template_conv = template_conv.format(type_stream, id, id, type_stream, id, id, type_prev, id, id, id, id)
                instances += template_conv
                #instances += template.format(id, id)
            # for other layers it is only needed the input
            else:
                template_buffer = None
                if var: 
                    type_prev = "in"
                else:
                    prev = writer_node.prev_layers 
                    match = re.match(r'([a-zA-Z]+)_([0-9]+)', prev[0].name)        
                    first_letters, last_number = match.groups()
                    content_file = "CCC"
                    content_file = content_file.replace("CCC", f"{first_letters[0].lower()}{last_number}")                
                    type_prev = content_file

                # fill the template
                number = ''.join(filter(str.isdigit, writer_node.name))
                if (writer_node.operation == "Gemm" ):
                    type_stream = "g" + number
                elif (writer_node.operation == "BatchNormalization"):
                    type_stream = "b" + number
                elif (writer_node.operation == "MaxPool"):
                    type_stream = "m" + number
                    template_buffer = """
            ACT_{} line_buffer_{}_stream ("line_buffer_{}_stream");
                    """
                elif (writer_node.operation == "GlobalAveragePool"):
                    type_stream = "gap" + number
                elif (writer_node.operation == "Sigmoid"):
                    type_stream = "s" + number
                elif (writer_node.operation == "Relu"):
                    type_stream = "r" + number

                    

                template_param = """
            ACT_{} {}_stream ("{}_stream");
      
        """
                instances += template_param.format(type_stream, writer_node.name, writer_node.name)

                if template_buffer is not None:
                    instances += template_buffer.format(type_prev, writer_node.name, writer_node.name)

                #
        # return instances
        return instances

    # generate XDF connections
    # generate XDF connections
    def get_TB_connections(self):


        # variable initialized for contening the connections
        connections = ""

        template_in = """
            initialize_stream_array_file("reshaped_input_vitis.txt", in_stream);
        """
        connections += template_in
        
        
        # template to be filled for each connection
        template_conv = """
                weight_{} (weight_{}_stream);
                bias_{} (bias_{}_stream);
                line_buffer_{} ({}_stream , line_buffer_{}_stream);
                pe_{} (line_buffer_{}_stream , pe_{}_stream, weight_{}_stream, bias_{}_stream);
        """        
        template_maxpool = \
            """
                line_buffer_{} ({}_stream , line_buffer_{}_stream);
                {} (line_buffer_{}_stream , {}_stream);"""

        template_layers = \
            """
                {} ({}_stream , {}_stream);
            """

        # for every node in the Writer net
        for writer_node in self.node_list:

            

             
            if (writer_node.operation == "Conv"):

                var = writer_node.is_Input_prev()

                id = writer_node.name

                if var: 
                    prev_layer = "in"
                else:
                    prev = writer_node.prev_layers 
                    prev_layer = prev[0].name
                

                template_conv = template_conv.format( id, id, id , id, id, prev_layer, id, id, id, id, id, id) 
                connections += template_conv

            elif (writer_node.operation == "MaxPool"):

                var = writer_node.is_Input_prev()

                id = writer_node.name

                if var: 
                    prev_layer = "in"
                else:
                    prev = writer_node.prev_layers 
                    prev_layer = prev[0].name

                template_maxpool = template_maxpool.format( id, prev_layer, id ,id, id, id)
                connections += template_maxpool
                
            else:

                var = writer_node.is_Input_prev()
                id = writer_node.name

                if var: 
                    prev_layer = "in"
                else:
                    prev = writer_node.prev_layers 
                    prev_layer = prev[0].name


                # Create a new formatted string instead of overwriting template_layers
                formatted_layer = template_layers.format(id, prev_layer, id)

                connections += formatted_layer  # Append to connections

        
        # return connections
        return connections

    # generate XDF ending
    def get_XDF_ending(self):

        # generate XDF ending string
        ending = \
            """
        }"""

        

        functions = \
            """
        
        void initialize_stream_array(ACT_in* array,stream<ACT_in> &in_stream) {

            for (int i = 0; i < SIZE; i++) {
                in_stream.write(array[i]);
            }
        }

        

        void initialize_stream_file(const char* file_name, stream<ACT_in> &in_stream){
            std::ifstream infile(filename);  // Open the file
            if (!infile) {
                std::cerr << "Error: Cannot open file " << filename << std::endl;
                return;
            }

            float value;
            while (infile >> value) {  // Read float value from the file
                ACT_in converted_value = (ACT_in) value;  // Implicit conversion from float to ap_fixed
                in_stream.write(converted_value);  // Send to the stream
            }

            infile.close();  // Close the file
        }

        
            """
        
        ending += functions

        # return string contaning XDF ending
        return ending


    

        



    def mapping_names_to_ids(self):

        """
            Maps the input and output elements of various neural network layers to their corresponding IDs.
            This method iterates over the nodes in the `self.node_list` and assigns mappings of input and output elements
            to each node based on its operation type. The supported operations include "Conv", "Gemm", "MaxPool", "Sigmoid",
            "GlobalAveragePool", "Relu", "BatchNormalization", and "Concat". For unsupported operations, an error message
            is printed.
            The mappings are stored in dictionaries `map_of_in_elements` and `map_of_out_elements` for each node, which are
            then assigned to the node's attributes `map_of_in_elements` and `map_of_out_elements`.
            The mappings are determined as follows:
            - "Conv" and "Gemm": One input and up to two parameters (weight and bias).
            - "MaxPool", "Sigmoid", and "GlobalAveragePool": One input and no parameters.
            - "Relu": One input and no parameters. If followed by a "Quant" operation, the output is modified.
            - "BatchNormalization": One input and up to four parameters (weight, bias, running_mean, running_var).
            - "Concat": Multiple inputs and no parameters.
            Returns:
                None
        """
           
            
        def mapping_function(writer_node):
        # if the layer is a Conv or a Gemm one, there is an input and
        # there can be two parameters, which are weight and bias
            
            map_of_in_elements = {}
            map_of_out_elements = {}

            if (writer_node.operation == "Conv" or writer_node.operation == "Gemm"):

                # identify input
                map_of_in_elements["input_0"] = writer_node.input_[0]

                # identify parameters
                for index, enter_id in enumerate(writer_node.parameters):

                    # define weight
                    if (index == 0):
                        map_of_in_elements["weight"] = enter_id

                    # define bias
                    if (index == 1):
                        map_of_in_elements["bias"] = enter_id

                # identify output
                map_of_out_elements["output_0"] = writer_node.output_[0]

            # if the layer is Relu, MaxPool, Sigmoid, there is an input
            # and there are no parameters
            elif (writer_node.operation == "MaxPool"
                or writer_node.operation == "Sigmoid" or writer_node.operation == "GlobalAveragePool"):

                # identify input
                map_of_in_elements["input_0"] = writer_node.input_[0]

                # identify output
                map_of_out_elements["output_0"] = writer_node.output_[0]

            elif(writer_node.operation == "Relu"):

                quant_relu = False
                for pred in writer_node.model.find_direct_successors(writer_node.node):
                    if "Quant" in pred.name:
                        quant_relu = True
                    
                
                if quant_relu:    
                    # identify input
                    map_of_in_elements["input_0"] = writer_node.input_[0]

                    # identify output
                    map_of_out_elements["output_0"] = "_"+writer_node.model.find_direct_successors(writer_node.node)[0].output[0]
                else:
                    map_of_in_elements["input_0"] = writer_node.input_[0]

                    # identify output
                    map_of_out_elements["output_0"] = writer_node.output_[0]

            # if the node is BatchNorm, I have one input and there can
            # be four possible parameters, which are weight, bias, mean, var
            #
            elif (writer_node.operation == "BatchNormalization"):

                # identify input
                map_of_in_elements["input_0"] = writer_node.input_[0]

                # identify parameters
                for index, enter_id in enumerate(writer_node.parameters):

                    # identify weights
                    if (index == 0):
                        map_of_in_elements["weight"] = enter_id

                    # identify bias
                    if (index == 1):
                        map_of_in_elements["bias"] = enter_id

                    # identify mean
                    if (index == 2):
                        map_of_in_elements["running_mean"] = enter_id

                    # identify var
                    if (index == 3):
                        map_of_in_elements["running_var"] = enter_id

                # identify output
                map_of_out_elements["output_0"] = writer_node.output_[0]

            # if the node is Concat, I can have more than one input and no parameters
            elif (writer_node.operation == "Concat"):

                # identify every input
                for index, enter_id in enumerate(writer_node.input_):
                    # give a numbered name
                    input_name = "input_" + str(index)

                    # assign it to the input
                    map_of_in_elements[input_name] = enter_id

                # identify the output
                map_of_out_elements["output_0"] = writer_node.output_[0]
                

            else:

                # fi the node type is not found, then there's something wrong
                print("[ERROR] This type of operator is not supported: ", writer_node.operation)

            return map_of_in_elements, map_of_out_elements


        # for every Writer node
        for writer_node in self.node_list:

           writer_node.map_of_in_elements, writer_node.map_of_out_elements = mapping_function(writer_node)
                


    def get_node_name_in_writer_list_from_its_output_id(self, node_list, searched_id):

        # for every writer node
        for node in node_list:

            # if the node has its output name equal to the one searched,
            # then return the node name
            if (searched_id in node.map_of_out_elements["output_0"]):
                return node.name
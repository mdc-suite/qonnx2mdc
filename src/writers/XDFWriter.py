# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

import os

#--------------------------------------------------------------------------
# HLS WRITER OBJECT CLASS

class XDFWriter():

    def __init__(self, writer_node_list, path):

        self.node_list = writer_node_list
        self.path = path

        # recover data from reader node
        self.mapping_names_to_ids()
        self.generate_XDF_file()
   
    # ----------------------------------------------------------------
    # METHODS FOR GENERATING XDF FILE

    # generates the XDF file
    def generate_XDF_file(self):

        # return the prefix of the XDF file
        prefix_XDF_file = self.get_XDF_prefix()

        # return the instances of the XDF file
        instances_XDF_file = self.get_XDF_instances()

        # return connections of the XDF file
        connections_XDF_file = self.get_XDF_connections()

        # return the ending of the xDF file
        ending_XDF_file = self.get_XDF_ending()

        # compose the XDF file
        XDF_file = prefix_XDF_file \
                    + instances_XDF_file \
                    + connections_XDF_file \
                    + ending_XDF_file

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
        with open(os.path.join(path, "XDF_file.xdf"), "w") as the_XDF_file:
            the_XDF_file.write(XDF_file)

    # generate XDF prefix
    def get_XDF_prefix(self):

        # template to be filled
        template = """<?xml version="1.0" encoding="UTF-8"?>
    <XDF name="{}">
    <Port kind="Input" name="{}">
        <Type name="int">
            <Entry kind="Expr" name="size">
                <Expr kind="Literal" literal-kind="Integer" value="{}"/>
            </Entry>
        </Type>
    </Port>            
    <Port kind="Output" name="{}">
        <Type name="int">
            <Entry kind="Expr" name="size">
                <Expr kind="Literal" literal-kind="Integer" value="{}"/>
            </Entry>
        </Type>
    </Port>"""

        # procedure for setting net input bit size in XDF file:

        # default value assigned as bit size of the net input
        

        _ , input_n_bits,_ , _, _ , _ = self.node_list[0].get_my_size(spec_node = str(self.node_list[0].init.net_input))

        # if the net input id is in the list of keys of any layer for bit
        # size directives, then, for that layer, take the input size
        #   check directives format file for better understanding

      

        # define net output bit size
        _ , _,_ , output_n_bits, _ , _ = self.node_list[0].get_my_size(spec_node = str(self.node_list[0].init.net_output))


        # fill template with name of the model, net input, input bit size, net output, output bit size
        #self.net_input_id = "input_net"
        #self.net_output_id = "output_net"
        prefix_XDF_file = template.format(self.node_list[0].model.graph.name, self.node_list[0].init.net_input, input_n_bits, self.node_list[0].init.net_output, output_n_bits)

        # return XDF file
        return prefix_XDF_file

    # generate XDF instances
    def get_XDF_instances(self):

        # initialize variable for containing instances
        instances = ""

        # template to be filled for every instance
        template = \
            """ 
    <Instance id="{}">
        <Class name="actors.{}"/>
    </Instance>"""

        # for every writer node in the net
        for writer_node in self.node_list:

            # particular care for Conv layer
            if (writer_node.operation == "Conv"):

                # input
                id = "pe_" + writer_node.name
                instances += template.format(id, id)

                # weight
                id = "weight_" + writer_node.name 
                instances += template.format(id, id)

                # bias
                id = "bias_" + writer_node.name 
                instances += template.format(id, id)

            # for other layers it is only needed the input
            else:

                # input
                id = writer_node.name
                instances += template.format(id, id)

            # create buffer later conv e maxpool
            if (writer_node.operation == "Conv"):
                # create instance of line buffer
                id_buffer = "line_buffer_" + writer_node.name
                instances += template.format(id_buffer, id_buffer)
            elif (writer_node.operation == "MaxPool"):
                # create instance of line buffer
                id_buffer = "line_buffer_" + writer_node.name
                id_buffer = id_buffer.replace("MaxPool", "mp")
                instances += template.format(id_buffer, id_buffer)
            
        # return instances
        return instances

    # generate XDF connections
    # generate XDF connections
    def get_XDF_connections(self):

        # list of all the possible parameters which can be found
        # in these layers
        parameters = "weight, bias, running_mean, running_var"

        # variable initialized for contening the connections
        connections = ""

        # template to be filled for each connection
        template = \
            """
    <Connection dst="{}" dst-port="{}" src="{}" src-port="{}"/>"""

        # for every node in the Writer net
        for index, writer_node in enumerate(self.node_list):

            # if the node is a Conv, use a particular procedure
            if (writer_node.operation == "Conv"):
               
                        
                # if the Conv has an input equal to the net input,
                # use specific procedure
                if (writer_node.map_of_in_elements["input_0"] == writer_node.init.net_input):

                    # net_input -> buffer
                    src = ""
                    src_port = writer_node.init.net_input
                    dst = "line_buffer_" + writer_node.name
                    dst_port = "input_0"

                    connections += template.format(dst, dst_port, src, src_port)

                    # buffer -> input
                    src = "line_buffer_" + writer_node.name
                    src_port = "output_0"
                    dst = "pe_" + writer_node.name
                    dst_port = "input_0"

                    connections += template.format(dst, dst_port, src, src_port)

                # if not, still use a similar but specific procedure
                elif ("Quant" in writer_node.map_of_in_elements["input_0"]):
                    #need to find the node which has the output which goes in input to the Quant layer
                    src_node_name = self.get_node_name_in_writer_list_from_its_input_id(
                        self.node_list, writer_node.map_of_in_elements["input_0"])
                    print(f"DEBUG: Inside CONV {writer_node.name} - {src_node_name}")
                else:

                    src_node_name = self.get_node_name_in_writer_list_from_its_output_id(
                        self.node_list, writer_node.map_of_in_elements["input_0"])
                    print(f"DEBUG: {writer_node.name} - {src_node_name}")
                    # prev_layer -> buffer
                    src = src_node_name
                    src_port = "output_0"
                    dst = "line_buffer_" + writer_node.name
                    dst_port = "input_0"

                    connections += template.format(dst, dst_port, src, src_port)

                    # buffer -> input
                    src = "line_buffer_" + writer_node.name
                    src_port = "output_0"
                    dst = "pe_" + writer_node.name
                    dst_port = "input_0"

                    connections += template.format(dst, dst_port, src, src_port)

                # weight -> input
                src = "weight_" + writer_node.name
                src_port = "weight_" + writer_node.name + "_r"
                dst = "pe_" +writer_node.name
                dst_port = "weight_" + writer_node.name +"_r"

                connections += template.format(dst, dst_port, src, src_port)

                # bias -> input
                src = "bias_" + writer_node.name
                src_port = "bias_" + writer_node.name +"_r"
                dst = "pe_" +writer_node.name
                dst_port = "bias_" + writer_node.name +"_r"

                connections += template.format(dst, dst_port, src, src_port)

            # if the node is a MaxPool one, still use a specific procedure
            elif (writer_node.operation == "MaxPool"):

                # if one of his input is the net input, use another specific procedure
                if (writer_node.map_of_in_elements["input_0"] == writer_node.init.net_input):

                    # net_input -> buffer
                    src = ""
                    src_port = writer_node.init.net_input
                    dst = "line_buffer_" + writer_node.name
                    dst = dst.replace("MaxPool","mp")
                    dst_port = "input_0"

                    connections += template.format(dst, dst_port, src, src_port)

                    # buffer -> input
                    src = "line_buffer_" + writer_node.name
                    src = src.replace("MaxPool","mp")
                    src_port = "output_0"
                    dst = writer_node.name
                    dst_port = "input_0"

                    connections += template.format(dst, dst_port, src, src_port)
            # if the node is a MaxPool one, still use a specific procedure
                # if not, still use a specific procedure
                else:

                    src_node_name = self.get_node_name_in_writer_list_from_its_output_id(
                        self.node_list, writer_node.map_of_in_elements["input_0"])

                    # prev_layer -> buffer
                    src = src_node_name
                    src_port = "output_0"
                    dst = "line_buffer_" + writer_node.name
                    dst = dst.replace("MaxPool","mp")
                    dst_port = "input_0"

                    connections += template.format(dst, dst_port, src, src_port)

                    # buffer -> input
                    src = "line_buffer_" + writer_node.name
                    src = src.replace("MaxPool","mp")
                    src_port = "output_0"
                    dst = writer_node.name
                    dst_port = "input_0"

                    connections += template.format(dst, dst_port, src, src_port)

            # other nodes
            #   there can be layers with more than one input
            else:

                # prepare a list that will contain the list of inputs of a certain node
                input_names = []

                # recover all the elements in "map_of_in_elements" which are not
                # parameters ("parameters" contain the name of the parameters, see
                # at the beginning of this method)
                for element in writer_node.map_of_in_elements.keys():

                    if (element not in parameters):
                        input_names.append(element)

                # for every input of the layer
                for input_name in input_names:

                    # apply specific procedure if the input of the layer is
                    # the net input
                    if (writer_node.map_of_in_elements[input_name] == writer_node.init.net_input):

                        src = ""
                        src_port = writer_node.init.net_input
                        dst = writer_node.name
                        dst_port = input_name 

                        connections += template.format(dst, dst_port, src, src_port)

                    # otherwise apply another procedure
                    else:
                        print(f"DEBUG: {writer_node.name} - {input_name} - {writer_node.map_of_in_elements[input_name]}")
                        src_node_name = self.get_node_name_in_writer_list_from_its_output_id(
                            self.node_list, writer_node.map_of_in_elements[input_name])
                        print(writer_node.name)
                        print(writer_node.map_of_in_elements)
                        print(writer_node.map_of_out_elements)
                        print(f"node list {self.node_list}, writer_node_map {writer_node.map_of_in_elements[input_name]}")
                        
                        if("Conv" in src_node_name):
                            if(("weight" not in src_node_name) and ("weight" not in src_node_name) and ("weight" not in src_node_name)):
                                src_node_name = "pe_" + src_node_name
                        src = src_node_name
                        src_port = "output_0"
                        dst = writer_node.name
                        dst_port = input_name 

                        connections += template.format(dst, dst_port, src, src_port)

        # make final connections:

        # recover index of last Writer node
        # in the net and add to the connections
        index_last_node = len(self.node_list) - 1

        src = self.node_list[index_last_node].name
        src_port = "output_0"
        dst = ""
        dst_port = writer_node.init.net_output

        if ("MaxPool" in self.node_list[index_last_node].name):
            src_port = src_port 

        connections += template.format(dst, dst_port, src, src_port)

        # return connections
        return connections

    # generate XDF ending
    def get_XDF_ending(self):

        # generate XDF ending string
        ending = \
            """
    </XDF>"""

        # return string contaning XDF ending
        return ending


    

        


    def mapping_names_to_ids(self):
            
        def mapping_function(writer_node):
        # if the layer is a Conv or a Gemm one, there is an input and
        # there can be two parameters, which are weight and bias
            
            map_of_in_elements = {}
            map_of_out_elements = {}

            if (writer_node.operation == "Conv" or writer_node.operation == "Gemm"):

                # identify input
                print(f"DEBUG map function: {writer_node.input_}")
                first_input = writer_node.input_[0]
                
                if writer_node.prev_layers[0] == writer_node.init.net_input:
                    first_input = writer_node.init.net_input
                elif "Quant" in first_input:
                    first_input = writer_node.prev_layers[0].output[0]
                map_of_in_elements["input_0"] = first_input

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
            
    def get_node_name_in_writer_list_from_its_input_id(self, node_list, searched_id):
        # for every writer node
        for node in node_list:

            # if the node has its input name equal to the one searched,
            # then return the node name
            if (searched_id in node.map_of_in_elements["input_0"]):
                return node.name
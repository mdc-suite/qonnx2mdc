# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito
# Note: adapted from ALOHA onnxparser - https://gitlab.com/aloha.eu/onnxparser/-/blob/master/Initializer.py

#--------------------------------------------------------------------------
#IMPORTS
from onnx import numpy_helper

#--------------------------------------------------------------------------
# INITIALIZER OBJECT CLASS

class Initializer:

        def __init__ (self,model):

            # save model e graph
            self.model = model
            self.graph = model.graph

            # save the list of initializers
            self.initializer = model.graph.initializer

            # map all parameters values and shape of each node to their name
            self.parameters_values = self.get_parameters_values_and_their_shapes()

            # save the net input name
            self.net_input = self.get_net_input()

            print("Net input name: ", self.net_input)

            # save the net input shape
            self.net_input_shape = self.get_net_input_shape()

            print("Net input shape: ", self.net_input_shape)

            # save the net output name
            self.net_output = self.get_net_output()

            print("Net output name: ", self.net_output)

            # save the net output shape
            self.net_output_shape = self.get_net_output_shape()

            print("Net output shape: ", self.net_output_shape)


#---------------------------------------------------------------------------
# METHODS

        # return parameter values and their shape of each node
        def get_parameters_values_and_their_shapes(self):

            values = {}

            # for every initializer of each node that has parameters
            for k in self.model.graph.initializer:

                # bring in the parameter name and do some reformat
                parameters_name = k.name
                parameters_name = parameters_name.replace("/", "_").replace(".", "_")
                parameters_name = "_{}".format(parameters_name)

                # recover the values associated to that parameter
                init_el = numpy_helper.to_array(k)

                # values are saved using the name of that input
                values[parameters_name] = init_el

                

            # return values and shapes of each parameter
            return values

        # return net input name
        def get_net_input (self):

            net_input = False

            output_list = []

            # for every node in the nodes
            for node in self.model.graph.node:

                # for every output of each node
                for j,l in enumerate(node.output):

                    # save the output node name in the list after some reformat
                    l = l.replace("/","_").replace(".","_")
                    l = "_{}".format(l)
                    output_list.append(l)

            # for every node in the nodes
            for node in self.model.graph.node:

                # for every input of each node
                for k in range (len(node.input)):

                    # save the input name after some reformat
                    node.input[k] = node.input[k]. replace("/","_").replace(".","_")
                    node.input[k] = "_{}".format(node.input[k])

                    # if the input name is not a parameter and it is not in the list of output of each node, then
                    # it is the net input name
                    if  node.input[k] not in self.parameters_values.keys() and node.input[k] not in output_list:

                        # some reformat of the name
                        net_input = node.input[k]
                        net_input = net_input.replace("/","_").replace(".","_")

                    #every input name receives some reformat
                    node.input[k] = node.input[k][1:]

            if(net_input == False):

                print("[ERROR IN INITIALIZER] Net Input not found")

            #restituisce l'id dell'input della rete
            return net_input

        # return net output name
        def get_net_output (self):

            output_name = self.graph.output[0].name
            output_name = output_name.replace("/", "_").replace(".", "_")
            output_name = "_{}".format(output_name)

            return output_name

        # return net input shape
        def get_net_input_shape(self):

            dimensions = []

            # for every input in graph.input
            for element in self.graph.input:

                # do some reformat to the name
                input_name = element.name
                input_name = input_name.replace("/", "_").replace(".", "_")
                input_name = "_{}".format(input_name)

                # if the name is the same of the net input
                if(input_name == self.net_input):

                    # for every dimension of the net input
                    for dim in element.type.tensor_type.shape.dim:

                        # save shape
                        dimensions.append(dim.dim_value)

            # return shape
            return dimensions

        # return net output shape
        def get_net_output_shape(self):

            dimensions = []

            # for every input in graph.input
            for element in self.graph.output:

                # do some reformat to the name
                input_name = element.name
                input_name = input_name.replace("/", "_").replace(".", "_")
                input_name = "_{}".format(input_name)

                # if the name is the same of the net input
                if (input_name == self.net_output):

                    # for every dimension of the net input
                    for dim in element.type.tensor_type.shape.dim:

                        # save shape
                        dimensions.append(dim.dim_value)

            # return shape
            return dimensions

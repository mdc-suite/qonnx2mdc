# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors:  
# Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
# Stefano Esposito

class HLSWriter():

    def __init__(self, node, model, init, json_file):

        # recover data from reader node
        self.recover_data_from_reader(node, model, init, json_file)

        



     # return the list of inputs/parameters of the ONNX node
    def nodes_input(self):

        # if node exists
        if self.node is not None:

            # list that will contain the input of the node
            nodes_input = []

            # list that will contain the parameters of the node
            nodes_parameters = []

            # if the node has not a Constant operator
            if self.node.op_type != "Constant":

                # for every input/parameter of the ONNX node
                for input_ in self.node.input:

                    # take the input/parameter name and do some reformat
                    input_ = input_.replace("/", "_").replace(".", "_")
                    input_ = "_{}".format(input_)

                    # if element in node.input is not in the list of parameters names,
                    # it is the actual input
                    if input_ not in self.init.parameters_values.keys():

                        # add the input name to the list of the input/parameters of the node
                        nodes_input.append(input_)

                    # other elements are just parameters, they are saved in another list
                    else:

                        # add the parameter name to the list of the input/parameters of the node
                        nodes_parameters.append(input_)
                        if self.node.op_type == "MatMul":
                            #initializer can be the input actually
                            print("matmul can have initializers as inputs, so I add it to the inputs")
                            for key,value in self.init.parameters_values.items():
                                if input_ in key:
                                    nodes_input.append(value)

            # if the node has a Constant operator
            else:

                # add "constant" to the list of inputs/parameters
                nodes_input.append("constant")

            # if the node has an Add operator
            if self.node.op_type == "Add":

                # reverse the order of the list of inputs/parameters
                nodes_input.reverse()

            # return the list of inputs/parameters of the node
            return nodes_input,nodes_parameters

        # if node do not exist, return None
        else:

            return [], []
        

    # return the output name of the node
    def nodes_output(self):

        # if the ONNX node exists:
        if self.node is not None:

            # list that will contain the output name of the node
            nodes_output = []

            # for every output of the ONNX node
            for output_ in self.node.output:

                # save the value and do some reformat
                line = output_
                output_ = line.replace("/", "_").replace(".", "_")
                output_ = "_{}".format(output_)

                #save the output name of the node
                nodes_output.append(output_)

            #return the output name of the node
            return nodes_output

        # if ONNX node does not exist, return None
        else:
            return None


    def node_isizes(self):
        return self.model.get_tensor_shape(self.node.input[0])
    
    def node_osizes(self):
        return self.model.get_tensor_shape(self.node.output[0])

    # recover data from reader node
    def recover_data_from_reader(self,node, model, init, json_file):

        
        # recover all these values from the reader node
        self.name = node.name
        self.node = node
        self.model = model
        self.operation = node.op_type
        self.init = init
        self.json_file = json_file
        
        self.prev_layers = self.find_prev_layers()

        self.input_, self.parameters = self.nodes_input()
        if self.prev_layers and "Quant" in self.prev_layers:
            self.prev_layers = self.model.find_direct_predecessors(self.prev_layers[0])
            self.input_, self.parameters = self.nodes_input()
        self.output_ = self.nodes_output()
        
        self.isizes = self.node_isizes()
        self.osizes = self.node_osizes()
        
        
        # dict that map inputs and parameters of a node
        # to a certain role
        self.map_of_in_elements = {}

        # dict that map the output of a node
        # to a certain role
        self.map_of_out_elements = {}

# find previous layer of the ONNX node
    def find_prev_layers(self):

        prev = []

        # save all the inputs and parameters name of the ONNX node
        inputs = ["_"+n for n in self.node.input]

        


        # for every Reader node in the net of reader nodes
        for layer in self.model.graph.node:

            # if the output of a Reader node is in the list of the
            # input and parameters of the ONNX node, then the Reader Node
            # is a previous layer of the ONNX node
            if "_"+layer.output[0] in inputs and "Quant" in layer.name:
                prev_layers = self.model.find_direct_predecessors(layer)
                for prv in prev_layers:
                    prev.append(prv)
            elif "_"+layer.output[0] in inputs and "Quant" not in layer.name :
                # save the Reader node
                prev.append(layer)

        # return the list of previous Reader node if not empty,
        # else return None
        return prev if prev else None

    # sort Reader nodes according to their relation IO
    #   it has the same procedure as the one in onnxparser.py
    def sort_layers(self):

        done = False

        while not done:
            done = True

            for i, layer in enumerate(self.net):

                out = layer.output_[0]

                for j, l in enumerate(self.net):

                    if out in l.input_ and j<i:
                        done = False
                        self.net[i], self.net[j] = self.net[j], self.net[i] #swap



    def get_my_size(self, spec_node = None):

        ap_fixed_COEFF_tot = ""
        ap_fixed_COEFF_int = ""
        selector_COEFF = ""

        

        # Access the fields of a particular layer, e.g., "Sigmoid_0"
        if spec_node:
            layer_data = self.json_file[spec_node]
        else:
            layer_data = self.json_file[self.name]

        # Extract the values from the "INPUT" and "OUTPUT" arrays
        
        if layer_data.get("INPUT", [None, None]):
            input_values = layer_data.get("INPUT", [None, None])
        else:
            input_values = [None, None]

        if layer_data.get("OUTPUT", [None, None]):
            output_values = layer_data.get("OUTPUT", [None, None])
        else:
            output_values = [None, None]

        if layer_data.get("COEFF", [None, None]):
            coeff_values = layer_data.get("COEFF", [None, None])
        else:
            coeff_values = [None, None]


        # Save the values into separate variables
        ap_fixed_INP_tot, ap_fixed_INP_int = input_values
        ap_fixed_OUT_tot, ap_fixed_OUT_int = output_values
        ap_fixed_COEFF_tot, ap_fixed_COEFF_int = coeff_values

        


        
        

        return ap_fixed_INP_int , ap_fixed_INP_tot,ap_fixed_OUT_int , ap_fixed_OUT_tot, ap_fixed_COEFF_int , ap_fixed_COEFF_tot
    

    def get_MAC_size(self):
        return 32,16


    def is_Input_prev(self):
        for inp in self.node.input:
            
            if "_"+str(inp) == str(self.init.net_input):
                return True
            
        return False
    

    
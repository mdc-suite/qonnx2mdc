# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper as mw
from qonnx.transformation.base import Transformation




class RemoveSqueeze(Transformation):
                  
    def apply(self, model):
        
        wrap = mw(model)

        net_input = wrap.graph.input[0].name

        for node in wrap.graph.node:
            if node.op_type == "Squeeze" or node.op_type == "Unsqueeze" :

                # Get direct successors and predecessors
                successors = wrap.find_direct_successors(node)
                predecessors = wrap.find_direct_predecessors(node)
                input_name = None

                if predecessors :
                    predecessor = predecessors[0]           
                else:
                    predecessor = None
                    
                

                if predecessor:
                    input_name = predecessor.output[0]
                else:
                    input_name = net_input
                    


                if successors:
                    successor = successors[0]
                    input_to_eliminate = successor.input[0]
                    
                else:
                    successor = None
                    input_to_eliminate = None

                if node.op_type == "Squeeze":
                    target_shape = wrap.get_tensor_shape(input_name)
                else:
                    target_shape = wrap.get_tensor_shape(input_to_eliminate)
                
                # Set the shape of the replaced tensor
                wrap.set_tensor_shape(input_name, target_shape)
                
                if successor:
                    # Modify the successor node's input connections
                    if input_to_eliminate in successor.input:
                        successor.input.remove(input_to_eliminate)
                        successor.input.insert(0, input_name)
                else:
                    print("last node")
                    predecessor.output[0] = wrap.graph.output[0].name

                # Remove the Reshape node
                wrap.graph.node.remove(node)
                
        return model, False
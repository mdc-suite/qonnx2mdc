# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper as mw
from qonnx.transformation.base import Transformation




class RemoveTranspose(Transformation):
                  
    def apply(self, model):
        
        wrap = mw(model)

        net_input = wrap.graph.input[0].name

        for node in wrap.graph.node:
            if node.op_type == "Transpose" :

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
# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper as mw
from qonnx.transformation.base import Transformation




class Set4D(Transformation):
    
    def apply(self, model):
        wrap = mw(model)
        net_input = wrap.graph.input[0].name
        
        # Create a copy of the list of nodes
        nodes = list(wrap.graph.node)

        dummy_nodes = ["Relu", "Quant"]
        
        for node in nodes:
            if node.op_type in dummy_nodes:
                # Get direct successors and predecessors
        
                input_shape = wrap.get_tensor_shape(node.input[0])

                
                output_shape = wrap.get_tensor_shape(node.output[0])

                print("Node type:", node.op_type)
                print("output_shape:", output_shape)
                print("input shape", input_shape)
                
                # Ensure 4D shape
                if output_shape != input_shape:
                    print("Dummy node has strange dimensions!")
                    
                wrap.set_tensor_shape(node.output[0], input_shape)
                # Remove the Reshape node
            elif node.op_type == "GlobalAveragePool":
                
                output_shape = wrap.get_tensor_shape(node.output[0])

                if len(output_shape) < 4:
                    wrap.set_tensor_shape(node.output[0], output_shape + [1])

                
        return model, False
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
            
            elif node.op_type == "MaxPool":

                strides = None
                kernel = None

                # Extract strides and kernel size
                for attr in node.attribute:
                    if attr.name == "strides":
                        strides = attr.ints
                    elif attr.name == "kernel_shape":
                        kernel = attr.ints

                # Check if attributes exist
                if strides is None or kernel is None:
                    continue  # Skip if missing parameters

                # Get input shape
                input_shape = wrap.get_tensor_shape(node.input[0])
                if input_shape is None or len(input_shape) < 4:
                    continue  # Skip if shape is invalid

                # Ensure it's a mutable list
                new_shape = list(input_shape)

                print("old shape:", new_shape)
                print(kernel)
                print(strides)

                # Apply MaxPool output shape formula (with integer division)
                new_shape[2] = (new_shape[2] - kernel[0]) // strides[0] + 1
                new_shape[3] = (new_shape[3] - kernel[1]) // strides[1] + 1

                print("new shape:", new_shape)
                # Update the output shape
                wrap.set_tensor_shape(node.output[0], new_shape)
            
                
        return model, False
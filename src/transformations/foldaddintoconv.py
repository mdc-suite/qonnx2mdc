# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper as mw
from qonnx.transformation.base import Transformation

class FoldAddIntoConv(Transformation):
    "We transform a Conv + Add into a Conv layer, preserving attributes and output shape."
    
    def __init__(self):
        super().__init__()
        self.applied = False  # Avoid reapplying transformation

    def apply(self, model):
        if self.applied:
            return model, False  
        
        resh_cnt = -1    
        wrap = mw(model)

        for node in wrap.graph.node:
            if node.op_type == "Conv":
                resh_cnt += 1

        for node in wrap.graph.node:
            to_fold = False

            if node.op_type == "Conv":
                # Get direct successors
                successors = wrap.find_direct_successors(node)
                for successor in successors:
                    if successor.op_type == "Add":
                        to_fold = True

                if to_fold:
                    successor = successors[0]

                    new_input = [node.input[0], node.input[1], successor.input[1]]
                    new_output = successor.output[0]

                    # Copy attributes from the old Conv node
                    conv_attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}

                    # Get the original Conv's output shape
                    old_output_shape = wrap.get_tensor_shape(node.output[0])

                    # Create new Conv node with same attributes
                    conv_node = helper.make_node(
                        "Conv",
                        inputs=new_input,
                        outputs=[new_output],
                        name="Conv_" + str(resh_cnt),
                        **conv_attributes  # Apply original attributes
                    )

                    # Set the output shape of the new Conv node
                    wrap.set_tensor_shape(new_output, old_output_shape)

                    # Replace nodes in the graph
                    wrap.graph.node.append(conv_node)
                    wrap.graph.node.remove(node)
                    wrap.graph.node.remove(successor)

        self.applied = True  
        return model, False

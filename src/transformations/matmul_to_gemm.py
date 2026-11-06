# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper as mw
from qonnx.transformation.base import Transformation


"""
class MatMul_to_Gemm(Transformation):

    "we transform a MatMul + Add into a Gemm layer"
    
    def __init__(self):
        super().__init__()
        self.applied = False  # Add a flag to track if the transformation has been applied

    def apply(self, model):
        if self.applied:
            return model, False  # Return early if the transformation has already been applied
        
        resh_cnt = 0    
        wrap = mw(model)

        for node in wrap.graph.node:
            if node.op_type == "Gemm":
                resh_cnt = resh_cnt + 1

        for node in wrap.graph.node:

            is_gemm = False

            if node.op_type == "MatMul":

                predecessors = wrap.find_direct_predecessors(node)
                     
                # Get direct successors and predecessors
                successors = wrap.find_direct_successors(node)
                for successor in successors:
                    if successor.op_type == "Add":
                        is_gemm = True

                if is_gemm:
                    successor = successors[0]
                    predecessor = predecessors[0]

                    new_input = [node.input[0], node.input[1], successor.input[1]]
                    new_output = successor.output[0]
            
                    # Create the Flatten node with specified attributes
                    output_name = "Gemm_output"+str(resh_cnt)
                    

                    gemm_node = helper.make_node(
                                                op_type = 'Gemm',
                                                inputs = new_input,
                                                outputs = [new_output],
                                                name = "Gemm_"+str(resh_cnt),
                                            )
                    
                    
                    wrap.graph.node.extend([gemm_node])
                    wrap.graph.node.remove(node)
                    wrap.graph.node.remove(successor)

        self.applied = True  # Set the flag to True after applying the transformation
        return model, False
"""

class MatMul_to_Gemm(Transformation):

    def __init__(self):
        super().__init__()
        self.applied = False

    def apply(self, model):

        if self.applied:
            return model, False

        wrap = mw(model)

        # Count existing Gemm nodes
        resh_cnt = sum(
            1 for node in wrap.graph.node
            if node.op_type == "Gemm"
        )

        # Make a copy because we are modifying graph.node
        nodes = list(wrap.graph.node)

        for node in nodes:

            if node.op_type != "MatMul":
                continue

            # Find direct successors
            successors = wrap.find_direct_successors(node)

            # find_direct_successors may also return None
            if successors is None:
                continue

            # Look specifically for MatMul -> Add
            add_node = None

            for successor in successors:
                if successor.op_type == "Add":
                    add_node = successor
                    break

            # No Add after MatMul -> nothing to convert
            if add_node is None:
                continue

            # MatMul inputs:
            #
            #   input[0] = A
            #   input[1] = B
            #
            # Add:
            #
            #   input[1] = bias
            #
            new_input = [
                node.input[0],
                node.input[1],
                add_node.input[1]
            ]

            new_output = add_node.output[0]

            gemm_node = helper.make_node(
                op_type="Gemm",
                inputs=new_input,
                outputs=[new_output],
                name="Gemm_" + str(resh_cnt)
            )

            # Add Gemm
            wrap.graph.node.extend([gemm_node])

            # Remove old MatMul and Add
            wrap.graph.node.remove(node)
            wrap.graph.node.remove(add_node)

            resh_cnt += 1

        self.applied = True

        return model, False
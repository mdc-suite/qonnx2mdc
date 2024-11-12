# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper as mw
from qonnx.transformation.base import Transformation



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
                    info = wrap.get_tensor_valueinfo(node.input[1])
                    dim1 = info.type.tensor_type.shape.dim

                    for dims in dim1:
                        dim1 = dims.dim_value
                        break

                    gemm_node = helper.make_node(
                                                op_type = 'Gemm',
                                                inputs = new_input,
                                                outputs = [new_output],
                                                name = "Gemm_"+str(resh_cnt),
                                            )
                    
                    
                    wrap.graph.node.extend([gemm_node])
                    node.input.remove(node.input[1])
                    node.input.insert(1,output_name)
                    wrap.graph.node.remove(node)
                    wrap.graph.node.remove(successor)

        self.applied = True  # Set the flag to True after applying the transformation
        return model, False

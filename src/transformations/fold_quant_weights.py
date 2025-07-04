# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)
# Note: adapted from the FINN compiler - https://github.com/Xilinx/finn/blob/main/src/finn/transformation/qonnx/fold_quant_weights.py

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import remove_node_and_rewire
from qonnx.core.datatype import FixedPointType
from math import ceil, log2

class FoldQuantWeights(Transformation):
    """Merges Quant nodes, which are used as weights into the initializer
    of the weight tensor.
    """
    

    def apply(self, model):
        data_type = "ap_fixed"
        graph = model.graph
        node_ind = 0
        graph_modified = False
        execution_context = model.make_empty_exec_context()
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Quant" or n.op_type == "BipolarQuant":
                node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input)) #It uses the map function to apply the model.get_initializer function to each element of n.input
                node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits)) #This lambda function filters out elements in node_inp_inits that are None
                node_out = n.output[0]
                is_all_constant_inputs = len(node_inp_dyn) == 0
                ishape = model.get_tensor_shape(n.input[0])
                is_const_shape = (n.op_type == "Shape") and (ishape is not None)
                if is_all_constant_inputs or is_const_shape:
                    # Check node validity
                    if (
                        n.op_type == "Quant"
                        and not model.get_initializer(n.input[2]) == 0
                    ):
                        raise ValueError(
                            "Only Quant nodes with zero-point == 0 "
                            "are currently supported."
                        )
                    if model.is_fork_node(n):
                        raise ValueError(
                            "Weights quantized with the Quant node are not "
                            "allowed to be fork nodes node."
                        )
                    target_node = model.find_direct_successors(n)
                    if target_node is None:
                        raise RuntimeError(
                            "Weights quantized with the Quant node must have "
                            "a successor node."
                        )
                    else:
                        target_node = target_node[0]
                    # If there is a DebugMarker in the weight path,
                    # then the DebugMarker needs to be removed before any further
                    # action is taken. Because this node interferes
                    # with how the constant folding tries to determine how to
                    # apply scale factors and in any case the DebugMarker would not
                    # return useful information after folding.
                    if target_node.op_type == "DebugMarker":
                        remove_node_and_rewire(model, target_node)
                        model = model.transform(FoldTransposeIntoQuantInit())
                        return model, True

                    # Continue with constant folding the quant node
                    scale = model.get_initializer(n.input[1])
                    unity_scale = (scale.flatten() == 1.0).all()
                    # this node has no dynamic inputs, only constant ones -- so we can
                    # do constant folding.
                    oxe.execute_node(n, execution_context, graph)
                    q_node_output = execution_context[node_out]
                    # Check we can directly constant fold
                    if unity_scale:
                        print("unity scale")
                        # use the execution result as an initializer
                        model.set_initializer(node_out, q_node_output)
                    else:
                        # Check next operator type
                        mul_like_nodes = ["Mul", "Div", "Conv", "MatMul", "Gather"]
                        add_like_nodes = ["Add", "Sub"]
                        all_supported_ops = mul_like_nodes.copy()
                        all_supported_ops.extend(add_like_nodes)

                        if target_node.op_type not in all_supported_ops:
                            raise ValueError(
                                f"Can't constant fold Quant weight node "
                                f"into node type {target_node.op_type} "
                                f"at node: {target_node}."
                            )

                        # For both mul and Add:
                        # Move the scale factor behind the next operator
                        scale = model.get_initializer(n.input[1])
                        fract_width = ceil(log2(1/scale))
                        
                        new_initializer = q_node_output / scale
                        # Round, to correct for floating point errors
                        new_initializer = np.round(new_initializer)
                        #Dequantize so that the scale op is not needed
                        print(f"DEBUG_frontend_foldweights: fract_width = {fract_width} - scale = {scale} - scale_fw = {2 ** -fract_width}")
                        #new_initializer = new_initializer * scale
                        new_initializer = new_initializer * 2 ** -fract_width
                        q_inst = getCustomOp(n)
                        if data_type == "ap_fixed":
                            
                            tot_width = model.get_initializer(n.input[3])
                            if fract_width > tot_width:
                                fract_width = tot_width
                            ap_fixed_type = FixedPointType(tot_width, tot_width - fract_width)
                            model.set_initializer(node_out, new_initializer)
                            model.set_tensor_datatype(node_out, ap_fixed_type)
                        else:
                            new_dtype = q_inst.get_integer_datatype(model)
                            model.set_initializer(node_out, new_initializer)
                            model.set_tensor_datatype(node_out, new_dtype)

                        # Reshape scale for Conv if required
                        target_output_shape = model.get_tensor_shape(
                            target_node.output[0]
                        )
                        if target_node.op_type == "Conv" and len(scale.shape) > 0:
                            conv_out_shape = [1] * len(target_output_shape)
                            # only support per-output channel scaling
                            # (i.e. all scale shape elems besides 0th must be 1s)
                            if len(scale.shape) > 1:
                                assert (
                                    np.prod(scale.shape[1:]) == 1
                                ), "Can't fold scale beyond per-out-channel granularity"
                            # collect all scaling in channels dim (since we constrain)
                            conv_out_shape[1] = -1
                            scale = scale.reshape(conv_out_shape)
###################################################################################################################
                        
                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
                    #model = model.transform(InferShapes())
                    return (model, graph_modified)
        return (model, graph_modified)

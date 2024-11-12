# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper as mw
from qonnx.transformation.base import Transformation




class RemoveIdentityOperations(Transformation):
    
    def apply(self, model):
        wrap = mw(model)
        net_input = wrap.graph.input[0].name
        
        # Create a copy of the list of nodes
        nodes = list(wrap.graph.node)
        
        for node in nodes:
            if node.op_type == "Identity":
                # Get direct successors and predecessors
                successors = wrap.find_direct_successors(node)
                predecessors = wrap.find_direct_predecessors(node)
                
                input_name = predecessors[0].output[0] if predecessors else net_input
                input_to_eliminate = successors[0].input[0] if successors else None
                
                if successors:
                    # Modify the successor node's input connections
                    if input_to_eliminate in successors[0].input:
                        successors[0].input.remove(input_to_eliminate)
                        successors[0].input.insert(0, input_name)
                else:
                    print("last node")
                    predecessors[0].output[0] = wrap.graph.output[0].name
                
                # Remove the Reshape node
                wrap.graph.node.remove(node)
                
        return model, False
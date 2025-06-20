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
        nodes_to_remove = []

        for node in list(wrap.graph.node):  # Convert to list to avoid modification issues
            if node.op_type == "Transpose":
                # Get direct successors and predecessors
                successors = wrap.find_direct_successors(node)
                predecessors = wrap.find_direct_predecessors(node)

                if predecessors:
                    predecessor = predecessors[0]
                    input_name = predecessor.output[0]
                else:
                    predecessor = None
                    input_name = net_input  # If no predecessor, use the network input

                if successors:
                    for successor in successors:
                        # Replace occurrences of the Transpose node's output in the successor
                        for i, inp in enumerate(successor.input):
                            if inp == node.output[0]:  # If input matches the Transpose output
                                successor.input[i] = input_name  # Replace it with the correct input

                else:
                    # If the node is the last in the graph, update the predecessor to connect to the output
                    if predecessor:
                        predecessor.output[0] = wrap.graph.output[0].name

                # Mark Transpose for removal
                nodes_to_remove.append(node)

        # Remove all marked Transpose nodes
        for node in nodes_to_remove:
            wrap.graph.node.remove(node)

        return model, False

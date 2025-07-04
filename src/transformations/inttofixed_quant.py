# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper as mw
from qonnx.transformation.base import Transformation
from qonnx.core.datatype import FixedPointType
from math import ceil, log2

class IntToFixedQuant(Transformation):
    "Convert the annotation of the Quant layer from Int to Fixed"
    
    def __init__(self):
        super().__init__()
        self.applied = False  # Avoid reapplying transformation

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Quant" or n.op_type == "BipolarQuant":
                node_out = n.output[0]
                # Continue with constant folding the quant node
                scale = model.get_initializer(n.input[1])
                fract_width = ceil(log2(1/scale))
                tot_width = model.get_initializer(n.input[3])

                if fract_width > tot_width:
                    fract_width = tot_width

                ap_fixed_type = FixedPointType(tot_width, tot_width - fract_width)
                model.set_tensor_datatype(node_out, ap_fixed_type)
                graph_modified = True


        return model, False

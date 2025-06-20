# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper as mw
from qonnx.transformation.base import Transformation




class NCL_to_4d(Transformation):
        
	def apply(self, model):

		wrap = mw(model)

		net_input = wrap.graph.input[0].name

		#NCL Format
		input_shape = wrap.get_tensor_shape(net_input)
		
		if len(input_shape) == 3:
			input_shape = np.insert(input_shape, 1, 1)


		input_shape = np.array(input_shape)

		# Transpose from NCL to 1NCL = NCHW format
		order = [0,1,2,3]

		new_shape = input_shape[order]

		print("Input shape (1NCL):", new_shape.tolist())

		wrap.set_tensor_shape(net_input,new_shape.tolist())

		return model, False

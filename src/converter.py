# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)
from qonnx.transformation.base import Transformation
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from transformations.fold_quant_weights import FoldQuantWeights
from transformations.matmul_to_gemm import MatMul_to_Gemm
from transformations.remove_transpose import RemoveTranspose
from transformations.remove_reshape import RemoveReshape
from transformations.remove_identity import RemoveIdentityOperations
from transformations.remove_flatten import RemoveFlatten
from transformations.set_nchw import SetNCHW_Shape
from transformations.remove_squeeze import RemoveSqueeze
from transformations.foldaddintoconv import FoldAddIntoConv
from transformations.inttofixed_quant import IntToFixedQuant
from qonnx.transformation.infer_shapes import InferShapes
from transformations.set_4d_shape import Set4D

import onnx

class Converter_qonnx(Transformation):
    
    def __init__(
        self
    ):
        super().__init__()
        

    def apply(self, model):
        
        
        
        
        model = model.transform(RemoveReshape())
        model = model.transform(FoldTransposeIntoQuantInit())
        model = model.transform(RemoveTranspose())
        model = model.transform(RemoveSqueeze())
        model = model.transform(FoldAddIntoConv())
        
        
        
        model = model.transform(RemoveReshape())
        model = model.transform(RemoveTranspose())
        model = model.transform(RemoveSqueeze())
        model = model.transform(FoldAddIntoConv())
        
        model = model.transform(FoldQuantWeights())
        model = model.transform(IntToFixedQuant())
        model = model.transform(MatMul_to_Gemm())
        model = model.transform(RemoveIdentityOperations())
        model = model.transform(RemoveFlatten())
        #model = model.transform(SetNCHW_Shape())
        model = model.transform(Set4D())
        model = model.transform(InferShapes())
        
        
        

        return model, False
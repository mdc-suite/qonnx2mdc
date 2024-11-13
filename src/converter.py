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



class Converter_qonnx(Transformation):
    
    def __init__(
        self
    ):
        super().__init__()
        

    def apply(self, model):
        
        model = model.transform(FoldTransposeIntoQuantInit())
        model = model.transform(SetNCHW_Shape())
        model = model.transform(FoldQuantWeights())
        model = model.transform(RemoveTranspose())
        model = model.transform(RemoveReshape())
        model = model.transform(MatMul_to_Gemm())
        model = model.transform(RemoveIdentityOperations())
        model = model.transform(RemoveFlatten())
        
        return model, False
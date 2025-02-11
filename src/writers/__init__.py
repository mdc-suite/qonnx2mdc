# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)

from .HLSWriter import HLSWriter

from .ConvWriter import ConvWriter
from .MaxPoolWriter import MaxPoolWriter
from .ReluWriter import ReluWriter
from .GemmWriter import GemmWriter
from .BatchNormalizationWriter import BatchNormalizationWriter
from .SigmoidWriter import SigmoidWriter
from .ConcatWriter import ConcatWriter
from .GlobalAveragePoolWriter import GlobalAveragePoolWriter


mapping = {"Conv": ConvWriter,
           "Gemm": GemmWriter,
           "MaxPool": MaxPoolWriter,
           "Relu": ReluWriter,
           "BatchNormalization": BatchNormalizationWriter,
           "Sigmoid": SigmoidWriter,
           "Concat": ConcatWriter,
           "GlobalAveragePool": GlobalAveragePoolWriter
           }

skipped_operators = ["Input"]
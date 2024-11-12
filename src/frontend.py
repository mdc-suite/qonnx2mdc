# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)

import os
import sys
import onnx
import tensorflow as tf
from qonnx.core.modelwrapper import ModelWrapper as mw
from converter import Converter_qonnx


def frontend(onnx_path = "None"):

    tf.keras.backend.clear_session()
    print("TensorFlow version:", tf.__version__)
    print("Onnx version:", onnx.__version__)

 
    script_path = os.path.abspath(__file__)

    current_directory = os.path.dirname(script_path)

    print("Current working directory:", current_directory)


    if onnx_path != "None":
        if not os.path.exists(onnx_path):
            print(f"Error: The provided path '{onnx_path}' does not exist.", file=sys.stderr)
            sys.exit(1)
        else:
            model_path = onnx_path
            print(f"The provided path '{onnx_path}' exist.", file=sys.stderr)       
    else:
        print(f"The provided path '{onnx_path}' is not valid.", file=sys.stderr)  
        sys.exit(1)

   
    #--------------LOAD MODEL------------------#

    onnx_model = onnx.load(model_path)

    qonnx_model = mw(onnx_model)

    #---------------------------Optimization for QONNX Model----------------------------------------#

    directory = os.path.dirname(model_path)
    path = directory +'/new_qonnx_model.onnx'
 
    converter = Converter_qonnx()

    new_qonnx_model, _ = converter.apply(qonnx_model)

    new_qonnx_model.cleanup()

    new_qonnx_model.save(path)

    return new_qonnx_model

   





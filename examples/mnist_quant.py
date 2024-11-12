# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Federico Manca (<name>.<surname>@unica.it)


#############################################################
# Examples script to train a tiny CNN on the MNIST dataset
# Various quantization levels can be selected
#############################################################

import keras
import tensorflow as tf


import qkeras

tf.keras.backend.clear_session()
print("TensorFlow version:", tf.__version__)
print("QKeras version:", qkeras.__version__)
print("Keras version:", keras.__version__)
import os

#If available, use GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Specify the folder name
folder_name = 'Mnist_Training'

script_path = os.path.abspath(__file__)
# Get the current working directory
current_directory = os.path.dirname(script_path)

# Print the current working directory
print("Current working directory:", current_directory)


# Create the full path to the new folder
output_path = current_directory + "/" + folder_name

# Check if the folder already exists
if not os.path.exists(output_path):
    # Create the folder
    os.makedirs(output_path)
    print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")

print(output_path)


################################## LOADING AND PREPROCESSING ONNX MODEL#######################################################################
#.......FLAGS.........
mnist_baseline = False
mnist16_8 = False
mnist16_4 = False
mnist8_8 = False
mnist8_4 = False
mnist4_4 = False
mnist4_2 = False
mnist2_2 = False
mnist_bin = False
mnist_hls4ml = True

#This flag is responsible for the activation or deactivation of the transformation of the qkeras model to a qonnx model
convert_to_qonnx = True

#.........................
# Load ONNX model

from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import MaxPool2D, Conv2D, Flatten, Dense, Activation
from keras.models import Sequential


from keras.datasets import mnist

# Prepare the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
mnist_input_shape = (1,28,28,1)


            

if mnist_baseline:

    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=mnist_input_shape[1:]))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.7))
    model.add(Dense(10,activation = "sigmoid"))

    batch_size = 128
    epochs = 15

    with tf.device('/GPU:0'):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        model.save( output_path +'/mnist_baseline.keras')

elif mnist16_8:

    from keras.layers import *
    from qkeras import *
    from qkeras.qlayers import QDense, QActivation, quantized_bits, quantized_relu
    from qkeras import QConv2D
    from keras.models import Model

    x = x_in = Input(mnist_input_shape[1:])

    x = (QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=8, integer=4, alpha=1),
        bias_quantizer= quantized_bits(bits=8, integer=4, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=16, integer=8, use_sigmoid=0, negative_slope=0.0), name="act_1"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=8, integer=4, alpha=1),
        bias_quantizer= quantized_bits(bits=8, integer=4, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=16, integer=8, use_sigmoid=0, negative_slope=0.0), name="act_2"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(Flatten())(x)
    x =(Dropout(0.5))(x)
    x = QDense((10),kernel_quantizer= quantized_bits(bits=8, integer=4, alpha=1),
        bias_quantizer= quantized_bits(bits=8, integer=4, alpha=1)) (x)   # num_classes = 10
    x =(Activation(activation='sigmoid', name='out_activation'))(x)

    model = Model(inputs=x_in, outputs=x)

    batch_size = 128
    epochs = 15

    with tf.device('/GPU:0'):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        model.save( output_path +'/mnist16_8.keras')
    
elif mnist16_4:

    from keras.layers import *
    from qkeras import *
    from qkeras.qlayers import QDense, QActivation, quantized_bits, quantized_relu
    from qkeras import QConv2D
    from keras.models import Model

    x = x_in = Input(mnist_input_shape[1:])

    x = (QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=16, integer=8, use_sigmoid=0, negative_slope=0.0), name="act_1"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=16, integer=8, use_sigmoid=0, negative_slope=0.0), name="act_2"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(Flatten())(x)
    x =(Dropout(0.6))(x)
    x = QDense((10),kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)) (x)   # num_classes = 10
    x =(Activation(activation='sigmoid', name='out_activation'))(x)

    model = Model(inputs=x_in, outputs=x)

    batch_size = 128
    epochs = 40

    with tf.device('/GPU:0'):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        model.save( output_path +'/mnist16_4.keras')

elif mnist8_8:

    from keras.layers import *
    from qkeras import *
    from qkeras.qlayers import QDense, QActivation, quantized_bits, quantized_relu
    from qkeras import QConv2D
    from keras.models import Model

    x = x_in = Input(mnist_input_shape[1:])

    x = (QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=8, integer=4, alpha=1),
        bias_quantizer= quantized_bits(bits=8, integer=4, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=8, integer=4, use_sigmoid=0, negative_slope=0.0), name="act_1"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=4, integer=2, use_sigmoid=0, negative_slope=0.0), name="act_2"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(Flatten())(x)
    x =(Dropout(0.5))(x)
    x = QDense((10),kernel_quantizer= quantized_bits(bits=8, integer=4, alpha=1),
        bias_quantizer= quantized_bits(bits=8, integer=4, alpha=1)) (x)   # num_classes = 10
    x =(Activation(activation='sigmoid', name='out_activation'))(x)

    model = Model(inputs=x_in, outputs=x)

    batch_size = 128
    epochs = 15

    with tf.device('/GPU:0'):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        model.save( output_path +'/mnist8_8_mxd.keras')

elif mnist8_4:

    from keras.layers import *
    from qkeras import *
    from qkeras.qlayers import QDense, QActivation, quantized_bits, quantized_relu
    from qkeras import QConv2D
    from keras.models import Model

    x = x_in = Input(mnist_input_shape[1:])

    x = (QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=8, integer=4, use_sigmoid=0, negative_slope=0.0), name="act_1"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=8, integer=4, use_sigmoid=0, negative_slope=0.0), name="act_2"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(Flatten())(x)
    x =(Dropout(0.5))(x)
    x = QDense((10),kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)) (x)   # num_classes = 10
    x =(Activation(activation='sigmoid', name='out_activation'))(x)

    model = Model(inputs=x_in, outputs=x)

    batch_size = 128
    epochs = 15

    with tf.device('/GPU:0'):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        model.save( output_path +'/mnist8_4.keras')

elif mnist4_4:

    from keras.layers import *
    from qkeras import *
    from qkeras.qlayers import QDense, QActivation, quantized_bits, quantized_relu
    from qkeras import QConv2D
    from keras.models import Model

    x = x_in = Input(mnist_input_shape[1:])

    x = (QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=4, integer=2, use_sigmoid=0, negative_slope=0.0), name="act_1"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=4, integer=2, use_sigmoid=0, negative_slope=0.0), name="act_2"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(Flatten())(x)
    x =(Dropout(0.5))(x)
    x = QDense((10),kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)) (x)   # num_classes = 10
    x =(Activation(activation='sigmoid', name='out_activation'))(x)

    model = Model(inputs=x_in, outputs=x)

    batch_size = 128
    epochs = 40

    with tf.device('/GPU:0'):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        model.save( output_path +'/mnist4_4.keras')

elif mnist4_2:

    from keras.layers import *
    from qkeras import *
    from qkeras.qlayers import QDense, QActivation, quantized_bits, quantized_relu
    from qkeras import QConv2D
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping

    weight_decay = 0
    x = x_in = Input(mnist_input_shape[1:])

    x = (QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=2, integer=1, alpha=1),
        bias_quantizer= quantized_bits(bits=2, integer=1, alpha=1),kernel_regularizer=regularizers.l2(weight_decay)))(x)
    x =(QActivation(quantized_relu(bits=4, integer=2, use_sigmoid=0, negative_slope=0.0), name="act_1"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=2, integer=1, alpha=1),
        bias_quantizer= quantized_bits(bits=2, integer=1, alpha=1),kernel_regularizer=regularizers.l2(weight_decay)))(x)
    x =(QActivation(quantized_relu(bits=4, integer=2, use_sigmoid=0, negative_slope=0.0), name="act_2"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(Flatten())(x)
    x =(Dropout(0.05))(x)
    x = QDense((10),kernel_quantizer= quantized_bits(bits=2, integer=1, alpha=1),
        bias_quantizer= quantized_bits(bits=2, integer=1, alpha=1),kernel_regularizer=regularizers.l2(weight_decay)) (x)   # num_classes = 10
    x =(Activation(activation='sigmoid', name='out_activation'))(x)

    model = Model(inputs=x_in, outputs=x)

    batch_size = 128
    epochs = 40

    with tf.device('/GPU:0'):
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate = 0.001), metrics=["accuracy"])
        early_stopping_monitor = EarlyStopping(
                                        monitor='accuracy',
                                        min_delta=0,
                                        patience=4,
                                        verbose=0,
                                        mode='auto',
                                        baseline=None,
                                        restore_best_weights=True
                                    )

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25, callbacks = [early_stopping_monitor])

        model.save( output_path +'/mnist4_2.keras')

elif mnist2_2:

    from keras.layers import *
    from qkeras import *
    from qkeras.qlayers import QDense, QActivation, quantized_bits, quantized_relu
    from qkeras import QConv2D
    from keras.models import Model
    from keras.optimizers import Adam

    weight_decay = 0
    x = x_in = Input(mnist_input_shape[1:])

    x = (QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=2, integer=1, alpha=1),
        bias_quantizer= quantized_bits(bits=2, integer=1, alpha=1),kernel_regularizer=regularizers.l2(weight_decay)))(x)
    x =(QActivation(quantized_relu(bits=2, integer=1, use_sigmoid=0, negative_slope=0.0), name="act_1"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=2, integer=1, alpha=1),
        bias_quantizer= quantized_bits(bits=2, integer=1, alpha=1),kernel_regularizer=regularizers.l2(weight_decay)))(x)
    x =(QActivation(quantized_relu(bits=2, integer=1, use_sigmoid=0, negative_slope=0.0), name="act_2"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(Flatten())(x)
    #x =(Dropout(0.3))(x)
    x = QDense((10),kernel_quantizer= quantized_bits(bits=16, integer=8, alpha=1),
        bias_quantizer= quantized_bits(bits=2, integer=1, alpha=1),kernel_regularizer=regularizers.l2(weight_decay)) (x)   # num_classes = 10
    x =(Activation(activation='sigmoid', name='out_activation'))(x)

    model = Model(inputs=x_in, outputs=x)

    batch_size = 128
    epochs = 40

    with tf.device('/GPU:0'):
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate = 0.01), metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25)

        model.save( output_path +'/mnist_mxd_16_2.keras')

elif mnist_bin:

    from keras.layers import *
    from qkeras import *
    from qkeras.qlayers import QDense, QActivation, quantized_bits, quantized_relu
    from qkeras import QConv2D
    from keras.models import Model

    x = x_in = Input(mnist_input_shape[1:])

    x = (QConv2D(32, (3,3), padding='same',kernel_quantizer= binary(use_01=False, alpha=1, use_stochastic_rounding=False),
        bias_quantizer= binary(use_01=False, alpha=1, use_stochastic_rounding=False)))(x)
    x =(QActivation(binary(use_01=False, alpha=1, use_stochastic_rounding=False), name="act_1"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(QConv2D(32, (3,3), padding='same',kernel_quantizer= binary(use_01=False, alpha=1, use_stochastic_rounding=False),
        bias_quantizer= binary(use_01=False, alpha=1, use_stochastic_rounding=False)))(x)
    x =(QActivation(binary(use_01=False, alpha=1, use_stochastic_rounding=False), name="act_2"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(Flatten())(x)
    x =(Dropout(0.5))(x)
    x = QDense((10),kernel_quantizer= binary(use_01=False, alpha=1, use_stochastic_rounding=False),
        bias_quantizer= binary(use_01=False, alpha=1, use_stochastic_rounding=False)) (x)   # num_classes = 10
    x =(Activation(activation='sigmoid', name='out_activation'))(x)

    model = Model(inputs=x_in, outputs=x)

    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        ]

    batch_size = 128
    epochs = 60

    with tf.device('/GPU:0'):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks = callbacks)

        model.save( output_path +'/mnist_binary.keras')

elif mnist_hls4ml:

    from keras.layers import *
    from qkeras import *
    from qkeras.qlayers import QDense, QActivation, quantized_bits, quantized_relu
    from qkeras import QConv2D
    from keras.models import Model

    x = x_in = Input(mnist_input_shape[1:])

    x = (QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=4, integer=2, use_sigmoid=0, negative_slope=0.0), name="act_1"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(QConv2D(32, (3,3), padding='same',kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)))(x)
    x =(QActivation(quantized_relu(bits=4, integer=2, use_sigmoid=0, negative_slope=0.0), name="act_2"))(x)
    x =(MaxPool2D(pool_size=(2,2)))(x)
    x =(Flatten())(x)
    x =(Dropout(0.5))(x)
    x = QDense((10),kernel_quantizer= quantized_bits(bits=4, integer=2, alpha=1),
        bias_quantizer= quantized_bits(bits=4, integer=2, alpha=1)) (x)   # num_classes = 10
    x =(Activation(activation='sigmoid', name='out_activation'))(x)

    model = Model(inputs=x_in, outputs=x)

    batch_size = 128
    epochs = 40

    with tf.device('/GPU:0'):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        model.save( output_path +'/mnist4_4.keras')





######################---------------------------Qonnx_from_Keras----------------------------------------##############################
#This script part is responsible of morfing the keras/qkeras model 

if convert_to_qonnx:

    from qonnx.converters import from_keras

    #we have ptq_model, converted in qkeras, and k_model, just in keras 

    path = output_path +'/qonnx_model.onnx'
    print("conversion to qonnx...")
    qonnx_model, _  = from_keras(
        model,
        name="qkeras_to_qonnx_converted",
        input_signature=None,
        opset=None,
        custom_ops=None,
        custom_op_handlers=None,
        custom_rewriter=None,
        inputs_as_nchw=None,
        extra_opset=None,
        shape_override=None,
        target=None,
        large_model=False,
        output_path = path,
    )
# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Francesco Ratto, Federico Manca (federico.manca@unica.it)

import tensorflow as tf
tf.keras.backend.clear_session()
print("TensorFlow version:", tf.__version__) # add debug mode
import json
import onnx
import os
import os
from datetime import datetime
from qonnx.core.modelwrapper import ModelWrapper as mw
from writers.Initializer import Initializer
from qonnx.custom_op.registry import getCustomOp


#check how to make this imports correctly
#maybe a python package with a __init__.py file is required
from writers import ConvWriter, GemmWriter, ReluWriter, BatchNormalizationWriter, MaxPoolWriter, SigmoidWriter, GlobalAveragePoolWriter , ConcatWriter# Import your layer writer functions
from writers.XDFWriter import XDFWriter
from writers.TCLWriter import TCLWriter
from writers.TestBenchCppWriter import TestBenchCppWriter
from writers.TestBenchCppWriter import TestBenchCppWriter



def backend(path_output = "None", model_qonnx = "None"):

    if path_output:
        # Specify the output path
        outputs_folder = "/example_output"
        output_path = path_output + outputs_folder
    else:
        # Specify the output path
        outputs_folder = "/example_output"
        output_path = os.path.expanduser('~/qonnx2mdc_outputs') + outputs_folder

    # Check if the folder exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Folder '{output_path}' created successfully.")
    else:
        # Create a folder with the current date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path + f"_{timestamp}"
        os.makedirs(output_path)
        print(f"Folder '{output_path}' created successfully.")
    #--------------LOAD MODEL------------------#
    qonnx_model = model_qonnx

    cal_fold = "/CAL"
    cpp_fold = "/Cpp"
    hls_fold = "/HLS"

    path_cal = output_path + cal_fold
    path_cpp = output_path + cpp_fold
    path_hls = output_path + hls_fold

    if not os.path.exists(path_cal):
        os.makedirs(path_cal)
        print(f"Folder '{cal_fold}' created successfully.")
    else:
        print(f"Folder '{cal_fold}' already exists.")

    if not os.path.exists(path_cpp):
        os.makedirs(path_cpp)
        print(f"Folder '{cpp_fold}' created successfully.")
    else:
        print(f"Folder '{cpp_fold}' already exists.")

    if not os.path.exists(path_hls):
        os.makedirs(path_hls)
        print(f"Folder '{hls_fold}' created successfully.")
    else:
        print(f"Folder '{hls_fold}' already exists.")


    #---------------------------Optimization for QONNX Model----------------------------------------#

    init = Initializer(qonnx_model)

    writeJson(qonnx_model,output_path + "/json.json", init)
    with open(output_path + "/json.json", "r") as json_file:
        json_data = json.load(json_file)


    write_cpp_equivalent(qonnx_model, init, json_data, path_cal, path_cpp, output_path)

    generate_SYNTH_files(path_hls)


def write_cpp_equivalent(onnx_model, init, json_file, path_cal, path_cpp, output_path):
    # Load the ONNX model
    
    
    # Dictionary mapping ONNX operation types to layer writer functions
    layer_writers = {
        "Conv": ConvWriter,  
        "Gemm": GemmWriter,  
        "Relu": ReluWriter,  
        "BatchNormalization": BatchNormalizationWriter,
        "MaxPool": MaxPoolWriter,
        "Sigmoid": SigmoidWriter,
        "GlobalAveragePool": GlobalAveragePoolWriter,
        "Concat": ConcatWriter
        # Add more mappings for other layer types as needed
    }

    relu_check = False
    writer_list = []
    count = 0
# Iterate through each node in the graph
    for node in onnx_model.graph.node:
        node.domain= "qonnx.custom_op.general"

        # Check if the node represents a ReLU operation
        if node.op_type == "Relu":
            relu_check = True
            # Here you can add your logic to handle the ReLU node
            # For example, you can check its successors to see if a quantization layer follows
            # If so, extract parameters and skip the quantization layer
        # Check the type of the node and call the corresponding writer function
        writer_function = layer_writers.get(node.op_type)
        if writer_function:
            
            path_writers_cpp = path_cpp + "/" + node.name
            if not os.path.exists(path_writers_cpp):
                # Create the folder
                os.makedirs(path_writers_cpp)


            if count == 0:
                count += 1
                types_file = path_cpp + "/my_types.h"
                size_file = path_cpp + "/my_sizes.h"
                types_tmp = """
#ifndef MY_TYPES_H
#define MY_TYPES_H
#include <ap_fixed.h>
#include "my_sizes.h" 

    typedef ap_fixed< 32, 16, AP_RND, AP_SAT>  ACT_mac;
    typedef ap_fixed< 32, 16, AP_RND, AP_SAT>  ACT_in;

                """
                size_tmp = """
#ifndef MY_SIZES_H
#define MY_SIZES_H
                """
                # file creation
                with open(types_file, "w") as new_file:
                    new_file.write(types_tmp)
                
                # file creation
                with open(size_file, "w") as new_file:
                    new_file.write(size_tmp)

                Writer = writer_function(node, onnx_model, init, json_file, types_file, size_file) 

            else:
                Writer = writer_function(node, onnx_model, init, json_file, types_file, size_file)            
                
            tmp_end = """
#endif
                """
            
            Writer.write_CAL(path_cal)  
            Writer.write_HLS(path_writers_cpp)
            writer_list.append(Writer)
            

            
            
            
        elif node.op_type == "Quant" and relu_check:
            relu_check = False
        else:
            print(f"Unsupported layer type: {node.op_type}")
        
         

    XDFWriter_ = XDFWriter(writer_list, path_cal)

    TCLWriter_ = TCLWriter(writer_list, output_path)

    TestBenchCppWriter_ = TestBenchCppWriter(writer_list, path_cpp)

    with open(types_file, "a") as new_file:
                new_file.write(tmp_end)
        
    # file creation
    with open(size_file, "a") as new_file:
        new_file.write(tmp_end)   

        
def parse_value(value_str):
    import re
    if value_str == "BINARY":
        return 1
    elif value_str == "FLOAT32":
        return 32
    else:
        # Extract the integer value after the "INT" string using regular expression
        match = re.search(r'INT(\d+)', str(value_str))
        if match:
            return int(match.group(1))
        else:
            # Handle case if pattern not found
            raise ValueError("Invalid input format")

def get_tensor_sizes(predecessor, output_info):
    
    if predecessor:
        predecessor_info = output_info.get(predecessor.name, {})
        return predecessor_info.get("OUTPUT", None)
    return None

#initial json prototype: some layers cannot be quantized so the directives will include them directly
def writeJson(onnx_model,path, init, default_precision = [32,16]):
        import json
        mac_size = [32,16]
    
        model = onnx_model
        output_file_path = path

        input_name = model.graph.input[0].name
        output_name = model.graph.output[0].name

        # Initialize variables to store the extracted information
        output_info = {}
        

        output_info[init.net_input]={
                "OP_TYPE": "Input",
                "DATATYPE": "ap_fixed",
                "INPUT": default_precision
                }
        output_info[init.net_output]={
                "OP_TYPE": "Output",
                "DATATYPE": "ap_fixed",
                "OUTPUT": mac_size
                }

    


        # Iterate through the keys and values in the quantizer_config dictionary
        for node in model.graph.node:
            if node.op_type != "Quant" and node.op_type != "BipolarQuant":
                predecessors = onnx_model.find_direct_predecessors(node)

                if not predecessors:
                    predecessor = init.net_input
                    prev_layer_size = default_precision
                else:
                    predecessor = predecessors[0]
                    if predecessor.op_type == "Quant" or predecessor.op_type == "BipolarQuant":
                        predecessor = onnx_model.find_direct_predecessors(predecessor)[0]
                    prev_layer_size = get_tensor_sizes(predecessor, output_info)
            else:
                prev_layer_size = default_precision


            if node.op_type == "Quant" or node.op_type == "BipolarQuant":
                print("skipping quant")
            elif node.op_type == "BatchNormalization":
                output_info[node.name]={
                "OP_TYPE": node.op_type,
                "DATATYPE": "ap_fixed",
                "INPUT": prev_layer_size,
                "COEFF": default_precision,
                "OUTPUT": default_precision
                }
            elif node.op_type == "Sigmoid":
                output_info[node.name]={
                "OP_TYPE": node.op_type,
                "DATATYPE": "ap_fixed",
                "INPUT": prev_layer_size,
                "OUTPUT": mac_size
                }

            elif node.op_type == "Gemm" or node.op_type == "Conv":
                bit_width = onnx_model.get_tensor_datatype(node.input[1])
                bit_width = parse_value(bit_width)
                output_info[node.name]={
                "OP_TYPE": node.op_type,
                "DATATYPE": "ap_fixed",
                "INPUT": prev_layer_size,
                "COEFF": [bit_width, int(bit_width / 2)],
                "OUTPUT": mac_size     
                }
            elif node.op_type == "MaxPool" or node.op_type == "GlobalAveragePool":
                output_info[node.name]={
                "OP_TYPE": node.op_type,
                "DATATYPE": "ap_fixed",
                "INPUT": prev_layer_size,
                "OUTPUT": prev_layer_size
                }

            elif node.op_type == "Concat":
                output_info[node.name]={
                "OP_TYPE": node.op_type,
                "DATATYPE": "ap_fixed",
                "INPUT": prev_layer_size,
                "OUTPUT": prev_layer_size
                }
            elif node.op_type == "Relu":
                successors = onnx_model.find_direct_successors(node)
                successor = successors[0]

                if successor.op_type == "Quant":
                    tensor_name = successor.input[3]
                    for initializer in init.initializer:
                        if initializer.name == tensor_name:
                            bit_width = onnx.numpy_helper.to_array(initializer)
                            bit_width = int(bit_width)
                            

                    
                else:
                    bit_width = default_precision[0]

                output_info[node.name]={
                "OP_TYPE": node.op_type,
                "DATATYPE": "ap_fixed",
                "INPUT": mac_size,
                "OUTPUT": [bit_width, int(bit_width/2)]
                }

        print(output_info)
        
        with open(output_file_path, "w") as json_file:
            json.dump(output_info, json_file, indent=4)

        print(f"Information has been written to '{output_file_path}'")


    








    # generate the .v FIFO file:
def generate_SYNTH_files(path):
    

    # create a folder in the path (if not existing)
    if os.path.exists(path) is not True:
        os.makedirs(path)

    generate_FIFO_file(path)

# generate a default FIFO file for
    # net generation
def generate_FIFO_file(path):

    # name of the file
    name_file = "fifo_gen.v"

    # content of the file
    content_file = \
        """
        `timescale 1ns / 1ps

//-------------------------------------------------
// NB: fifio must have 2^n locations 
//-------------------------------------------------

module fifo_gen #(
width = 16,
depth = 4096
) (
output  full_n,
input  [width-1:0] din,
input write,

output  empty_n,
output  reg [width-1:0] dout,
input read,

input ap_clk,
input ap_rst 
);
wire wr_clk, rd_clk;

assign wr_clk = ap_clk;
assign rd_clk = ap_clk;

wire empty, full;

assign full_n = !full;
assign empty_n = !empty;


//--------------------------------------------------
// the write and read pointers over-dimensioned
// by 1 bit with respect to the number of addresses tp generate full and empty
// [Cummings SNUG 2002]
//--------------------------------------------------


    reg read_c, write_c;
    reg empty_c;
    reg [$clog2(depth):0] read_addr, write_addr;
    reg [width-1:0] inputD, outputD, inputR;
    wire [width-1:0] outputR;

    ram #(
        .depth(depth),
        .size(width)
    ) mem (
        .clock(wr_clk),
        .data(inputR),
        .write_address(write_addr[$clog2(depth)-1:0]),
        .read_address(read_addr[$clog2(depth)-1:0]),
        .we(write),
        .q(outputR)
    );

    always@(posedge wr_clk)
        begin
            write_c <= write;
            read_c <= read;
            empty_c <= empty;
            outputD <= inputD;
        end

    always@(*)
        if( (empty_c) && (read_c==1'b1) && (write_c==1'b1) )
            begin
                inputD = din;
                inputR = {width{1'bx}};
                dout = outputD;
            end
        else
            begin
                inputD = {width{1'bx}};
                inputR = din;
                dout = outputR;
            end  

//------------------------------------------
// reading address handling

always@(posedge rd_clk)
        if (ap_rst)
            read_addr <= 0;
        else if(read && !empty)
                read_addr <= read_addr+1; 

//---------------------------------------------                     
// writing address handling
    always@(posedge wr_clk)
    if (ap_rst)
        write_addr <= 0;
    else if(write && !full)
        write_addr = write_addr+1; 

//---------------------------------------------
// empty generation


assign empty = (read_addr == write_addr) ? 1 : 0;

//----------------------------------------------
// full generation


assign full = ((read_addr[$clog2(depth)-1:0] == write_addr[$clog2(depth)-1:0])
            && (read_addr[$clog2(depth)] != write_addr[$clog2(depth)]))    ? 1 : 0;


//----------------------------------------------

endmodule



module ram #(
depth = 16,
size = 16
) (
input clock,
input [size-1:0] data,
input [$clog2(depth)-1:0] write_address,
input [$clog2(depth)-1:0] read_address,
input we,
output [size-1:0] q
);
reg [size-1:0] ram_block [0:depth-1];
always@(posedge clock)
    if(we)
        ram_block[write_address] <= data;

assign q = ram_block[read_address];

endmodule
        """

    # file creation
    with open(os.path.join(path, name_file), "w") as new_file:
        new_file.write(content_file)


def run_vitis_hls(script_file):
    import subprocess

    # Command to run Vitis HLS
    command = ['vitis_hls', '-f', script_file]
    
    # Run the command
    try:
        subprocess.run(command, check=True)
        print("Vitis HLS completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error running Vitis HLS:", e)


    
# Run Vitis HLS
#run_vitis_hls(script_file)



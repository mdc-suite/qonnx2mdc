# QONNX2MDC 


# Introduction

Python frontend to generate reconfigurable CNN accelerators on FPGAs from QONNX models.

It leverages the MDC tool and Vitis HLS.


# Requirements
    * git
    * miniconda 

It is needed to create a virtual environment with the packates included in the requirements.txt file (see Step 3). 
This tutorial has been tested with Python 3.10.10.

## Step 1: Install Git
Windows: 
    https://git-scm.com/download/win

Linux: 
    sudo apt install git

## Step 2: Install Miniconda 
Windows: 

    download installer from https:https://docs.anaconda.com/free/miniconda/  (follow the guide and remember to Add to PATH)

Linux: 

    download installer from https://docs.anaconda.com/free/miniconda/

    Make the installer executable: chmod +x Miniconda3-latest-Linux-x86_64.sh

    Run the installer: ./Miniconda3-latest-Linux-x86_64.sh

    Follow the prompts on the installer: Press Enter to review the license agreement, then Enter to scroll through it, and type yes to accept it.

    Choose the installation location (default is typically recommended).

    Initialize Miniconda: after the installation is complete, you'll be asked if you want to initialize Miniconda. Type yes and press Enter. Close and reopen your terminal or run the following command to activate the changes: source ~/.bashrc

## Step 3: Create a virtual environment
conda create -n myenv python=3.10.10

## Step 4: Activate the virtual environment from the cmd line
conda activate myenv

## Step 5: Git clone the repository and branch to the right version
git clone https://github.com/mdc-suite/qonnx2mdc.git

## Step 6: Install the packages
pip install -r requirements.txt


# Application
The three scripts mnist_tutorial_mdc.ipynb, fronted.py and backend.py are responible, respectively, to Train a model and transforming it from Qkeras to the QONNX format, taking the QONNX model and transform it into a intermidiate format
accepted by the Writer, which ultimately takes the model and translates it into cpp files. These files can then be fed to Vitis HLS, which synthetizes them and makes them available for 
the MDC tool to be managed. MDC gives as output a .tcl file that can be used by Vivado to generate a Neural Network Accelerator, which can then be ported into an FPGA device. The Writer and Parser ar executed with the Qonnx_Parser.py script.



## Step 1: 
cd examples

## Step 2: Run jupyter notebook and select mnist_tutorial_mdc.ipynb
jupyter notebook


## Step 3: Run the qonnx2mdc script, pointing out the right paths
python ./qonnx2mdc.py /path/to/onnx_model /path/to/destination_folder

## (Optional) Step 4 Synthetize the example model with Vitis_HLS:
Go to ./Conv0_example and run the TCL Script.

On Linux: 

vitis_hls -f TCL_file.tcl

On Windows: 

Open the Vitis_HLS command line and navigate to the Conv0_example folder and run the following command

vitis_hls -f TCL_file.tcl

After the synthesis is done, go to Conv0_example/Conv_0/Conv_0/sultion1/sytnh/report and open pe_Conv_0_csytnh.rpt



# QONNX2MDC 


# Introduction

Python library to design reconfigurable CNN accelerators on FPGAs from QONNX models.

It leverages the MDC tool and Vitis HLS as backends for generating the accelerator hardware description.


# Installation
    * git
    * miniconda 


You need to create a virtual environment with the packets included in the requirements.txt file (see Step 3). 
This tutorial has been tested with Python 3.10.10.

## Step 1: Install Git
Windows: 
    https://git-scm.com/download/win

Linux: 
    sudo apt install git

## Step 2: Install Miniconda 
Windows: 

    download the installer from https:https://docs.anaconda.com/free/miniconda/  (follow the guide and remember to Add to PATH)

Linux: 

    download the installer from https://docs.anaconda.com/free/miniconda/

    Make the installer executable: chmod +x Miniconda3-latest-Linux-x86_64.sh

    Run the installer: ./Miniconda3-latest-Linux-x86_64.sh

    Follow the prompts on the installer: Press Enter to review the license agreement, then Enter to scroll through it, and type yes to accept it.

    Choose the installation location (default is typically recommended).

    Initialize Miniconda: after the installation is complete, you'll be asked if you want to initialize Miniconda. Type yes and press Enter. Close and reopen your terminal or run the following command to activate the changes: source ~/.bashrc

## Step 3: Create a virtual environment

```
conda create -n myenv python=3.10.10
```

## Step 4: Activate the virtual environment from the cmd line

```
conda activate myenv
```

## Step 5: Git clone the repository
```
git clone https://github.com/mdc-suite/qonnx2mdc.git
cd qonnx2mdc
```

## Step 6: Install the packages
```
pip install -r requirements.txt
```



_____________________________________________________________


# Tutorial


## Step 1: 
```
cd examples
```

## Step 2: Run jupyter notebook and select mnist_tutorial_mdc.ipynb
This step trains a model with QKeras and exports it in QONNX.

```
jupyter notebook
```


## Step 3: Run the qonnx2mdc script, pointing out the right paths
This steps generates the source files for MDC and Vitis HLS.

```
python ./qonnx2mdc.py /path/to/onnx_model /path/to/destination_folder
```

## (Optional) Step 4 Synthetize an example layer with Vitis HLS 
A Vitis HLS installation is required for this step.

Go to ./Conv0_example and run the TCL Script.

On Linux: 

```
vitis_hls -f TCL_file.tcl
```

On Windows: 

Open the Vitis_HLS command line and navigate to the Conv0_example folder and run the following command

```
vitis_hls -f TCL_file.tcl
```

After the synthesis is done, go to Conv0_example/Conv_0/Conv_0/sultion1/sytnh/report and open pe_Conv_0_csytnh.rpt

_________________________________________________________________
# Acknowledgements

Funded by the European Union, by grant No. 101135183. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

# Citations
If you use this work in your research, please consider citing the following paper: 

```bibtex
@inproceedings{manca2025onnx,
  author    = {Federico Manca and Francesco Ratto and Francesca Palumbo},
  title     = {ONNX-to-Hardware Design Flow for Adaptive Neural-Network Inference on FPGAs},
  booktitle = {Embedded Computer Systems: Architectures, Modeling, and Simulation -- Proceedings of the 24th International Conference, SAMOS 2024, Part II},
  pages     = {85--96},
  year      = {2025},
  publisher = {Springer},
  address   = {Berlin, Heidelberg},
  doi       = {10.1007/978-3-031-78380-7_7},
  url       = {https://doi.org/10.1007/978-3-031-78380-7_7},
  keywords  = {Convolutional Neural Networks, Approximate Computing, FPGAs, Cyber-Physical Systems},
  location  = {Samos, Greece}
}
```

# QONNX2MDC 


# Introduction

Python library to design reconfigurable CNN accelerators on FPGAs from QONNX models.

It leverages the MDC tool and Vitis HLS as backends for generating the accelerator hardware description.


# Installation
    * docker

You need to build a Docker image with the provided Dockerfile.
This tutorial has been tested with Python 3.10.10.

## Step 1: Git clone the repository
```
git clone https://github.com/mdc-suite/qonnx2mdc.git
```

# Step 2: Install Docker
Windows: 
    https://docs.docker.com/desktop/setup/install/windows-install/#install-docker-desktop-on-windows

Linux: 
    https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

## Step 3: Build Docker Image
	Open a bash shell and cd to the directory where the Dockerfile is located `/absolute_path/qonnx2mdc`.
	Then build the image with the command `docker build ./ -t <user/project>:<version>`. For example, `docker build ./ -t jonguti/qonnx2mdc:latest`
	When it finishes, type `docker images` and your built image should appear.
	
## Step 4: Running the container from the image
	Open a bash shell and type docker run -it --rm -v `/absolute_path/qonnx2mdc:/absolute_path/qonnx2mdc -w /absolute_path/qonnx2mdc -p 8888:8888 -e TZ=Europe/Madrid <user/project>:<version>

_____________________________________________________________


# Tutorial


## Step 1: 
```
cd examples
```

## Step 2: Run jupyter notebook and select mnist_tutorial_mdc.ipynb
This step trains a model with QKeras and exports it in QONNX.

```
python3 -m notebook --ip 0.0.0.0 --no-browser --allow-root
```
Then copy to a browser the URL that starts with http://127.0.0.1:8888/...


## Step 3: Run the qonnx2mdc script, pointing out the right paths
This steps generates the source files for MDC and Vitis HLS.

```
python3 ./qonnx2mdc.py /path/to/onnx_model /absolute_path/destination_folder
```

## (Optional) Step 4 Synthetize an example layer with Vitis HLS 
A Vitis HLS installation is required for this step.
This step must be run outside the container.
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

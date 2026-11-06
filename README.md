# QONNX2MDC

Python tool that turns a quantized [QONNX](https://github.com/fastmachinelearning/qonnx) model into the sources needed to build a CNN accelerator on FPGA.

It generates:

- **HLS C++** actors, one per layer, for [Vitis HLS](https://www.xilinx.com/products/design-tools/vitis/vitis-hls.html).
- **CAL** actors and an **XDF** dataflow network for the [MDC](https://mdc-suite.github.io/) tool. MDC can merge several networks into one reconfigurable datapath.
- A Vitis HLS **TCL** script and a C++ **testbench**.
- A Verilog FIFO (`fifo_gen.v`) for connecting the generated actors.

## How it works

```
QONNX model ──> frontend ──> backend ──> CAL/ + Cpp/ + HLS/ + TCL + XDF + testbench
               (src/frontend.py,   (src/backend.py,
                src/converter.py)   src/writers/)
```

1. **Frontend** loads the model and runs graph transformations (`src/transformations/`): removes Squeeze, Reshape, Transpose, Identity and Flatten nodes, folds Add into Conv, folds quantized weights, converts integer quantization to fixed point, turns MatMul into Gemm, and forces 4D shapes. The cleaned model is saved as `new_qonnx_model.onnx` next to the input model.
2. **Backend** infers the fixed-point precision of each layer (`json.json`) and calls one writer per node (`src/writers/`).

### Supported layers

| ONNX op | Writer |
|---|---|
| `Conv` | `ConvWriter` |
| `Gemm` (and `MatMul`, converted) | `GemmWriter` |
| `Relu` | `ReluWriter` |
| `BatchNormalization` | `BatchNormalizationWriter` |
| `MaxPool` | `MaxPoolWriter` |
| `GlobalAveragePool` | `GlobalAveragePoolWriter` |
| `Sigmoid` | `SigmoidWriter` |
| `Concat` | `ConcatWriter` |

`Quant` nodes set layer precision and are not turned into actors. Any other op is skipped with an `Unsupported layer type` message.

## Installation

Tested with Python 3.10.10 on Linux. For a Docker setup, see [README_Docker.md](README_Docker.md).

```bash
git clone --recurse-submodules https://github.com/mdc-suite/qonnx2mdc.git
cd qonnx2mdc

conda create -n qonnx2mdc python=3.10.10
conda activate qonnx2mdc
pip install -r requirements.txt
```

Optional tools:

- **Vitis HLS**: synthesize and simulate the generated C++.
- **MDC**: build the reconfigurable dataflow from the CAL/XDF output.

No conda? Get Miniconda from <https://docs.anaconda.com/free/miniconda/>.

## Usage

Run the script from `src/`, because it uses local imports:

```bash
cd src
python qonnx2mdc.py /path/to/model.onnx /path/to/output_dir
```

Output tree:

```
/path/to/output_dir/example_output[_<timestamp>]/
├── json.json           # per-layer fixed-point precision
├── TCL_file.tcl        # Vitis HLS project script
├── CAL/                # CAL actors + XDF_file.xdf
├── Cpp/                # one folder per layer, my_types.h, my_sizes.h, Testbench_vitis_hls.cpp
└── HLS/fifo_gen.v
```

If `example_output` already exists, a new folder with a timestamp suffix is created.

### Synthesize with Vitis HLS

```bash
cd /path/to/output_dir/example_output
vitis_hls -f TCL_file.tcl
```

On Windows, run the same command from the Vitis HLS command prompt.

## Tutorial: MNIST

1. Open the notebook `examples/mnist_tutorial_mdc .ipynb` (the file name has a space before `.ipynb`). It trains a small CNN with QKeras and exports it to QONNX.
   ```bash
   cd examples
   jupyter notebook
   ```
2. Convert the exported model:
   ```bash
   cd ../src
   python qonnx2mdc.py ../examples/<exported_model>.onnx ../qonnx2mdc_outputs
   ```
3. Optional: synthesize with Vitis HLS (see above).

More notebooks in `examples/`:

- `reconfigurable_mnist_tutoirial.ipynb`: exports two MNIST models to merge in one reconfigurable MDC build.
- `inception_net.ipynb`: Inception-style network with `Concat`.
- `IEEE_ESL/IEEE_ESL_example.ipynb`: how the models in the IEEE ESL paper were obtained (A8-W8, Mixed-2).

## Vision Transformer (work in progress)

The `VIT_branch` branch adds Vision Transformer support.

- `external/vit_hls`: git submodule with the hand-written ViT HLS library.
- `ViT_example/`: quantized ViT notebook, HLS sources and a Vitis HLS script (`tcl_file.tcl`).

The converter does not yet map ViT QONNX graphs to these HLS blocks. Progress is tracked in [FEATURES.md](FEATURES.md).

## Project layout

```
src/
├── qonnx2mdc.py        # CLI entry point
├── frontend.py         # load + transform model
├── converter.py        # transformation pipeline
├── backend.py          # precision inference + writer dispatch
├── transformations/    # QONNX graph passes
└── writers/            # per-layer HLS/CAL writers, XDF, TCL, testbench
examples/               # tutorials and notebooks
ViT_example/            # ViT HLS example
external/vit_hls/       # ViT HLS submodule
```

## Acknowledgements

Funded by the European Union, by grant No. 101135183. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

## License

BSD 3-Clause. See [LICENSE](LICENSE).

## Citation

If you use this work in your research, please cite:

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

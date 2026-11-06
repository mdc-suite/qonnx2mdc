# FEATURES TRACKER

# Features

- HARDWARE
    - [x] Precompute 1/std for BatchNorm layer (needs fix: see BatchNorm item below)
    - [ ] BatchNorm: fold to `x * scale + shift` with `scale = gamma / sqrt(var + eps)`
    - [ ] SiLU activation (`Sigmoid` + `Mul`, or one fused actor)
    - [ ] Elementwise `Add` and `Mul` of two streams (residual connections)
    - [ ] `Resize` / nearest-neighbour Upsample
    - [ ] MaxPool with stride 1 and padding
    - [ ] Stream fan-out: one producer, several consumers
    - [ ] Several `Concat` layers in one network
    - [ ] Multi-output networks
    - [ ] Consider packing more data together for higher parallelization (for now, stream one data at time)
    - [ ] Consider tiling and partial array buffering
    - [ ] Support configurable bit widths (B as template parameter)

- SOFTWARE
    - [ ] Conv and Gemm without bias
    - [ ] Node names like `/model.0/conv/Conv` (PyTorch / Brevitas exports)
    - [ ] Graph passes that rewire every consumer, not only the first one
    - [ ] Add unit tests for network and partial layers
    - [ ] Add golden references for HLS - Python (ONNX Runtime)
    - [ ] Consider at least a basic NAS to set PE and SIMD features <-- Future work

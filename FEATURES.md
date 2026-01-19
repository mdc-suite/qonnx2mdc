# FEATURES TRACKER


# Features

- HARDWARE
    - [x] Basic softmax implementation (Q × Kᵀ + A × V)
    - [x] Denominator accumulation (DA) with safe rescaling --> implicit rescaling as written in https://github.com/pulp-platform/ITA.git
    - [x] Possibly precompute 1/std to alleviate computation for BatchNorm layer
    - [x] Test the whole system
    - [x] Implement Add&Norm layer (using BatchNorm)
    - [x] Implement the whole MultiHeadAttention structure (BatchNorm, FFN, ..)
    - [ ] Consider packing more data together for higher parallelization (for now, stream one data at time)
    - [ ] Consider tiling and partial array buffering 
    - [ ] Implement masking to support Decoder architecture
    - [ ] Serial divider for denominator inversion (DI) --> need to check best way to implement it in HLS
    - [ ] Parallelization of Q × Kᵀ (PE-level optimization)
    - [ ] Support configurable bit widths (B as template parameter)
    - [ ] Consider the implementation of binary layers 
    - [ ] Consider level of parallelization of Heads 

- SOFTWARE
    - [ ] Add unit tests for network and partial layers
    - [ ] Consider at least a basic NAS to set PE and SIMD features <-- Future work
    - [ ] At least basic support of ViT Architecture for QONNX2MDC  <-- Work in progress
    - [ ] Add golden references for HLS - Python (for ONNX Runtime, still need work)
# Ideas

- In the QONNX, considerate three cases (or more) for Reshape layers:
    - Input size == Output size --> remove reshape (usually the transposition is handled by the HLS code naturally)
    - Input size > Output size  --> remove reshape and set size as Output size (flattening carried out naturally by streaming nature of HLS layers)
    - Input size < Output size  --> in ViT cases, we are splitting a tensor, so implement a custom operator (can be called HeadSplit) so that the HLS can infer X heads 

- I am pondering between two solutions for supporting ViT's QONNX format:
    - Identify a whole MultiHeadAttention, checking for patterns like:
        
        Q path: Gemm → Reshape → Transpose ───────┐
                                                │
        K path: Gemm → Reshape → Transpose ──┐    ├→ MatMul → Div → Softmax → MatMul → Reshape → Transpose
                                            └────┘
        V path: Gemm → Reshape → Transpose ────────────────────────────────────────────────┘

        The Reshape first serves as a *head splitter*, while the second one does the *head merging*
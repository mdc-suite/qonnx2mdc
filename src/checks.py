# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)

"""Sanity checks run on the model after the frontend passes, before any code is generated."""


def check_model(model):
    """Raise ValueError listing every shape inconsistency the writers would turn into wrong hardware."""
    errors = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            _check_conv(model, node, errors)
        elif node.op_type == "Gemm":
            _check_gemm(model, node, errors)
    if errors:
        raise ValueError("Model inconsistent after frontend:\n  " + "\n  ".join(errors))


def _check_conv(model, node, errors):
    in_shape = model.get_tensor_shape(node.input[0])
    w_shape = model.get_tensor_shape(node.input[1])
    group = next((a.i for a in node.attribute if a.name == "group"), 1)
    if in_shape is None or w_shape is None or len(in_shape) != 4 or len(w_shape) != 4:
        errors.append(f"{node.name}: Conv needs 4D input and weight, got {in_shape} and {w_shape}")
    elif in_shape[1] != w_shape[1] * group:
        errors.append(
            f"{node.name}: input channels {in_shape[1]} != weight channels {w_shape[1]} * group {group} "
            f"(input shape {in_shape}, weight shape {w_shape})"
        )
    init = model.get_initializer(node.input[1])
    if init is not None and w_shape is not None and list(init.shape) != list(w_shape):
        errors.append(f"{node.name}: weight initializer shape {list(init.shape)} != declared {list(w_shape)}")


def _check_gemm(model, node, errors):
    in_shape = model.get_tensor_shape(node.input[0])
    w_shape = model.get_tensor_shape(node.input[1])
    trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
    if in_shape is None or w_shape is None or len(w_shape) != 2:
        errors.append(f"{node.name}: Gemm needs a 2D weight, got input {in_shape} and weight {w_shape}")
        return
    in_features = 1
    for d in in_shape[1:]:
        in_features *= d
    w_in = w_shape[1] if trans_b else w_shape[0]
    if in_features != w_in:
        errors.append(f"{node.name}: input features {in_features} != weight input size {w_in} (transB={trans_b})")

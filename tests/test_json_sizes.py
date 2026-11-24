"""Every layer's INPUT precision must equal the OUTPUT precision of the layer that feeds it.

Run: cd tests && python -m pytest   (or call the test function directly)
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = os.path.join(os.path.dirname(__file__), "..", "Mnist_Training", "qonnx_model_1.onnx")


def build_json(model_path):
    """Run frontend + writeJson the way backend() does; return (model, json dict)."""
    import json
    from frontend import frontend
    from backend import writeJson
    from writers.Initializer import Initializer

    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "m.onnx")
        shutil.copy(model_path, local)
        model = frontend(local)          # frontend saves next to the model, hence the temp copy
        init = Initializer(model)
        out = os.path.join(tmp, "json.json")
        writeJson(model, out, init)
        with open(out) as f:
            return model, json.load(f)


def mismatches(model, sizes):
    """List (layer, its INPUT, feeding layer, that layer's OUTPUT) for every disagreement."""
    bad = []
    for node in model.graph.node:
        if node.op_type in ("Quant", "BipolarQuant"):
            continue
        preds = model.find_direct_predecessors(node)
        if not preds:
            continue
        pred = preds[0]
        if pred.op_type in ("Quant", "BipolarQuant"):   # a Quant is folded into the layer before it
            qp = model.find_direct_predecessors(pred)
            if not qp:
                continue
            pred = qp[0]
        if sizes[node.name]["INPUT"] != sizes[pred.name]["OUTPUT"]:
            bad.append((node.name, sizes[node.name]["INPUT"], pred.name, sizes[pred.name]["OUTPUT"]))
    return bad


def test_input_precision_matches_previous_output():
    model, sizes = build_json(MODEL)
    assert mismatches(model, sizes) == []


if __name__ == "__main__":
    m, s = build_json(MODEL)
    for b in mismatches(m, s):
        print("MISMATCH", b)
    print("done")

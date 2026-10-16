# Collects a compiled model from AI Hub output as the app loads it: NAME.onnx
# (an EPContext wrapper, pointed at ./NAME.bin), NAME.bin (the QNN context
# binary) and metadata.json with each input's and output's type and, for a
# quantized one, its uint16 quantization, in the layout of Qualcomm's own
# release metadata. NAME defaults to unet; packaging a second model into the
# same OUT adds it to the existing metadata.json.
#   python package.py SRC OUT [NAME]
import hashlib, json, os, shutil, sys
import onnx
from onnx import numpy_helper

src, out = sys.argv[1], sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else "unet"
os.makedirs(out, exist_ok=True)

wrappers = []
for root, _, files in os.walk(src):
    for f in files:
        if f.endswith(".onnx"):
            p = os.path.join(root, f)
            m = onnx.load(p, load_external_data=False)
            if any(n.op_type == "EPContext" for n in m.graph.node):
                wrappers.append((p, m))
print("EPContext models:", [p for p, _ in wrappers])
found = [w for w in wrappers if name in w[0].lower()] or wrappers
if len(found) != 1:
    sys.exit(f"expected one compiled {name}, found {[p for p, _ in found]}")
path, model = found[0]

ctx = next(n for n in model.graph.node if n.op_type == "EPContext")
attr = next(a for a in ctx.attribute if a.name == "ep_cache_context")
bin_src = os.path.join(os.path.dirname(path), attr.s.decode())
shutil.copyfile(bin_src, os.path.join(out, f"{name}.bin"))
attr.s = f"./{name}.bin".encode()
onnx.save(model, os.path.join(out, f"{name}.onnx"))

inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
consts = {}
for n in model.graph.node:
    if n.op_type == "Constant":
        consts[n.output[0]] = numpy_helper.to_array(next(a for a in n.attribute if a.name == "value").t)
val = lambda name: inits[name] if name in inits else consts[name]
DT = {onnx.TensorProto.UINT16: "uint16", onnx.TensorProto.UINT8: "uint8", onnx.TensorProto.FLOAT: "float32",
      onnx.TensorProto.INT32: "int32", onnx.TensorProto.FLOAT16: "float16"}

def tensor(vi, node):
    t = vi.type.tensor_type
    d = {"shape": [x.dim_value for x in t.shape.dim], "dtype": DT.get(t.elem_type, str(t.elem_type))}
    if node is not None:
        d["quantization_parameters"] = {"scale": float(val(node.input[1]).reshape(-1)[0]),
                                        "zero_point": int(val(node.input[2]).reshape(-1)[0])}
    return d

ins = {vi.name: tensor(vi, next((n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] == vi.name), None))
       for vi in model.graph.input}
outs = {vi.name: tensor(vi, next((n for n in model.graph.node if n.op_type == "QuantizeLinear" and n.output[0] == vi.name), None))
        for vi in model.graph.output}
meta_path = os.path.join(out, "metadata.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {
    "model_id": os.environ.get("MODEL_ID", "stable_diffusion_v1_5_liberte_pixelartredmond"), "precision": "w8a16",
    "runtime": "precompiled_qnn_onnx", "model_files": {}}
meta["model_files"][f"{name}.onnx"] = {"inputs": ins, "outputs": outs}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))

with open(os.path.join(out, "SHA256SUMS"), "w") as f:
    for fname in sorted(os.listdir(out)):
        if fname in ("SHA256SUMS", "LICENSE-NOTICE.md"):
            continue
        h = hashlib.sha256()
        with open(os.path.join(out, fname), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        f.write(f"{h.hexdigest()}  {fname}\n")
print(open(os.path.join(out, "SHA256SUMS")).read())

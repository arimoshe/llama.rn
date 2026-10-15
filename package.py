# Collects the compiled UNet from `qai-hub-models export` output as the app
# loads it: unet.onnx (an EPContext wrapper, pointed at ./unet.bin), unet.bin
# (the QNN context binary) and metadata.json with the uint16 quantization of
# each input and output, in the layout of Qualcomm's own release metadata.
import hashlib, json, os, shutil, sys
import onnx
from onnx import numpy_helper

src, out = sys.argv[1], sys.argv[2]
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
unet = [w for w in wrappers if "unet" in w[0].lower()] or wrappers
if len(unet) != 1:
    sys.exit(f"expected one compiled UNet, found {[p for p, _ in unet]}")
path, model = unet[0]

ctx = next(n for n in model.graph.node if n.op_type == "EPContext")
attr = next(a for a in ctx.attribute if a.name == "ep_cache_context")
bin_src = os.path.join(os.path.dirname(path), attr.s.decode())
shutil.copyfile(bin_src, os.path.join(out, "unet.bin"))
attr.s = b"./unet.bin"
onnx.save(model, os.path.join(out, "unet.onnx"))

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
meta = {"model_id": "stable_diffusion_v1_5_pixelartredmond", "precision": "w8a16", "runtime": "precompiled_qnn_onnx",
        "model_files": {"unet.onnx": {"inputs": ins, "outputs": outs}}}
with open(os.path.join(out, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))

with open(os.path.join(out, "SHA256SUMS"), "w") as f:
    for name in sorted(os.listdir(out)):
        if name == "SHA256SUMS":
            continue
        h = hashlib.sha256()
        with open(os.path.join(out, name), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        f.write(f"{h.hexdigest()}  {name}\n")
print(open(os.path.join(out, "SHA256SUMS")).read())

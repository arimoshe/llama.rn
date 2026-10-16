# IP-Adapter's image encoder for the 8 Elite NPU: exported to ONNX
# (image_encoder.py), quantized w8a16 by an AI Hub quantize job on the
# calibration portraits (preprocessed as the app does), compiled by an AI Hub
# compile job as precompiled QNN ONNX, and downloaded to OUT/compiled.
#   python build_image_encoder.py ENCODER_DIR PORTRAITS_DIR OUT
import glob, os, sys, zipfile
import numpy as np
import onnx
import qai_hub as hub
import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection
sys.path.insert(0, os.path.dirname(__file__))
from clip_preprocess import clip_input
from image_encoder import ImageEncoder

encoder_dir, portraits_dir, out = sys.argv[1:4]
src = f"{out}/image_encoder"  # a directory: one .onnx and its .data
os.makedirs(src, exist_ok=True)

model = ImageEncoder(CLIPVisionModelWithProjection.from_pretrained(encoder_dir)).eval()
images = [clip_input(Image.open(p)) for p in sorted(glob.glob(f"{portraits_dir}/*.png"))]
print(f"{len(images)} calibration images")
with torch.no_grad():
    want = model(torch.from_numpy(images[0])).numpy()
    torch.onnx.export(model, (torch.from_numpy(images[0]),), f"{src}/tmp.onnx", input_names=["image"],
                      output_names=["image_embeds"], opset_version=17, dynamo=False)
m = onnx.load(f"{src}/tmp.onnx")
for f in os.listdir(src):
    os.remove(f"{src}/{f}")
onnx.save(m, f"{src}/image_encoder.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
          location="image_encoder.data")
del m

import onnxruntime as ort  # noqa: E402
got = ort.InferenceSession(f"{src}/image_encoder.onnx").run(None, {"image": images[0]})[0]
print(f"ONNX vs torch: max |diff| {float(np.abs(got - want).max()):.2e}")

device = hub.Device("Samsung Galaxy S25 (Family)")
qjob = hub.submit_quantize_job(model=src, calibration_data={"image": images}, weights_dtype=hub.QuantizeDtype.INT8,
                               activations_dtype=hub.QuantizeDtype.INT16, name="ip_image_encoder_w8a16")
print("quantize job", qjob.job_id, flush=True)
cjob = hub.submit_compile_job(model=qjob.get_target_model(), device=device,
                              options="--target_runtime precompiled_qnn_onnx", name="ip_image_encoder_8elite")
print("compile job", cjob.job_id, flush=True)
target = cjob.get_target_model()
assert target is not None, f"compile job {cjob.job_id} failed: {cjob.get_status().message}"
path = target.download(f"{out}/compiled_download")
os.makedirs(f"{out}/compiled", exist_ok=True)
if str(path).endswith(".zip"):
    zipfile.ZipFile(path).extractall(f"{out}/compiled")
else:
    os.replace(path, f"{out}/compiled/{os.path.basename(path)}")
for root, _, files in os.walk(f"{out}/compiled"):
    for f in files:
        print(os.path.join(root, f), os.path.getsize(os.path.join(root, f)))

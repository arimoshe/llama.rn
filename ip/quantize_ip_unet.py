# Quantizes the IP-Adapter UNet w8a16 with Qualcomm's AIMET recipe (the same
# QuantSim settings as their SD1.5 UNet) on make_calib.py's data, and saves an
# AIMET checkpoint (OUT/unet/model.onnx + model.encodings) for export.
#   IP_ADAPTER_WEIGHTS=.../ip-adapter_sd15.safetensors \
#   python quantize_ip_unet.py MERGED CALIB.pt OUT
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from ip_unet import IPUnetQuantizable
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader

merged, calib, out = sys.argv[1:4]
data = torch.load(calib, weights_only=False)
order = ["latent", "timestep", "text_emb", "image_embeds", "ip_scale"]
entries = {k: [v.numpy() for v in data[k]] for k in order}
n = len(entries["latent"])
print(f"{n} calibration samples; ip_scale values {sorted({float(s) for s in data['ip_scale']})}")
del data

unet = IPUnetQuantizable.from_pretrained(checkpoint=merged, host_device="cpu")
unet.quantize(dataset_entries_to_dataloader(entries), num_samples=n)
unet.save_calibrated_checkpoint(output_checkpoint=out)

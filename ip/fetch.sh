#!/usr/bin/env bash
# The merged checkpoint (Liberte UNet + PixelArtRedmond, SD1.5 text encoder
# and VAE; merge.py) at /mnt/work/merged, and IP-Adapter's SD1.5 weights and
# image encoder (h94/IP-Adapter, pinned revision) at /mnt/work/ip.
set -euo pipefail
curl -sSfL -o /mnt/work/liberte.safetensors https://civitai.com/api/download/models/100409
echo "04a89af9496dba5d124ecacd2d534cae11200fd5cb03faa61e48c5b3f160ab49  /mnt/work/liberte.safetensors" | sha256sum -c -
python merge.py /mnt/work/merged /mnt/work/liberte.safetensors
rm /mnt/work/liberte.safetensors
# qai-hub-models 0.64.0 recognizes a local checkpoint folder only by these names
for d in unet text_encoder vae; do
  (cd /mnt/work/merged/$d && ln -sf config.json config.yaml && ln -sf "$(ls *.safetensors | head -1)" model.safetensor)
done
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["models/ip-adapter_sd15.safetensors", "models/image_encoder/config.json",
          "models/image_encoder/model.safetensors"]:
    hf_hub_download("h94/IP-Adapter", f, revision="018e402774aeeddd60609b4ecdb7e298259dc729", local_dir="/mnt/work/ip")
PY
ls -la /mnt/work/ip/models /mnt/work/ip/models/image_encoder

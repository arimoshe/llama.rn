#!/usr/bin/env bash
# Runner setup shared by sd15-pixel-ip-npu.yml's jobs: scratch disk on /mnt,
# 16 GB of swap (Qualcomm asks for 32 GB of RAM to quantize SD; the runner
# has 16), and the pinned Python packages the other workflow uses.
set -euo pipefail
sudo mkdir -p /mnt/work && sudo chown "$USER" /mnt/work
sudo swapoff -a || true
sudo fallocate -l 16G /mnt/swapfile
sudo chmod 600 /mnt/swapfile && sudo mkswap /mnt/swapfile && sudo swapon /mnt/swapfile
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0" "torchvision==0.26.0"
pip install "qai-hub-models==0.64.0" "aimet-onnx==2.28.0" "diffusers==0.35.2" \
  "transformers==4.56.2" "onnx>=1.17,<=1.18.0" "onnxsim<=0.4.36" "peft==0.17.1" \
  "accelerate==1.15.0" "huggingface_hub==0.36.2" "pillow==11.3.0" "safetensors==0.8.0"
python -c "from diffusers import StableDiffusionPipeline; import aimet_onnx, onnxruntime; print('imports ok')"
free -g; df -h / /mnt

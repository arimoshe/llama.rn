# Stable Diffusion 1.5 + PixelArtRedmond 1.5, w8a16 UNet for the Snapdragon 8 Elite for Galaxy

The UNet of Stable Diffusion 1.5 with the PixelArtRedmond 1.5 LoRA fused in
at scale 1.0, quantized w8a16 with Qualcomm AI Hub Models' AIMET recipe and
compiled by Qualcomm AI Hub as a precompiled QNN ONNX model (unet.onnx, an
EPContext wrapper, and unet.bin, its QNN context binary). Use it with
Qualcomm's own text encoder and VAE for the same chip
(qai-hub-models stable_diffusion_v1_5, release v0.64.0). Prompts should
include the LoRA's trigger: "pixel art, PixArFK".

Sources and licenses:

- Stable Diffusion 1.5 (stable-diffusion-v1-5/stable-diffusion-v1-5):
  CreativeML OpenRAIL-M, https://huggingface.co/spaces/CompVis/stable-diffusion-license
  — its use restrictions apply to this model.
- PixelArtRedmond 1.5 by artificialguybr
  (artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5):
  https://multimodal.art/civitai-licenses?allowNoCredit=True&allowCommercialUse=Rent&allowDerivatives=True&allowDifferentLicense=False
  — derivatives allowed under the same terms.
- Qualcomm AI Hub Models (quantization recipe): BSD-3-Clause.

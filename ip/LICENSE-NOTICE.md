# Liberte Redmond + PixelArtRedmond 1.5 + IP-Adapter, w8a16 for the Snapdragon 8 Elite for Galaxy

Two models, both quantized w8a16 and compiled by Qualcomm AI Hub as
precompiled QNN ONNX (NAME.onnx, an EPContext wrapper, and NAME.bin, its QNN
context binary), with their input and output types in metadata.json:

- unet: the UNet of Liberte Redmond 1.0 with the PixelArtRedmond 1.5 LoRA
  fused in at scale 1.0 and IP-Adapter (ip-adapter_sd15) built in. Inputs:
  latent, timestep, text_emb (as Qualcomm's SD1.5 UNet), image_embeds (the
  image encoder's output, or zeros) and ip_scale (0 = no image prompt).
- image_encoder: IP-Adapter's CLIP ViT-H/14 image encoder. Input: image
  [1,224,224,3], RGB in 0..1 (box-filtered from the drawing); output:
  image_embeds [1,1024].

Use them with Qualcomm's own SD1.5 text encoder and VAE for the same chip
(qai-hub-models stable_diffusion_v1_5, release v0.64.0). Prompts should
include the LoRA's trigger: "pixel art, PixArFK".

Sources and licenses:

- Stable Diffusion 1.5 (stable-diffusion-v1-5/stable-diffusion-v1-5):
  CreativeML OpenRAIL-M, https://huggingface.co/spaces/CompVis/stable-diffusion-license
  — its use restrictions apply to this model.
- Liberte Redmond 1.0 by artificialguybr (Civitai model version 100409):
  https://multimodal.art/civitai-licenses?allowNoCredit=True&allowCommercialUse=Rent&allowDerivatives=True&allowDifferentLicense=False
  — derivatives allowed under the same terms.
- PixelArtRedmond 1.5 by artificialguybr
  (artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5): same terms.
- IP-Adapter (h94/IP-Adapter, Tencent AI Lab): Apache-2.0.
- OpenCLIP ViT-H/14 (laion/CLIP-ViT-H-14-laion2B-s32B-b79K, as packaged in
  h94/IP-Adapter): MIT.
- Qualcomm AI Hub Models (quantization recipe): BSD-3-Clause.

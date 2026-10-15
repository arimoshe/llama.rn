# Fuses PixelArtRedmond 1.5 into Liberte Redmond's UNet (the SD1.5 fine-tune
# the LoRA was made for; fused into plain SD1.5 it draws far worse) and saves
# the result in diffusers layout, the checkpoint qai_hub_models' quantize and
# export take. The text encoder, VAE, tokenizer and scheduler are SD1.5's own:
# the app runs Qualcomm's builds of those, Liberte's VAE decodes these images the
# same, and its own text encoder (fine-tuned) changes details, not quality.
#
#   python merge.py OUT LIBERTE_CHECKPOINT [LORA_SCALE]
import sys, torch
from diffusers import StableDiffusionPipeline
out, base = sys.argv[1], sys.argv[2]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    SD15, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
liberte = StableDiffusionPipeline.from_single_file(
    base, config=SD15, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
pipe.unet = liberte.unet
del liberte
pipe.load_lora_weights("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
                       weight_name="PixelArtRedmond15V-PixelArt-PIXARFK.safetensors", adapter_name="pixel")
pipe.fuse_lora(lora_scale=scale)
pipe.unload_lora_weights()
pipe.save_pretrained(out, safe_serialization=True)
print("saved", out)

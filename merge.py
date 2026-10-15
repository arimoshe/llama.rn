# Fuses PixelArtRedmond 1.5 into Stable Diffusion 1.5 and saves the result in
# diffusers layout, the checkpoint qai_hub_models' quantize/export take.
import sys, torch
from diffusers import StableDiffusionPipeline
out = sys.argv[1]
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32, safety_checker=None,
    requires_safety_checker=False)
pipe.load_lora_weights("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
                       weight_name="PixelArtRedmond15V-PixelArt-PIXARFK.safetensors", adapter_name="pixel")
pipe.fuse_lora(lora_scale=scale)
pipe.unload_lora_weights()
pipe.save_pretrained(out, safe_serialization=True)
print("saved", out)

# Calibration data for the IP-Adapter UNet: the phone's pipeline in full
# precision (the merged checkpoint: Liberte UNet + PixelArtRedmond, SD1.5 text
# encoder and VAE; Euler 20 steps, guidance 7.5) with every UNet input
# recorded, cond and uncond, in the UNet's input order.
#
#  1. A portrait per character prompt, adapter off (ip_scale 0, zero image
#     embedding): calibrates the plain path, and is the character's reference.
#  2. Each portrait's image embedding, preprocessed as the app does.
#  3. A scene per scene prompt with one of those characters, at an ip_scale
#     drawn from 0.4-0.7 so the range the app may use is calibrated.
#
# Writes OUT/unet_calib.pt, OUT/portraits/*.png (the image encoder's
# calibration set) and OUT/scenes/*.png (for a look).
#   python make_calib.py MERGED IP_REPO_DIR CHARACTERS.txt SCENES.txt OUT STEPS
import os, random, sys, time
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection
sys.path.insert(0, os.path.dirname(__file__))
from clip_preprocess import clip_input
from image_encoder import ImageEncoder
from PIL import Image

merged, ip_repo, chars_file, scenes_file, out, steps = sys.argv[1:7]
steps = int(steps)
os.makedirs(f"{out}/portraits", exist_ok=True)
os.makedirs(f"{out}/scenes", exist_ok=True)
lines = lambda f: [l.strip() for l in open(f) if l.strip() and not l.startswith("#")]
characters, scenes = lines(chars_file), lines(scenes_file)
random.seed(0)
torch.manual_seed(0)

pipe = StableDiffusionPipeline.from_pretrained(merged, torch_dtype=torch.float32, safety_checker=None)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter(ip_repo, subfolder="models", weight_name="ip-adapter_sd15.safetensors", image_encoder_folder=None)
encoder = CLIPVisionModelWithProjection.from_pretrained(f"{ip_repo}/models/image_encoder").eval()
torch.set_num_threads(os.cpu_count())

data = {k: [] for k in ["latent", "timestep", "text_emb", "image_embeds", "ip_scale"]}
current = {"scale": 0.0}


def record(module, args, kwargs):
    sample, t = args[0], args[1]
    text = kwargs.get("encoder_hidden_states", args[2] if len(args) > 2 else None)
    image = kwargs["added_cond_kwargs"]["image_embeds"][0]  # [batch, 1, 1024]
    for i in range(sample.shape[0]):
        data["latent"].append(sample[i:i + 1].detach().clone())
        data["timestep"].append(torch.tensor([[float(t)]]))
        data["text_emb"].append(text[i:i + 1].detach().clone())
        data["image_embeds"].append(image[i].reshape(1, -1).detach().clone())
        data["ip_scale"].append(torch.tensor([[current["scale"]]]))


pipe.unet.register_forward_pre_hook(record, with_kwargs=True)


def draw(prompt, seed, embeds, scale):
    current["scale"] = scale
    pipe.set_ip_adapter_scale(scale)
    # [uncond, cond], as the pipeline batches them for guidance
    both = torch.cat([torch.zeros_like(embeds), embeds])[:, None]
    g = torch.Generator().manual_seed(seed)
    return pipe(prompt, negative_prompt="", num_inference_steps=steps, guidance_scale=7.5, generator=g,
                ip_adapter_image_embeds=[both]).images[0]


t0 = time.time()
for i, prompt in enumerate(characters):
    img = draw(prompt, 1000 + i, torch.zeros(1, 1024), 0.0)
    img.save(f"{out}/portraits/{i:03d}.png")
    print(f"portrait {i + 1}/{len(characters)} ({time.time() - t0:.0f}s)", flush=True)

enc = ImageEncoder(encoder).eval()
refs = []
for i in range(len(characters)):
    with torch.no_grad():
        refs.append(enc(torch.from_numpy(clip_input(Image.open(f"{out}/portraits/{i:03d}.png")))))
del encoder, enc

for i, prompt in enumerate(scenes):
    scale = random.choice([0.4, 0.5, 0.6, 0.7])
    draw(prompt, 2000 + i, refs[i % len(refs)], scale).save(f"{out}/scenes/{i:03d}.png")
    print(f"scene {i + 1}/{len(scenes)} at {scale} ({time.time() - t0:.0f}s)", flush=True)

torch.save(data, f"{out}/unet_calib.pt")
print(f"{len(data['latent'])} samples -> {out}/unet_calib.pt")

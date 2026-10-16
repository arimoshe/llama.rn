# Checks adapt_ip_unet against diffusers in full precision:
#  1. ip_scale 0.5 with a character's embedding: the NPU-form UNet matches
#     diffusers' UNet with IP-Adapter (its own processors, scale 0.5)
#  2. ip_scale 0: it matches Qualcomm's rewrite of the plain UNet
#   python check_ip_unet.py MERGED_DIR IP_WEIGHTS IMAGE_ENCODER_DIR PORTRAIT.png
import copy, sys, torch
from PIL import Image
from diffusers import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection
sys.path.insert(0, __import__("os").path.dirname(__file__))
from clip_preprocess import clip_input
from image_encoder import ImageEncoder
from ip_unet import adapt_ip_unet, attach_ip_adapter
from qai_hub_models.models.templates.diffusion.model_adaptation import get_timestep_embedding, monkey_patch_model

merged, weights, encoder_dir, portrait = sys.argv[1:5]
torch.manual_seed(0)
enc = ImageEncoder(CLIPVisionModelWithProjection.from_pretrained(encoder_dir)).eval()
with torch.no_grad():
    emb = enc(torch.from_numpy(clip_input(Image.open(portrait))))
del enc
latent = torch.randn(1, 4, 64, 64)
t = torch.tensor([[500.0]])
text = torch.randn(1, 77, 768)

base = UNet2DConditionModel.from_pretrained(f"{merged}/unet").eval()
ref = attach_ip_adapter(copy.deepcopy(base), weights).eval()
for p in ref.attn_processors.values():
    if hasattr(p, "scale"):
        p.scale = [0.5]
with torch.no_grad():
    want_ip = ref(latent, t.flatten(), text, added_cond_kwargs={"image_embeds": [emb[:, None]]}).sample
del ref
ours = adapt_ip_unet(attach_ip_adapter(copy.deepcopy(base), weights).eval()).eval()
plain = copy.deepcopy(base)
plain.get_time_embed = get_timestep_embedding
monkey_patch_model(plain)
with torch.no_grad():
    got_ip = ours(latent, t, text, emb, torch.tensor([[0.5]]))
    got_0 = ours(latent, t, text, emb, torch.tensor([[0.0]]))
    want_0 = plain(latent, t, text, return_dict=False)[0]
rel = lambda a, b: float((a - b).norm() / b.norm())
print(f"scale 0.5 vs diffusers IP-Adapter: rel diff {rel(got_ip, want_ip):.2e}")
print(f"scale 0   vs plain UNet (Qualcomm form): rel diff {rel(got_0, want_0):.2e}")
print(f"IP effect at 0.5 (vs plain): rel {rel(got_ip, want_0):.2e}")
ok = rel(got_ip, want_ip) < 1e-3 and rel(got_0, want_0) < 1e-5
print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)

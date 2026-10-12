# Export AuroraIMG-6M to three ONNX graphs and check them against PyTorch.
# usage: python export.py <directory holding the model's Hugging Face files>
import json, re, sys
import numpy as np, torch, onnxruntime as ort
from PIL import Image
from safetensors.torch import load_file
from tokenizers import Tokenizer
sys.path.insert(0, '.')
from aurora_model import TextEncoder, Denoiser, VAE

A = sys.argv[1]
torch.manual_seed(0)
te, den, vae = TextEncoder(), Denoiser(), VAE()
for m, p in ((te, 'text_encoder'), (den, 'unet'), (vae, 'vae')):
    missing, unexpected = m.load_state_dict(load_file(f'{A}/{p}/diffusion_pytorch_model.safetensors'), strict=False)
    print(p, 'missing', missing, 'unexpected', unexpected)
    m.eval()
vcfg = json.load(open(f'{A}/vae/config.json'))
LAT_MU = torch.tensor(vcfg['lat_mu']).view(1, 8, 1, 1)
LAT_SD = torch.tensor(vcfg['lat_sd']).view(1, 8, 1, 1)
tok = Tokenizer.from_file(f'{A}/tokenizer/tokenizer.json')

def encode(text):
    s = re.sub(r"https?://\S+|www\.\S+", " ", re.sub(r"<[^>]+>", " ", text))
    p = re.sub(r"\s+", " ", s.lower()).strip()
    ids = tok.encode(p, add_special_tokens=False).ids[:62]
    seq = [1] + ids + [2]
    return seq + [0] * (64 - len(seq))

class TE(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, tok, mask): return s.m(tok.long(), mask != 0)
class UN(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x, t, ctx, mask): return s.m(x, t, ctx, mask != 0)
class DEC(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, z): return (s.m.decode(z * LAT_SD + LAT_MU) / 2 + 0.5).clamp(0, 1)

def sample(prompt, noise, steps=50, cfg=6.0, rescale=1.0, run=None):
    tk = torch.tensor([encode(prompt), encode('')], dtype=torch.int32); mk = (tk != 0).int()
    ctx = run['te'](tk, mk)
    lat = torch.cat([noise, noise])
    ts = torch.linspace(1, 0, steps + 1)
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        v = run['un'](lat, (t * 1000.0).expand(2), ctx, mk)
        vc, vu = v.chunk(2)
        g = vu + cfg * (vc - vu)
        nc = vc.flatten(1).norm(dim=1).view(-1, 1, 1, 1); ng = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        g = rescale * g * (nc / ng.clamp(min=1e-8)) + (1 - rescale) * g
        lat = lat + (tn - t) * torch.cat([g, g])
    return run['dec'](lat[:1])

tew, unw, decw = TE(te), UN(den), DEC(vae)
with torch.no_grad():
    torch_run = {'te': lambda a, b: tew(a, b), 'un': lambda *a: unw(*a), 'dec': lambda z: decw(z)}
    noise = torch.from_numpy(np.random.default_rng(42).standard_normal((1, 8, 16, 16)).astype(np.float32))
    img_t = sample('a red car on a street', noise, run=torch_run)
    Image.fromarray((img_t[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).resize((256, 256), Image.NEAREST).save('torch_red_car.png')

    B = 2
    ex_tok = torch.tensor([encode('a red car'), encode('')], dtype=torch.int32); ex_mask = (ex_tok != 0).int()
    torch.onnx.export(tew, (ex_tok, ex_mask), 'text_encoder.onnx', input_names=['tokens', 'mask'], output_names=['context'],
                      dynamic_axes={'tokens': {0: 'b'}, 'mask': {0: 'b'}, 'context': {0: 'b'}}, opset_version=17, dynamo=False)
    ex_ctx = tew(ex_tok, ex_mask)
    torch.onnx.export(unw, (torch.randn(B, 8, 16, 16), torch.tensor([500.0, 500.0]), ex_ctx, ex_mask), 'unet.onnx',
                      input_names=['latents', 't', 'context', 'mask'], output_names=['velocity'],
                      dynamic_axes={'latents': {0: 'b'}, 't': {0: 'b'}, 'context': {0: 'b'}, 'mask': {0: 'b'}, 'velocity': {0: 'b'}},
                      opset_version=17, dynamo=False)
    torch.onnx.export(decw, (torch.randn(1, 8, 16, 16),), 'vae_decoder.onnx', input_names=['latents'], output_names=['image'],
                      dynamic_axes={'latents': {0: 'b'}, 'image': {0: 'b'}}, opset_version=17, dynamo=False)

opts = ort.SessionOptions()
s_te, s_un, s_dec = (ort.InferenceSession(f, opts, providers=['CPUExecutionProvider']) for f in ('text_encoder.onnx', 'unet.onnx', 'vae_decoder.onnx'))
T = lambda x: torch.from_numpy(x)
ort_run = {
    'te': lambda a, b: T(s_te.run(None, {'tokens': a.numpy(), 'mask': b.numpy()})[0]),
    'un': lambda x, t, c, m: T(s_un.run(None, {'latents': x.numpy(), 't': t.numpy().astype(np.float32), 'context': c.numpy(), 'mask': m.numpy()})[0]),
    'dec': lambda z: T(s_dec.run(None, {'latents': z.numpy()})[0]),
}
with torch.no_grad():
    img_o = sample('a red car on a street', noise, run=ort_run)
print('max |torch - onnx| over the whole 50-step pipeline:', float((img_t - img_o).abs().max()))
Image.fromarray((img_o[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).resize((256, 256), Image.NEAREST).save('onnx_red_car.png')
for p in ['a castle on a hill at sunset', 'a dragon flying over a forest', 'a knight in armor']:
    with torch.no_grad(): im = sample(p, noise, run=ort_run)
    Image.fromarray((im[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).resize((192, 192), Image.NEAREST).save(p.replace(' ', '_') + '.png')
print('tokens for "a red car on a street":', encode('a red car on a street')[:10])

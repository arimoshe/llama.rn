# The model classes from AuroraIMG-6M's pipeline.py (AuroraAI-Research, Apache-2.0),
# with the diffusers/transformers wrappers removed so they load bare weights.
import torch
import torch.nn as nn
import torch.nn.functional as F

def gn(c): return nn.GroupNorm(8, c)

class VRes(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.n1, s.c1, s.n2, s.c2 = gn(c), nn.Conv2d(c, c, 3, padding=1), gn(c), nn.Conv2d(c, c, 3, padding=1)
    def forward(s, x):
        h = s.c1(F.silu(s.n1(x))); h = s.c2(F.silu(s.n2(h))); return x + h

class VAE(nn.Module):
    def __init__(self, lat_mu=None, lat_sd=None):
        super().__init__()
        s = self
        s.enc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), VRes(32), nn.Conv2d(32, 64, 3, 2, 1), VRes(64),
                              nn.Conv2d(64, 128, 3, 2, 1), VRes(128), gn(128), nn.SiLU(), nn.Conv2d(128, 16, 3, padding=1))
        s.dec = nn.Sequential(nn.Conv2d(8, 128, 3, padding=1), VRes(128), nn.Upsample(scale_factor=2, mode="nearest"),
                              nn.Conv2d(128, 64, 3, padding=1), VRes(64), VRes(64), nn.Upsample(scale_factor=2, mode="nearest"),
                              nn.Conv2d(64, 32, 3, padding=1), VRes(32), VRes(32), gn(32), nn.SiLU(), nn.Conv2d(32, 3, 3, padding=1), nn.Tanh())
    def encode(s, x):
        m, lv = s.enc(x).chunk(2, 1); return m, lv.clamp(-30, 20)
    def decode(s, z): return s.dec(z)
    def forward(s, x):
        m, lv = s.encode(x)
        z = m + torch.exp(0.5 * lv) * torch.randn_like(m)
        return s.dec(z), m, lv

class RoPE(nn.Module):
    def __init__(self, hd, max_len=64, theta=1000.0):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.outer(torch.arange(max_len).float(), inv)
        ang = torch.cat([ang, ang], -1)
        self.register_buffer("cos", ang.cos()[None, None], persistent=False)
        self.register_buffer("sin", ang.sin()[None, None], persistent=False)
    @staticmethod
    def rot(x):
        x1, x2 = x.chunk(2, -1); return torch.cat((-x2, x1), -1)
    def forward(self, q, k):
        n = q.shape[2]
        cos, sin = self.cos[:, :, :n].to(q.dtype), self.sin[:, :, :n].to(q.dtype)
        return q * cos + self.rot(q) * sin, k * cos + self.rot(k) * sin

class MHA(nn.Module):
    def __init__(s, dim, heads=4, ctx_dim=None, use_rope=False):
        super().__init__()
        s.h = heads; cd = ctx_dim or dim
        s.q, s.k, s.v, s.o = nn.Linear(dim, dim), nn.Linear(cd, dim), nn.Linear(cd, dim), nn.Linear(dim, dim)
        s.use_rope = use_rope
        if use_rope: s.rope = RoPE(dim // heads, max_len=64, theta=1000.0)
    def forward(s, x, ctx=None, mask=None):
        self_attn = ctx is None
        ctx = x if self_attn else ctx
        B, N, Cc = x.shape; Lc = ctx.shape[1]; h = s.h
        q = s.q(x).view(B, N, h, -1).transpose(1, 2)
        k = s.k(ctx).view(B, Lc, h, -1).transpose(1, 2)
        v = s.v(ctx).view(B, Lc, h, -1).transpose(1, 2)
        if s.use_rope and self_attn: q, k = s.rope(q, k)
        if mask is not None:
            if mask.dtype == torch.bool:
                attn_mask = torch.zeros_like(mask, dtype=q.dtype)
                attn_mask.masked_fill_(~mask, float('-inf'))
            else:
                attn_mask = mask
        else:
            attn_mask = None
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return s.o(o.transpose(1, 2).reshape(B, N, Cc))

class TBlock(nn.Module):
    def __init__(s, d=128, ff=512):
        super().__init__()
        s.ln1, s.attn, s.ln2 = nn.LayerNorm(d), MHA(d, 4, use_rope=True), nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
    def forward(s, x, m):
        x = x + s.attn(s.ln1(x), None, m); return x + s.ff(s.ln2(x))

class TextEncoder(nn.Module):
    def __init__(self, vocab=4096, d=128, layers=4):
        super().__init__()
        s = self
        s.tok_emb = nn.Embedding(vocab, d)
        nn.init.normal_(s.tok_emb.weight, std=0.02)
        s.blocks = nn.ModuleList([TBlock(d) for _ in range(layers)])
        s.ln_f = nn.LayerNorm(d)
    def forward(s, tok, mask):
        x = s.tok_emb(tok)
        am = mask[:, None, None, :]
        for b in s.blocks: x = b(x, am)
        return s.ln_f(x)

class DRes(nn.Module):
    def __init__(s, cin, cout, tdim=128):
        super().__init__()
        s.n1, s.c1, s.t_proj = gn(cin), nn.Conv2d(cin, cout, 3, padding=1), nn.Linear(tdim, 3 * cout)
        s.n2, s.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        s.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(s.t_proj.weight); nn.init.zeros_(s.t_proj.bias)
    def forward(s, x, temb):
        h = s.c1(F.silu(s.n1(x)))
        sc, sh, gate = s.t_proj(temb).chunk(3, 1)
        h = s.n2(h) * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
        return gate[:, :, None, None] * s.c2(F.silu(h)) + s.skip(x)

class AttnBlock(nn.Module):
    def __init__(s, c, n_tok, ctx_dim=128, ff_mult=2):
        super().__init__()
        s.norm = gn(c); s.sa = MHA(c, 4); s.ln2 = nn.LayerNorm(c); s.ca = MHA(c, 4, ctx_dim)
        s.pos_emb = nn.Parameter(torch.randn(1, n_tok, c) * 0.02)
        s.ln3 = nn.LayerNorm(c); s.ff = nn.Sequential(nn.Linear(c, ff_mult * c), nn.GELU(), nn.Linear(ff_mult * c, c))
    def forward(s, x, ctx, cmask):
        B, Cc, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        n = s.norm(x).flatten(2).transpose(1, 2)
        t = t + s.sa(n + s.pos_emb.to(n.dtype))
        t = t + s.ca(s.ln2(t), ctx, cmask[:, None, None, :])
        t = t + s.ff(s.ln3(t))
        return t.transpose(1, 2).reshape(B, Cc, H, W)

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        s = self
        s.register_buffer("t_freqs", torch.randn(32) * 10.0, persistent=True)
        s.time_mlp = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        s.pool_proj = nn.Linear(128, 128)
        s.conv_in = nn.Conv2d(8, 64, 3, padding=1)
        s.d0, s.down0 = DRes(64, 64), nn.Conv2d(64, 64, 3, 2, 1)
        s.h0_attn = AttnBlock(64, 256, ff_mult=2)
        s.d1, s.a1, s.down1 = DRes(64, 128), AttnBlock(128, 64, ff_mult=3), nn.Conv2d(128, 128, 3, 2, 1)
        s.d2, s.a2 = DRes(128, 128), AttnBlock(128, 16, ff_mult=4)
        s.m1, s.ma, s.m2 = DRes(128, 128), AttnBlock(128, 16, ff_mult=4), DRes(128, 128)
        s.u2, s.ua2, s.up2 = DRes(256, 128), AttnBlock(128, 16, ff_mult=4), nn.Conv2d(128, 128, 3, padding=1)
        s.u1, s.ua1, s.up1 = DRes(256, 128), AttnBlock(128, 64, ff_mult=3), nn.Conv2d(128, 64, 3, padding=1)
        s.u0 = DRes(128, 64)
        s.head = nn.Sequential(gn(64), nn.SiLU(), nn.Conv2d(64, 8, 3, padding=1))
        nn.init.zeros_(s.head[2].weight); nn.init.zeros_(s.head[2].bias)
    def forward(s, x, t, ctx, cmask):
        half = (t / 1000.0).float()[:, None] * s.t_freqs[None]
        temb = s.time_mlp(torch.cat([half.sin(), half.cos()], -1))
        m = cmask.to(ctx.dtype)[:, :, None]
        pooled = (ctx * m).sum(1) / m.sum(1).clamp(min=1)
        temb = temb + s.pool_proj(pooled).to(temb.dtype)
        x = s.conv_in(x)
        h0 = s.h0_attn(s.d0(x, temb), ctx, cmask)
        x = s.down0(h0)
        h1 = s.a1(s.d1(x, temb), ctx, cmask); x = s.down1(h1)
        h2 = s.a2(s.d2(x, temb), ctx, cmask); x = h2
        x = s.m2(s.ma(s.m1(x, temb), ctx, cmask), temb)
        x = s.ua2(s.u2(torch.cat([x, h2], 1), temb), ctx, cmask)
        x = s.up2(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = s.ua1(s.u1(torch.cat([x, h1], 1), temb), ctx, cmask)
        x = s.up1(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = s.u0(torch.cat([x, h0], 1), temb)
        return s.head(x)


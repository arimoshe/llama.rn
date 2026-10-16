# The pixel-art UNet with IP-Adapter (h94/IP-Adapter, ip-adapter_sd15) for the
# NPU: SD1.5's UNet that also attends to a character's image, so a scene can
# keep the look of a character drawn earlier.
#
# Qualcomm's UNet recipe (qai_hub_models' UnetQuantizable) rewrites every
# attention module into split-head form (SHAAttention) for the NPU, which
# would drop IP-Adapter's image branch. Here each cross-attention module gets
# SHAIPAttention instead: the same split-head text attention plus, per head,
# attention over the image tokens, added as text + ip_scale * image before
# the output projection - exactly IP-Adapter's decoupled cross-attention.
#
# Inputs (Qualcomm's three, then two):
#   latent [1,4,64,64], timestep [1,1], text_emb [1,77,768]
#   image_embeds [1,1024]  CLIP ViT-H image embedding of the character
#                          (zeros: no character, and the unconditional pass)
#   ip_scale [1,1]         0 is the plain pixel-art UNet; ~0.5 keeps a
#                          character while following the prompt
# The image embedding is projected to 4 tokens inside the graph and appended
# to the text tokens, so one tensor flows through Qualcomm's rewritten blocks
# and each SHAIPAttention splits it again.
from __future__ import annotations

import os
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

from qai_hub_models.models.stable_diffusion_v1_5 import model as sd
from qai_hub_models.models.templates.diffusion.model_adaptation import (
    SHAAttention, get_timestep_embedding, monkey_patch_model,
)
from qai_hub_models.utils.input_spec import TensorSpec

NUM_IP_TOKENS = 4
IMAGE_EMBED_DIM = 1024


class _Shared:
    """ip_scale for the current forward, read by every SHAIPAttention."""
    scale: torch.Tensor | None = None


class SHAIPAttention(nn.Module):
    def __init__(self, sha: SHAAttention, to_k_ip: nn.Linear, to_v_ip: nn.Linear, shared: _Shared) -> None:
        super().__init__()
        self.sha = sha
        self.shared = shared

        def per_head(lin: nn.Linear) -> nn.ModuleList:
            convs = nn.ModuleList(
                nn.Conv2d(lin.in_features, sha.dim_head, kernel_size=1, bias=lin.bias is not None)
                for _ in range(sha.kv_heads)
            )
            for i, conv in enumerate(convs):
                rows = slice(i * sha.dim_head, (i + 1) * sha.dim_head)
                conv.weight.data.copy_(lin.weight.data[rows].clone()[:, :, None, None])
                if lin.bias is not None:
                    conv.bias.data.copy_(lin.bias.data[rows].clone())
            return convs

        self.k_ip = per_head(to_k_ip)
        self.v_ip = per_head(to_v_ip)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                encoder_hidden_states: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        sha = self.sha
        assert encoder_hidden_states is not None and attention_mask is None
        bsz = hidden_states.shape[0]
        # (N, seq, dim) -> (N, dim, 1, seq), as SHAAttention does
        text = encoder_hidden_states[:, :-NUM_IP_TOKENS].permute(0, 2, 1).unsqueeze(2)
        image = encoder_hidden_states[:, -NUM_IP_TOKENS:].permute(0, 2, 1).unsqueeze(2)
        scale = self.shared.scale

        def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            q_flat = q.permute(0, 2, 3, 1)
            k_flat = k.view(bsz, 1, sha.dim_head, -1)
            v_flat = v.view(bsz, 1, sha.dim_head, -1)
            probs = torch.softmax(torch.matmul(q_flat, k_flat) * sha.scale, dim=-1)
            return torch.matmul(probs, v_flat.permute(0, 1, 3, 2))

        outs = []
        for h in range(sha.heads):
            q = sha.q_proj_sha[h](hidden_states)
            kv = h * sha.kv_heads // sha.heads
            o = attend(q, sha.k_proj_sha[kv](text), sha.v_proj_sha[kv](text))
            o = o + scale * attend(q, self.k_ip[kv](image), self.v_ip[kv](image))
            outs.append(o)
        out = torch.cat(outs, dim=-1)
        out = sha.to_out[0](out)
        out = sha.to_out[1](out)
        return out.permute(0, 3, 1, 2)


def attach_ip_adapter(unet: nn.Module, weights: str) -> nn.Module:
    """Loads IP-Adapter's processors and image projection into a diffusers UNet."""
    state = load_file(weights)
    split = {"image_proj": {}, "ip_adapter": {}}
    for k, v in state.items():
        head, rest = k.split(".", 1)
        split[head][rest] = v
    unet._load_ip_adapter_weights([split])
    return unet


class IPUNetWrapper(nn.Module):
    def __init__(self, unet: nn.Module, image_proj: nn.Module, shared: _Shared) -> None:
        super().__init__()
        self.model = unet
        self.image_proj = image_proj
        self.shared = shared

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor, text_emb: torch.Tensor,
                image_embeds: torch.Tensor, ip_scale: torch.Tensor) -> torch.Tensor:
        self.shared.scale = ip_scale.reshape(1, 1, 1, 1)
        tokens = torch.cat([text_emb, self.image_proj(image_embeds)], dim=1)
        return self.model(latent, timestep, tokens, return_dict=False)[0]

    def get_input_spec(self, batch_size: int = 1) -> dict:
        spec = sd.UnetBase.get_input_spec(sd.UnetQuantizable.__new__(sd.UnetQuantizable), batch_size)
        spec["image_embeds"] = TensorSpec(shape=(batch_size, IMAGE_EMBED_DIM), dtype="float32")
        spec["ip_scale"] = TensorSpec(shape=(batch_size, 1), dtype="float32")
        return spec

    def get_output_spec(self) -> dict:
        return {"output_latent": TensorSpec(apply_runtime_channel_reordering=True)}


def adapt_ip_unet(unet: nn.Module) -> IPUNetWrapper:
    """Qualcomm's on-device rewrite, with IP-Adapter's branch kept (see top)."""
    procs = {name.rsplit(".processor", 1)[0]: p for name, p in unet.attn_processors.items()
             if hasattr(p, "to_k_ip")}
    image_proj = unet.encoder_hid_proj.image_projection_layers[0]
    # the wrapper projects the image itself
    unet.encoder_hid_proj = None
    unet.config.encoder_hid_dim_type = None
    unet.get_time_embed = get_timestep_embedding
    monkey_patch_model(unet)
    shared = _Shared()
    for name, proc in procs.items():
        parent_name, child = name.rsplit(".", 1)
        parent = unet.get_submodule(parent_name)
        sha = getattr(parent, child)
        assert isinstance(sha, SHAAttention), f"{name}: {type(sha).__name__}"
        setattr(parent, child, SHAIPAttention(sha, proc.to_k_ip[0], proc.to_v_ip[0], shared))
    print(f"IP-Adapter: {len(procs)} cross-attention modules with an image branch")
    return IPUNetWrapper(unet, image_proj, shared)


class IPUnetQuantizable(sd.UnetQuantizable):
    """Qualcomm's quantizable UNet, built from adapt_ip_unet. The IP-Adapter
    weights come from $IP_ADAPTER_WEIGHTS."""

    @classmethod
    def torch_from_pretrained(cls, checkpoint: Any = "DEFAULT", subfolder: str = "",
                              host_device: torch.device | str = torch.device("cpu"),
                              adapt_torch_model_options: dict | None = None) -> nn.Module:
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(os.path.join(str(checkpoint), subfolder or "unet"))
        attach_ip_adapter(unet, os.environ["IP_ADAPTER_WEIGHTS"])
        return adapt_ip_unet(unet.eval()).to(torch.device(host_device)).eval()

    def get_input_spec(self, batch_size: int = 1) -> dict:  # type: ignore[override]
        return IPUNetWrapper.get_input_spec(self, batch_size)

    def forward(self, latent, time_emb, text_emb, image_embeds, ip_scale):  # type: ignore[override]
        from qai_hub_models.utils.quantization_aimet_onnx import AIMETOnnxQuantizableMixin
        return AIMETOnnxQuantizableMixin.forward(self, latent, time_emb, text_emb, image_embeds, ip_scale)

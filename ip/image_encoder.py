# IP-Adapter's image encoder (CLIP ViT-H/14 with projection, from
# h94/IP-Adapter models/image_encoder) as the phone runs it: image
# [1,224,224,3] in 0..1, channels last (clip_preprocess.py), normalized with
# CLIP's mean and std inside the graph; out image_embeds [1,1024].
import torch
from torch import nn

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class ImageEncoder(nn.Module):
    def __init__(self, clip_vision_with_projection: nn.Module) -> None:
        super().__init__()
        self.clip = clip_vision_with_projection
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image.permute(0, 3, 1, 2) - self.mean) / self.std
        return self.clip(pixel_values=x).image_embeds

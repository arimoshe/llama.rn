# export_unet.py for the IP-Adapter UNet: Qualcomm's export CLI with the
# pipeline's UNet class swapped for IPUnetQuantizable (two more inputs), the
# stock text encoder and VAE as before.
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from ip_unet import IPUnetQuantizable
from qai_hub_models.models.stable_diffusion_v1_5 import model as sd
from qai_hub_models_cli.cli import main


@classmethod
def from_pretrained(cls, checkpoint: str = "DEFAULT"):
    return cls(
        sd.TextEncoderQuantizable.from_pretrained(checkpoint="DEFAULT"),
        IPUnetQuantizable.from_pretrained(checkpoint=checkpoint),
        sd.VaeDecoderQuantizable.from_pretrained(checkpoint="DEFAULT"),
    )


sd.StableDiffusionV1_5_Quantized.from_pretrained = from_pretrained
sd.StableDiffusionV1_5_Quantized.component_classes = {
    **sd.StableDiffusionV1_5_Quantized.component_classes, "unet": IPUnetQuantizable}
main(["export", "stable_diffusion_v1_5", *sys.argv[1:]])

# `qai-hub-models export stable_diffusion_v1_5 --components unet` builds the
# whole pipeline from --checkpoint before exporting the one component, and
# its text encoder and VAE only load from an AIMET export or Qualcomm's own.
# The LoRA leaves both stock, so they come from Qualcomm's (DEFAULT, the ones
# the app runs) and only the UNet from the checkpoint; then the CLI's own
# export runs on the arguments given.
import sys

from qai_hub_models.models.stable_diffusion_v1_5 import model as sd
from qai_hub_models_cli.cli import main


@classmethod
def from_pretrained(cls, checkpoint: str = "DEFAULT"):
    return cls(
        sd.TextEncoderQuantizable.from_pretrained(checkpoint="DEFAULT"),
        sd.UnetQuantizable.from_pretrained(checkpoint=checkpoint),
        sd.VaeDecoderQuantizable.from_pretrained(checkpoint="DEFAULT"),
    )


sd.StableDiffusionV1_5_Quantized.from_pretrained = from_pretrained
main(["export", "stable_diffusion_v1_5", *sys.argv[1:]])

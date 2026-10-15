# Runs qai_hub_models' SD1.5 quantize for a local checkpoint. qai-hub-models
# 0.64.0 exports a custom checkpoint by asking the bare torch module that
# torch_from_pretrained returns for get_input_spec, which only the wrapper
# class has; the wrapper's specs are lent to the module here.
import sys
from types import SimpleNamespace
import qai_hub_models.models.stable_diffusion_v1_5.model as m
from qai_hub_models.models.templates.diffusion import stable_diffusion as sd

orig = m.UnetQuantizable.torch_from_pretrained.__func__

def torch_from_pretrained(cls, *args, **kwargs):
    module = orig(cls, *args, **kwargs)
    spec = SimpleNamespace(seq_len=m.SEQ_LEN, text_emb_dim=768)
    module.get_input_spec = lambda *a, **k: sd.UnetBase.get_input_spec(spec, *a, **k)
    module.get_output_spec = lambda *a, **k: sd.UnetBase.get_output_spec(spec, *a, **k)
    return module

m.UnetQuantizable.torch_from_pretrained = classmethod(torch_from_pretrained)

from qai_hub_models.models.templates.diffusion.quantize import stable_diffusion_quantize
from qai_hub_models.models.stable_diffusion_v1_5 import MODEL_ID, Model
sys.argv = [sys.argv[0], "--component", "unet"] + sys.argv[1:]
stable_diffusion_quantize(model_cls=Model, model_id=MODEL_ID,
                          default_num_steps=Model.default_num_steps)

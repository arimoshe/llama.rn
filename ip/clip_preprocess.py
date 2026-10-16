# The image encoder's input, made the way the app makes it: the 512 px
# drawing box-filtered (area average) to 224x224, RGB in 0..1, channels last.
# CLIP's own processor resizes bicubic; the area average is what the phone
# can do cheaply, so calibration and the check use it too.
import numpy as np
from PIL import Image

SIZE = 224


def clip_input(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((SIZE, SIZE), Image.BOX)
    return (np.asarray(img, dtype=np.float32) / 255.0)[None]  # [1,224,224,3]

# py_algo
Some algorithms I share


## Color correction
- modified from [sd-webui-stablesr colorfix](https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py)
- supports cpu, cuda(fp16/fp32)


```python
import numpy as np
import torch
from torch import Tensor
from utils.images_io import load_image_fp32, write_image
from utils.torch_tensor import img_to_tensor, tensor_to_img

from color_correction import ColorCorrection

device = "cuda"
fp16: bool = True

image_fp: str = ...
ref_fp: str = ...
out_fp: str = ...

in_img: np.ndarray = load_image_fp32(filepath=image_fp)
ref_img: np.ndarray = load_image_fp32(filepath=ref_fp)

tensor_dtype: torch.dtype = torch.float16 if fp16 else torch.float32
in_tensor: Tensor = img_to_tensor(
    d_img=torch.from_numpy(in_img).to(device),
    tensor_dtype=tensor_dtype,
    flip_r_b=True
)
ref_tensor: Tensor = img_to_tensor(
    d_img=torch.from_numpy(ref_img).to(device),
    tensor_dtype=tensor_dtype,
    flip_r_b=True
)

color_correction = ColorCorrection()
color_correction = color_correction.to(device=device)
with torch.inference_mode():
    d_out = color_correction(in_tensor, ref_tensor)
    d_out = torch.clamp(d_out, 0., 1.)

d_out: Tensor = tensor_to_img(
    tensor=d_out, img_dtype=torch.float32, flip_r_b=True
)
out_img = d_out.cpu().numpy()
write_image(filepath=out_fp, img=out_img)
```

- Functions not provided: `load_image_fp32`, `write_image`, `img_to_tensor`, `tensor_to_img`

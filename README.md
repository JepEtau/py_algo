# py_algo
Some algorithms I share


## Color correction
- modified from [sd-webui-stablesr colorfix](https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py)
- supports cpu, cuda(fp16/fp32)


```python
import numpy as np
import torch
from torch import Tensor
from color_correction import ColorCorrection

in_tensor: Tensor
ref_tensor: Tensor

color_correction = ColorCorrection()
color_correction = color_correction.to(device=device)
with torch.inference_mode():
    d_out: Tensor = color_correction(in_tensor, ref_tensor)
    d_out = torch.clamp(d_out, 0., 1.)

```

## Color transfer



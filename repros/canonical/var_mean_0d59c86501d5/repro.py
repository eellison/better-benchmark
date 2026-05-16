"""
Standalone repro captured via capture_hook.
Label: timm_convnext_tiny
Pattern hash: 0d59c86501d5
Shape hash: 7a8af858
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_21: "f32[8, 768, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:205 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default: "f32[8, 7, 7, 768]" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default, [3], correction = 0, keepdim = True);  permute_default = None
        getitem: "f32[8, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn(301056, dtype=torch.float32, device='cuda').as_strided([8, 768, 7, 7], [37632, 1, 5376, 768]),  # convolution_21
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_inference
Pattern hash: 129c6a6b3273
Shape hash: 2b1bedb9
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_38: "f32[32, 320, 18, 18]", add_71: "f32[32, 320, 18, 18]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[32, 320, 18, 18]" = torch.ops.aten.add.Tensor(convolution_38, add_71);  convolution_38 = add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default: "f32[32, 18, 18, 320]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 3, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default, [3], correction = 0, keepdim = True);  permute_default = None
        getitem: "f32[32, 18, 18, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 18, 18, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn(3317760, dtype=torch.float32, device='cuda').as_strided([32, 320, 18, 18], [103680, 1, 5760, 320]),  # convolution_38
    torch.randn(3317760, dtype=torch.float32, device='cuda').as_strided([32, 320, 18, 18], [103680, 1, 5760, 320]),  # add_71
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

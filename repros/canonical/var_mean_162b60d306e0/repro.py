"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: 162b60d306e0
Shape hash: 0b6ea462
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
    def forward(self, convolution: "f32[1, 96, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:321 in forward, code: return torch.permute(x, self.dims)
        permute_default: "f32[1, 56, 56, 96]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:608 in forward, code: x = self.features(x)
        clone_default: "f32[1, 56, 56, 96]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(clone_default, [3], correction = 0, keepdim = True);  clone_default = None
        getitem: "f32[1, 56, 56, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 56, 56, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([1, 96, 56, 56], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_infer
Pattern hash: 99db949c0182
Shape hash: 8d18dc3c
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 960, 1, 1], f32), T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)))"

class Repro(torch.nn.Module):
    def forward(self, convolution_90: "f32[512, 960, 1, 1]", cat_30: "f32[512, 960, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        add_tensor: "f32[512, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_90, 3);  convolution_90 = None
        clamp_min_default: "f32[512, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(cat_30, div_tensor);  cat_30 = div_tensor = None
        return mul_tensor



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

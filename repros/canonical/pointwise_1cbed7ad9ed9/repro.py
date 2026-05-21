"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_train
Pattern hash: 1cbed7ad9ed9
Shape hash: ec5ba4c9
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
_shapes_config = "(T([256, 960, 1, 1], f32), T([256, 960, 7, 7], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_59: "f32[256, 960, 1, 1]", div_25: "f32[256, 960, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        add_tensor: "f32[256, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_59, 3);  convolution_59 = None
        clamp_min_default: "f32[256, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[256, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[256, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_tensor, div_25);  div_tensor = div_25 = None
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

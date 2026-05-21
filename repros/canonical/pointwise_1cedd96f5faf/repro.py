"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_infer
Pattern hash: 1cedd96f5faf
Shape hash: 04355002
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
_shapes_config = "(T([512, 1280, 1, 1], f32), S([512, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_62: "f32[512, 1280, 1, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:325 in forward_head, code: x = self.act2(x)
        add_tensor: "f32[512, 1280, 1, 1]" = torch.ops.aten.add.Tensor(convolution_62, 3)
        clamp_min_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor: "f32[512, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_62, clamp_max_default);  convolution_62 = clamp_max_default = None
        div_tensor: "f32[512, 1280, 1, 1]" = torch.ops.aten.div.Tensor(mul_tensor, 6);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:326 in forward_head, code: x = self.flatten(x)
        reshape_default: "f32[512, 1280]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_0);  div_tensor = _shape_param_0 = None
        return reshape_default



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

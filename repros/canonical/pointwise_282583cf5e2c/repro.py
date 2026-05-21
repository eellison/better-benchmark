"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 282583cf5e2c
Shape hash: 7d4e313d
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
_shapes_config = "(T([128, 1536, 6, 6], f32, stride=(55296, 1, 9216, 1536)), T([128, 1536, 6, 6], f32, stride=(55296, 1, 9216, 1536)))"

class Repro(torch.nn.Module):
    def forward(self, add_110: "f32[128, 1536, 6, 6]", add_109: "f32[128, 1536, 6, 6]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_110, 0.5);  add_110 = None
        mul_tensor_1: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_109, add_109)
        mul_tensor_2: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_1, -0.5);  mul_tensor_1 = None
        exp_default: "f32[128, 1536, 6, 6]" = torch.ops.aten.exp.default(mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_4: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_109, mul_tensor_3);  add_109 = mul_tensor_3 = None
        add_tensor: "f32[128, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = mul_tensor_4 = None
        return add_tensor



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

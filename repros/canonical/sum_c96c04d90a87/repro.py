"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: c96c04d90a87
Shape hash: 0cc1c5cb
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
_shapes_config = "(T([128, 2560, 7, 7], f32, stride=(125440, 1, 17920, 2560)), T([128, 2560, 1, 1], f32), T([128, 2560, 7, 7], f32, stride=(125440, 1, 17920, 2560)), S([2560]))"

class Repro(torch.nn.Module):
    def forward(self, mul_90: "f32[128, 2560, 7, 7]", div_13: "f32[128, 2560, 1, 1]", getitem_38: "f32[128, 2560, 7, 7]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_tensor: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_90, div_13);  mul_90 = div_13 = None
        mul_scalar: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Scalar(mul_tensor, 1);  mul_tensor = None
        mul_tensor_1: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_38, mul_scalar);  getitem_38 = mul_scalar = None
        sum_dim_int_list: "f32[1, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 2, 3], True);  mul_tensor_1 = None
        reshape_default: "f32[2560]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
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

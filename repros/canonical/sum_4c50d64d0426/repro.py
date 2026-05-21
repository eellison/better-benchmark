"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 4c50d64d0426
Shape hash: b88dd92d
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
_shapes_config = "(T([512, 72, 28, 28], f32, stride=(56448, 1, 2016, 72)), T([1, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([72], f32), T([72], f32), T([512, 72, 28, 28], f32, stride=(56448, 1, 2016, 72)), T([512, 72, 1, 1], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_18: "f32[512, 72, 28, 28]", getitem_37: "f32[1, 72, 1, 1]", rsqrt_18: "f32[1, 72, 1, 1]", primals_114: "f32[72]", primals_115: "f32[72]", getitem_379: "f32[512, 72, 28, 28]", convolution_20: "f32[512, 72, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        sub_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37);  convolution_18 = getitem_37 = None
        mul_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_18);  sub_tensor = rsqrt_18 = None
        unsqueeze_default: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
        unsqueeze_default_1: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_default_3: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor_2: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_379, add_tensor);  getitem_379 = add_tensor = None
        sum_dim_int_list: "f32[512, 72, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None
        gt_scalar: "b8[512, 72, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_20, -3.0)
        lt_scalar: "b8[512, 72, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_20, 3.0);  convolution_20 = None
        bitwise_and_tensor: "b8[512, 72, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_3: "f32[512, 72, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 72, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_3, full_default);  bitwise_and_tensor = mul_tensor_3 = full_default = None
        return where_self



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

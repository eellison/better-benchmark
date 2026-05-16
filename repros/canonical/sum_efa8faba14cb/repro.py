"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_training
Pattern hash: efa8faba14cb
Shape hash: e1521889
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_355: "f32[32, 120, 28, 28]", cat_8: "f32[32, 120, 28, 28]", convolution_28: "f32[32, 120, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[32, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_355, cat_8);  getitem_355 = cat_8 = None
        sum_dim_int_list: "f32[32, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2, 3], True);  mul_tensor = None
        gt_scalar: "b8[32, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_28, -3.0)
        lt_scalar: "b8[32, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_28, 3.0);  convolution_28 = None
        bitwise_and_tensor: "b8[32, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_1: "f32[32, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[32, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_1, full_default);  bitwise_and_tensor = mul_tensor_1 = full_default = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn(3010560, dtype=torch.float32, device='cuda').as_strided([32, 120, 28, 28], [94080, 1, 3360, 120]),  # getitem_355
    torch.randn(3010560, dtype=torch.float32, device='cuda').as_strided([32, 120, 28, 28], [94080, 1, 3360, 120]),  # cat_8
    torch.randn([32, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

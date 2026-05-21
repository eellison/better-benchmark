"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_train
Pattern hash: 346c3ebe70cc
Shape hash: 2afe50c5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, relu_2: "f32[512, 58, 56, 56]", full_default: "f32[]", getitem_293: "f32[512, 58, 56, 56]", convolution_3: "f32[512, 58, 56, 56]", unsqueeze_852: "f32[1, 58, 1, 1]", squeeze_10: "f32[58]", primals_24: "f32[58]", getitem_299: "f32[512, 24, 28, 28]", convolution_1: "f32[512, 24, 28, 28]", unsqueeze_876: "f32[1, 24, 1, 1]", squeeze_4: "f32[24]", primals_12: "f32[24]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        le_scalar: "b8[512, 58, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_self: "f32[512, 58, 56, 56]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_293);  le_scalar = full_default = getitem_293 = None
        sum_dim_int_list: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[512, 58, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_852);  convolution_3 = unsqueeze_852 = None
        mul_tensor: "f32[512, 58, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[58]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.228077168367346e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[58]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.228077168367346e-07);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_tensor_4: "f32[58]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  squeeze_10 = primals_24 = None
        unsqueeze_default_6: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 58, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 58, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 58, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 58, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        sum_dim_int_list_2: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_299, [0, 2, 3])
        sub_tensor_3: "f32[512, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_876);  convolution_1 = unsqueeze_876 = None
        mul_tensor_8: "f32[512, 24, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_299, sub_tensor_3)
        sum_dim_int_list_3: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[24]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.4912308673469386e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[24]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.4912308673469386e-06);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_tensor_12: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  squeeze_4 = primals_12 = None
        unsqueeze_default_15: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[512, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[512, 24, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_299, mul_tensor_14);  getitem_299 = mul_tensor_14 = None
        sub_tensor_5: "f32[512, 24, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[512, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        return (mul_tensor_7, mul_tensor_15)


def _default_make_inputs():
    return [
    torch.randn(93126656, dtype=torch.float32, device='cuda').as_strided([512, 58, 56, 56], [181888, 1, 3248, 58]),  # relu_2
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(93126656, dtype=torch.float32, device='cuda').as_strided([512, 58, 56, 56], [181888, 1, 3248, 58]),  # getitem_293
    torch.randn(93126656, dtype=torch.float32, device='cuda').as_strided([512, 58, 56, 56], [181888, 1, 3248, 58]),  # convolution_3
    torch.randn([1, 58, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([512, 24, 28, 28], [18816, 1, 672, 24]),  # getitem_299
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([512, 24, 28, 28], [18816, 1, 672, 24]),  # convolution_1
    torch.randn([1, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

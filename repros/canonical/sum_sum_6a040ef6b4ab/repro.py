"""
Standalone repro captured via capture_hook.
Label: torchbench_dcgan_train_001
Pattern hash: 6a040ef6b4ab
Shape hash: dbd9eff0
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1024, 128, 16, 16], f32), T([1024, 128, 16, 16], f32), T([1024, 128, 16, 16], f32), T([1, 128, 1, 1], f32), T([128], f32), T([128], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg12_1: "f32[1024, 128, 16, 16]", getitem_6: "f32[1024, 128, 16, 16]", arg10_1: "f32[1024, 128, 16, 16]", arg22_1: "f32[1, 128, 1, 1]", arg11_1: "f32[128]", arg3_1: "f32[128]"):
        # No stacktrace found for following nodes
        gt_scalar: "b8[1024, 128, 16, 16]" = torch.ops.aten.gt.Scalar(arg12_1, 0);  arg12_1 = None
        mul_tensor: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_6, 0.2)
        where_self: "f32[1024, 128, 16, 16]" = torch.ops.aten.where.self(gt_scalar, getitem_6, mul_tensor);  gt_scalar = getitem_6 = mul_tensor = None
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(arg10_1, arg22_1);  arg10_1 = arg22_1 = None
        mul_tensor_1: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 2, 3]);  mul_tensor_1 = None
        mul_tensor_2: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.814697265625e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_2, 0);  mul_tensor_2 = None
        unsqueeze_default_1: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_3: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.814697265625e-06)
        mul_tensor_4: "f32[128]" = torch.ops.aten.mul.Tensor(arg11_1, arg11_1)
        mul_tensor_5: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        unsqueeze_default_3: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_4: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_6: "f32[128]" = torch.ops.aten.mul.Tensor(arg11_1, arg3_1);  arg3_1 = None
        unsqueeze_default_6: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_7: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_7: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_7);  where_self = mul_tensor_7 = None
        sub_tensor_2: "f32[1024, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_8: "f32[1024, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_9: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg11_1);  sum_dim_int_list_1 = arg11_1 = None
        return (mul_tensor_8, mul_tensor_9)

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

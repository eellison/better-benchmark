"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: 82a3f0084247
Shape hash: f0c1d821
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([401408, 128], f32), T([128], f32), T([128, 128, 56, 56], f32), T([128, 56, 56, 1], f32), T([128, 56, 56, 1], f32), T([128], f32), T([128], f32), T([128, 56, 56, 1], f32), T([128, 56, 56, 1], f32), T([128, 56, 56, 128], f32), S([8192, 49, 128]), S([8192, 7, 7, 128]), S([128, 8, 8, 7, 7, 128]), S([128, 56, 56, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_198: "f32[401408, 128]", arg4_1: "f32[128]", arg180_1: "f32[128, 128, 56, 56]", arg181_1: "f32[128, 56, 56, 1]", arg182_1: "f32[128, 56, 56, 1]", arg2_1: "f32[128]", arg3_1: "f32[128]", arg183_1: "f32[128, 56, 56, 1]", arg184_1: "f32[128, 56, 56, 1]", view_783: "f32[128, 56, 56, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[8192, 49, 128]" = torch.ops.aten.view.default(mm_198, _shape_param_0);  mm_198 = _shape_param_0 = None
        view_default_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        view_default_2: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5]);  view_default_2 = None
        clone_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        mul_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_default_3, arg4_1);  arg4_1 = None
        mul_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        permute_default_1: "f32[128, 56, 56, 128]" = torch.ops.aten.permute.default(arg180_1, [0, 2, 3, 1]);  arg180_1 = None
        sub_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute_default_1, arg181_1);  permute_default_1 = arg181_1 = None
        mul_tensor_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, arg182_1);  sub_tensor = None
        mul_tensor_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg2_1)
        add_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg3_1);  mul_tensor_3 = arg3_1 = None
        sub_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(add_tensor, arg183_1);  add_tensor = arg183_1 = None
        mul_tensor_4: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_1, arg184_1);  sub_tensor_1 = None
        mul_tensor_5: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [3], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_2: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_3: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_6);  sub_tensor_2 = mul_tensor_6 = None
        div_tensor: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(arg184_1, 128);  arg184_1 = None
        mul_tensor_7: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_3);  div_tensor = sub_tensor_3 = None
        mul_tensor_8: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_default_3, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1, 2]);  mul_tensor_8 = None
        sum_dim_int_list_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_default_3, [0, 1, 2]);  view_default_3 = None
        add_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_783, mul_tensor_7);  view_783 = mul_tensor_7 = None
        mul_tensor_9: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg2_1);  arg2_1 = None
        mul_tensor_10: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_9, 128)
        sum_dim_int_list_4: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [3], True)
        mul_tensor_11: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_tensor_2);  mul_tensor_9 = None
        sum_dim_int_list_5: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [3], True);  mul_tensor_11 = None
        mul_tensor_12: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_5);  sum_dim_int_list_5 = None
        sub_tensor_4: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_10, sum_dim_int_list_4);  mul_tensor_10 = sum_dim_int_list_4 = None
        sub_tensor_5: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_12);  sub_tensor_4 = mul_tensor_12 = None
        div_tensor_1: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(arg182_1, 128);  arg182_1 = None
        mul_tensor_13: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_5);  div_tensor_1 = sub_tensor_5 = None
        mul_tensor_14: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_6: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1, 2]);  mul_tensor_14 = None
        sum_dim_int_list_7: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1, 2]);  add_tensor_1 = None
        permute_default_2: "f32[128, 128, 56, 56]" = torch.ops.aten.permute.default(mul_tensor_13, [0, 3, 1, 2]);  mul_tensor_13 = None
        sum_dim_int_list_8: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_default_2, [0, 2, 3]);  permute_default_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, sum_dim_int_list_6, sum_dim_int_list_7, sum_dim_int_list_8)

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

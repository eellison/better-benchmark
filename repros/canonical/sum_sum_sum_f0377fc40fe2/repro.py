"""
Standalone repro captured via capture_hook.
Label: timm_deit_tiny_patch16_224.fb_in1k_train_001
Pattern hash: f0377fc40fe2
Shape hash: f26c5137
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
_shapes_config = "(T([25216, 192], f32), T([192], f32), T([128, 197, 192], f32), T([1, 197, 192], f32), T([128, 197, 1], f32), T([128, 197, 1], f32), T([128, 197, 192], f32), S([128, 197, 192]), S([128, 192, 14, 14]))"

class Repro(torch.nn.Module):
    def forward(self, mm_96: "f32[25216, 192]", arg3_1: "f32[192]", arg77_1: "f32[128, 197, 192]", arg2_1: "f32[1, 197, 192]", arg78_1: "f32[128, 197, 1]", arg79_1: "f32[128, 197, 1]", add_46: "f32[128, 197, 192]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 197, 192]" = torch.ops.aten.view.default(mm_96, _shape_param_0);  mm_96 = _shape_param_0 = None
        mul_tensor: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_default, arg3_1);  arg3_1 = None
        mul_tensor_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, 192)
        sum_dim_int_list: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        add_tensor: "f32[128, 197, 192]" = torch.ops.aten.add.Tensor(arg77_1, arg2_1);  arg77_1 = arg2_1 = None
        sub_tensor: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(add_tensor, arg78_1);  add_tensor = arg78_1 = None
        mul_tensor_2: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(sub_tensor, arg79_1);  sub_tensor = None
        mul_tensor_3: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 197, 1]" = torch.ops.aten.div.Tensor(arg79_1, 192);  arg79_1 = None
        mul_tensor_5: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        add_tensor_1: "f32[128, 197, 192]" = torch.ops.aten.add.Tensor(add_46, mul_tensor_5);  add_46 = mul_tensor_5 = None
        sum_dim_int_list_4: "f32[1, 197, 192]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0], True)
        slice_tensor: "f32[128, 1, 192]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 0, 1)
        slice_tensor_1: "f32[128, 196, 192]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 1, 197);  add_tensor_1 = None
        sum_dim_int_list_5: "f32[1, 1, 192]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0], True);  slice_tensor = None
        permute_default: "f32[128, 192, 196]" = torch.ops.aten.permute.default(slice_tensor_1, [0, 2, 1]);  slice_tensor_1 = None
        view_default_1: "f32[128, 192, 14, 14]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        sum_dim_int_list_6: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0, 2, 3]);  view_default_1 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, sum_dim_int_list_4, sum_dim_int_list_5, sum_dim_int_list_6)



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

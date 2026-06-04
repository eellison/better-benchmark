"""
Standalone repro captured via capture_hook.
Label: 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_train_010
Pattern hash: 5c7c5e63becb
Shape hash: 1f8157a1
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 2560], f32), T([2560], f32), T([16, 128, 2560], f32), T([16, 128, 1], f32), T([16, 128, 2560], f32), T([16, 128, 2560], b8), S([16, 128, 2560]), S([2048, 2560]), S([2560]))"

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[2048, 2560]", arg11_1: "f32[2560]", arg34_1: "f32[16, 128, 2560]", arg39_1: "f32[16, 128, 1]", arg43_1: "f32[16, 128, 2560]", arg33_1: "b8[16, 128, 2560]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[16, 128, 2560]" = torch.ops.aten.view.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(view_default, arg11_1);  arg11_1 = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, 2560)
        sum_dim_int_list: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg34_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(arg34_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(arg39_1, sub_tensor_1);  arg39_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(view_default, arg34_1);  arg34_1 = None
        sum_dim_int_list_2: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[2560]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(arg43_1, mul_tensor_4);  arg43_1 = mul_tensor_4 = None
        convert_element_type_default: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg33_1, torch.float32);  arg33_1 = None
        mul_tensor_6: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_7: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None
        view_default_1: "f32[2048, 2560]" = torch.ops.aten.view.default(mul_tensor_7, _shape_param_1);  mul_tensor_7 = _shape_param_1 = None
        permute_default: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_default_1, [1, 0])
        sum_dim_int_list_4: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[2560]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, view_default_2)

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

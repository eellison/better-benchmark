"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train_007
Pattern hash: 2005ccbce613
Shape hash: 851d64cc
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 1024], f32), T([8, 1024, 1024], f32), T([1024], f32), T([8192, 1024], f32), T([8, 1024, 1024], b8), T([8, 1024, 1024], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), S([8, 1024, 1024]), S([8, 1024, 1024]), S([8192, 1024]), S([1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[8192, 1024]", mul_4: "f32[8, 1024, 1024]", arg5_1: "f32[1024]", arg17_1: "f32[8192, 1024]", arg18_1: "b8[8, 1024, 1024]", arg0_1: "f32[8, 1024, 1024]", arg19_1: "f32[8, 1024, 1]", arg20_1: "f32[8, 1024, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None
        add_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_4, view_default);  mul_4 = view_default = None
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, arg5_1);  arg5_1 = None
        mul_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        view_default_1: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(arg17_1, _shape_param_1);  arg17_1 = _shape_param_1 = None
        mul_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(arg18_1, view_default_1);  view_default_1 = None
        mul_tensor_3: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        add_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(arg0_1, mul_tensor_3);  arg0_1 = mul_tensor_3 = None
        sub_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_1, arg19_1);  add_tensor_1 = arg19_1 = None
        mul_tensor_4: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, arg20_1);  sub_tensor = None
        mul_tensor_5: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(arg20_1, 1024);  arg20_1 = None
        mul_tensor_7: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_8: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1]);  add_tensor = None
        convert_element_type_default: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.float32);  arg18_1 = None
        mul_tensor_9: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_10: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None
        view_default_2: "f32[8192, 1024]" = torch.ops.aten.view.default(mul_tensor_10, _shape_param_2);  mul_tensor_10 = _shape_param_2 = None
        permute_default: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_default_2, [1, 0])
        sum_dim_int_list_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True);  view_default_2 = None
        view_default_3: "f32[1024]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_3);  sum_dim_int_list_4 = _shape_param_3 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, view_default_3)

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

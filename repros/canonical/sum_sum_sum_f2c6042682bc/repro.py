"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_004
Pattern hash: f2c6042682bc
Shape hash: e38cb0c3
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
_shapes_config = "(T([8192, 768], f32), T([768], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), S([8, 1024, 768]), S([8, 1024, 768]), S([8192, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[8192, 768]", arg1_1: "f32[768]", arg5_1: "f32[8192, 768]", arg6_1: "f32[8, 1024, 1]", arg7_1: "f32[8, 1024, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_default, arg1_1);  arg1_1 = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        view_default_1: "f32[8, 1024, 768]" = torch.ops.aten.view.default(arg5_1, _shape_param_1);  arg5_1 = _shape_param_1 = None
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_default_1, 0.5)
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_default_1, 0.7071067811865476)
        erf_default: "f32[8, 1024, 768]" = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor);  mul_tensor_2 = None
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_4, arg6_1);  mul_tensor_4 = arg6_1 = None
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, arg7_1);  sub_tensor = None
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_5);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True);  mul_tensor_6 = None
        mul_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_5, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_7);  sub_tensor_1 = mul_tensor_7 = None
        div_tensor: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(arg7_1, 768);  arg7_1 = None
        mul_tensor_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor_5);  mul_tensor_5 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        mul_tensor_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_default_1, view_default_1)
        mul_tensor_12: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_11, -0.5);  mul_tensor_11 = None
        exp_default: "f32[8, 1024, 768]" = torch.ops.aten.exp.default(mul_tensor_12);  mul_tensor_12 = None
        mul_tensor_13: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_default_1, mul_tensor_13);  view_default_1 = mul_tensor_13 = None
        add_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_14);  mul_tensor_10 = mul_tensor_14 = None
        mul_tensor_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_8, add_tensor_1);  mul_tensor_8 = add_tensor_1 = None
        view_default_2: "f32[8192, 768]" = torch.ops.aten.view.default(mul_tensor_15, _shape_param_2);  mul_tensor_15 = _shape_param_2 = None
        permute_default: "f32[768, 8192]" = torch.ops.aten.permute.default(view_default_2, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True);  view_default_2 = None
        view_default_3: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_3);  sum_dim_int_list_4 = _shape_param_3 = None
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

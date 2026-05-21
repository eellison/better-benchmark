"""
Standalone repro captured via capture_hook.
Label: timm_timm_vit_base_patch14_dinov2.lvd142m_train_train_001
Pattern hash: 41857d0f0554
Shape hash: 53398d19
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
_shapes_config = "(T([175360, 768], f32), T([768], f32), T([128, 1370, 768], f32), T([128, 1370, 1], f32), T([128, 1370, 768], f32), T([175360, 768], f32), T([768], f32), S([128, 1370, 768]), S([128, 1370, 768]), S([768]), S([175360, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_90: "f32[175360, 768]", arg7_1: "f32[768]", arg112_1: "f32[128, 1370, 768]", arg306_1: "f32[128, 1370, 1]", add_43: "f32[128, 1370, 768]", arg111_1: "f32[175360, 768]", arg6_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[128, 1370, 768]" = torch.ops.aten.view.default(mm_90, _shape_param_0);  mm_90 = _shape_param_0 = None
        mul_tensor: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(view_default, arg7_1);  arg7_1 = None
        mul_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg112_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg112_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg306_1, sub_tensor_1);  arg306_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(view_default, arg112_1);  arg112_1 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        add_tensor: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(add_43, mul_tensor_4);  add_43 = mul_tensor_4 = None
        view_default_1: "f32[128, 1370, 768]" = torch.ops.aten.view.default(arg111_1, _shape_param_1);  arg111_1 = _shape_param_1 = None
        mul_tensor_6: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(add_tensor, view_default_1);  view_default_1 = None
        mul_tensor_7: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(add_tensor, arg6_1);  add_tensor = arg6_1 = None
        sum_dim_int_list_4: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], True);  mul_tensor_6 = None
        view_default_2: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None
        view_default_3: "f32[175360, 768]" = torch.ops.aten.view.default(mul_tensor_7, _shape_param_3);  mul_tensor_7 = _shape_param_3 = None
        permute_default: "f32[768, 175360]" = torch.ops.aten.permute.default(view_default_3, [1, 0])
        sum_dim_int_list_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True);  view_default_3 = None
        view_default_4: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_5, _shape_param_4);  sum_dim_int_list_5 = _shape_param_4 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, view_default_2, permute_default, view_default_4)



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

"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train_001
Pattern hash: dcff8112d43c
Shape hash: 61765989
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
_shapes_config = "(T([25216, 768], f32), T([768], f32), T([25216, 768], f32), T([768], f32), T([128, 197, 768], f32), T([128, 197, 1], f32), T([128, 197, 1], f32), T([128, 197, 768], f32), S([128, 197, 768]), S([128, 197, 768]), S([25216, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_92: "f32[25216, 768]", arg8_1: "f32[768]", arg124_1: "f32[25216, 768]", arg2_1: "f32[768]", arg112_1: "f32[128, 197, 768]", arg125_1: "f32[128, 197, 1]", arg126_1: "f32[128, 197, 1]", add_43: "f32[128, 197, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[128, 197, 768]" = torch.ops.aten.view.default(mm_92, _shape_param_0);  mm_92 = _shape_param_0 = None
        mul_tensor: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(view_default, arg8_1);  view_default = arg8_1 = None
        mul_tensor_1: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        view_default_1: "f32[128, 197, 768]" = torch.ops.aten.view.default(arg124_1, _shape_param_1);  arg124_1 = _shape_param_1 = None
        mul_tensor_2: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(arg2_1, view_default_1);  view_default_1 = None
        add_tensor: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(arg112_1, mul_tensor_2);  arg112_1 = mul_tensor_2 = None
        sub_tensor: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(add_tensor, arg125_1);  add_tensor = arg125_1 = None
        mul_tensor_3: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, arg126_1);  sub_tensor = None
        mul_tensor_4: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_3);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_3, sum_dim_int_list_1);  mul_tensor_3 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_5);  sub_tensor_1 = mul_tensor_5 = None
        div_tensor: "f32[128, 197, 1]" = torch.ops.aten.div.Tensor(arg126_1, 768);  arg126_1 = None
        mul_tensor_6: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(add_43, mul_tensor_6);  add_43 = mul_tensor_6 = None
        mul_tensor_7: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg2_1);  add_tensor_1 = arg2_1 = None
        view_default_2: "f32[25216, 768]" = torch.ops.aten.view.default(mul_tensor_7, _shape_param_2);  mul_tensor_7 = _shape_param_2 = None
        return view_default_2



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

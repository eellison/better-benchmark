"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: e2d4961f3571
Shape hash: 3ba71d70
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32768, 512], f32), T([512], f32), T([32768, 512], f32), T([256, 128, 1], f32), T([256, 128, 1], f32), T([], f32), S([256, 128, 512]), S([256, 128, 512]), S([32768, 512]), S([512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_1: "f32[32768, 512]", arg581_1: "f32[512]", arg1119_1: "f32[32768, 512]", arg1120_1: "f32[256, 128, 1]", arg1121_1: "f32[256, 128, 1]", full_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_1, _shape_param_0);  mm_1 = _shape_param_0 = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(view_default, arg581_1);  arg581_1 = None
        mul_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        view_default_1: "f32[256, 128, 512]" = torch.ops.aten.view.default(arg1119_1, _shape_param_1);  arg1119_1 = _shape_param_1 = None
        relu_default: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_default_1);  view_default_1 = None
        sub_tensor: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(relu_default, arg1120_1);  arg1120_1 = None
        mul_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, arg1121_1);  sub_tensor = None
        mul_tensor_3: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(arg1121_1, 512);  arg1121_1 = None
        mul_tensor_5: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        le_scalar: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[256, 128, 512]" = torch.ops.aten.where.self(le_scalar, full_1, mul_tensor_5);  le_scalar = full_1 = mul_tensor_5 = None
        view_default_2: "f32[32768, 512]" = torch.ops.aten.view.default(where_self, _shape_param_2);  where_self = _shape_param_2 = None
        permute_default: "f32[512, 32768]" = torch.ops.aten.permute.default(view_default_2, [1, 0])
        sum_dim_int_list_4: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True);  view_default_2 = None
        view_default_3: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_3);  sum_dim_int_list_4 = _shape_param_3 = None
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

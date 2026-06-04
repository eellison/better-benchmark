"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: b35553d96630
Shape hash: 572a710f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([100352, 256], f32), T([256], f32), T([100352, 256], f32), T([128, 28, 28, 1], f32), T([128, 28, 28, 1], f32), T([128, 28, 28, 256], f32), S([2048, 49, 256]), S([2048, 7, 7, 256]), S([128, 4, 4, 7, 7, 256]), S([128, 28, 28, 256]), S([128, 28, 28, 256]), S([100352, 256]))"

class Repro(torch.nn.Module):
    def forward(self, mm_180: "f32[100352, 256]", arg20_1: "f32[256]", arg205_1: "f32[100352, 256]", arg206_1: "f32[128, 28, 28, 1]", arg207_1: "f32[128, 28, 28, 1]", view_712: "f32[128, 28, 28, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[2048, 49, 256]" = torch.ops.aten.view.default(mm_180, _shape_param_0);  mm_180 = _shape_param_0 = None
        view_default_1: "f32[2048, 7, 7, 256]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        view_default_2: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5]);  view_default_2 = None
        clone_default: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[128, 28, 28, 256]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        mul_tensor: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_default_3, arg20_1);  arg20_1 = None
        mul_tensor_1: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, 256)
        sum_dim_int_list: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        view_default_4: "f32[128, 28, 28, 256]" = torch.ops.aten.view.default(arg205_1, _shape_param_4);  arg205_1 = _shape_param_4 = None
        sub_tensor: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(view_default_4, arg206_1);  view_default_4 = arg206_1 = None
        mul_tensor_2: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, arg207_1);  sub_tensor = None
        mul_tensor_3: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 28, 28, 1]" = torch.ops.aten.div.Tensor(arg207_1, 256);  arg207_1 = None
        mul_tensor_5: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_default_3, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1, 2]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_default_3, [0, 1, 2]);  view_default_3 = None
        add_tensor: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_712, mul_tensor_5);  view_712 = mul_tensor_5 = None
        view_default_5: "f32[100352, 256]" = torch.ops.aten.view.default(add_tensor, _shape_param_5);  add_tensor = _shape_param_5 = None
        permute_default_1: "f32[256, 100352]" = torch.ops.aten.permute.default(view_default_5, [1, 0]);  view_default_5 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default_1)

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

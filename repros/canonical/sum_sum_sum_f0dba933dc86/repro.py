"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_train_001
Pattern hash: f0dba933dc86
Shape hash: 2f2590a2
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 768], f32), T([128, 768], f32), T([768], f32), T([128, 198, 768], f32), T([128, 198, 1], f32), S([25344, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 768]", mm_2: "f32[128, 768]", arg75_1: "f32[768]", arg236_1: "f32[128, 198, 768]", arg239_1: "f32[128, 198, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        full_default: "f32[128, 198, 768]" = torch.ops.aten.full.default([128, 198, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[128, 198, 768]" = torch.ops.aten.select_scatter.default(full_default, mm, 1, 1);  mm = None
        select_scatter_default_1: "f32[128, 198, 768]" = torch.ops.aten.select_scatter.default(full_default, mm_2, 1, 0);  full_default = mm_2 = None
        add_tensor: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        mul_tensor: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(add_tensor, arg75_1);  arg75_1 = None
        mul_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg236_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(arg236_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(arg239_1, sub_tensor_1);  arg239_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(add_tensor, arg236_1);  arg236_1 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1]);  add_tensor = None
        view_default: "f32[25344, 768]" = torch.ops.aten.view.default(mul_tensor_4, _shape_param_0);  mul_tensor_4 = _shape_param_0 = None
        permute_default: "f32[768, 25344]" = torch.ops.aten.permute.default(view_default, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default, [0], True);  view_default = None
        view_default_1: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_1);  sum_dim_int_list_4 = _shape_param_1 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, view_default_1)

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

"""
Standalone repro captured via capture_hook.
Label: timm_timm_vit_base_patch14_dinov2.lvd142m_train_train_001
Pattern hash: eca209605dde
Shape hash: 2403eb54
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
_shapes_config = "(T([128, 768], f32), T([768], f32), T([128, 1370, 768], f32), T([128, 1370, 1], f32), T([768], f32), S([175360, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg307_1: "f32[128, 768]", arg99_1: "f32[768]", arg282_1: "f32[128, 1370, 768]", arg283_1: "f32[128, 1370, 1]", arg98_1: "f32[768]", _shape_param_0):
        # No stacktrace found for following nodes
        full_default: "f32[128, 1370, 768]" = torch.ops.aten.full.default([128, 1370, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[128, 1370, 768]" = torch.ops.aten.select_scatter.default(full_default, arg307_1, 1, 0);  full_default = arg307_1 = None
        mul_tensor: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(select_scatter_default, arg99_1);  select_scatter_default = arg99_1 = None
        mul_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg282_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg282_1, sum_dim_int_list_1);  arg282_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg283_1, sub_tensor_1);  arg283_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, arg98_1);  mul_tensor_4 = arg98_1 = None
        view_default: "f32[175360, 768]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_0);  mul_tensor_5 = _shape_param_0 = None
        return view_default



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

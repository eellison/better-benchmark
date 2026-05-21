"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train_001
Pattern hash: 1078feabd90d
Shape hash: 258a9a8a
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
_shapes_config = "(T([128, 768], f32), T([768], f32), T([128, 768], f32), T([128, 1], f32), T([768], f32), S([128, 196, 768]), S([25216, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 768]", arg110_1: "f32[768]", arg307_1: "f32[128, 768]", arg309_1: "f32[128, 1]", arg106_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mm, arg110_1);  mm = arg110_1 = None
        mul_tensor_1: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_2: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg307_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [1], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 768]" = torch.ops.aten.mul.Tensor(arg307_1, sum_dim_int_list_1);  arg307_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 768]" = torch.ops.aten.mul.Tensor(arg309_1, sub_tensor_1);  arg309_1 = sub_tensor_1 = None
        unsqueeze_default: "f32[128, 1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 1);  mul_tensor_4 = None
        expand_default: "f32[128, 196, 768]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        div_scalar: "f32[128, 196, 768]" = torch.ops.aten.div.Scalar(expand_default, 196);  expand_default = None
        full_default: "f32[128, 197, 768]" = torch.ops.aten.full.default([128, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[128, 197, 768]" = torch.ops.aten.slice_scatter.default(full_default, div_scalar, 1, 1, 9223372036854775807);  full_default = div_scalar = None
        mul_tensor_5: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter_default, arg106_1);  slice_scatter_default = arg106_1 = None
        view_default: "f32[25216, 768]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_1);  mul_tensor_5 = _shape_param_1 = None
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

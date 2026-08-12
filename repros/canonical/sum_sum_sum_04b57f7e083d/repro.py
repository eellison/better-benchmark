"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch14_dinov2.lvd142m_train
Pattern hash: 04b57f7e083d
Shape hash: ef8b4ed0
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[128, 768]", arg1_1: "f32[768]", arg2_1: "f32[128, 1370, 768]", arg3_1: "f32[128, 1370, 1]", arg4_1: "bf16[175360, 768]", arg5_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        full: "f32[128, 1370, 768]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "f32[128, 1370, 768]" = torch.ops.aten.select_scatter.default(full, arg0_1, 1, 0);  full = arg0_1 = None
        mul: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(select_scatter, arg1_1);  arg1_1 = None
        mul_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = None
        sum_2: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  sum_2 = None
        sub: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg3_1, sub_1);  arg3_1 = sub_1 = None
        mul_5: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(select_scatter, arg2_1);  arg2_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None
        view: "bf16[128, 1370, 768]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        mul_6: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_4, view);  view = None
        mul_7: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg5_1);  arg5_1 = None
        convert_element_type: "bf16[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        sum_5: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1], True, dtype = torch.float32);  mul_6 = None
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_2);  sum_5 = _shape_param_2 = None
        view_2: "bf16[175360, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_3);  convert_element_type = _shape_param_3 = None
        permute: "bf16[768, 175360]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[768]" = torch.ops.aten.view.default(sum_6, _shape_param_4);  sum_6 = _shape_param_4 = None
        convert_element_type_1: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_2: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        return (mul_4, sum_3, sum_4, view_1, view_2, permute, convert_element_type_2)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

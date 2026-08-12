"""
Standalone repro captured via capture_hook.
Label: timm_deit_tiny_patch16_224.fb_in1k_train
Pattern hash: 0931873d4f42
Shape hash: 7639f983
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
    def forward(self, arg0_1: "bf16[128, 192]", arg1_1: "f32[192]", arg2_1: "f32[128, 197, 192]", arg3_1: "f32[128, 197, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 192]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        full: "f32[128, 197, 192]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "f32[128, 197, 192]" = torch.ops.aten.select_scatter.default(full, convert_element_type, 1, 0);  full = convert_element_type = None
        mul: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(select_scatter, arg1_1);  arg1_1 = None
        mul_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, 192)
        sum_1: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = None
        sum_2: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  sum_2 = None
        sub: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(arg3_1, sub_1);  arg3_1 = sub_1 = None
        mul_5: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(select_scatter, arg2_1);  arg2_1 = None
        sum_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[192]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None
        convert_element_type_1: "bf16[128, 197, 192]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        view: "bf16[25216, 192]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        permute: "bf16[192, 25216]" = torch.ops.aten.permute.default(view, [1, 0])
        sum_5: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32)
        view_1: "f32[192]" = torch.ops.aten.view.default(sum_5, _shape_param_2);  sum_5 = _shape_param_2 = None
        convert_element_type_2: "bf16[192]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_3: "f32[192]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        return (mul_4, sum_3, sum_4, view, permute, convert_element_type_3)



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

"""
Standalone repro captured via capture_hook.
Label: deit_tiny_patch16_224.fb_in1k
Pattern hash: 811d62470089
Shape hash: fec840e6
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 192]", arg2_1: "f32[192]", arg3_1: "f32[128, 197, 192]", arg4_1: "f32[128, 197, 1]", arg5_1: "bf16[192, 768]", arg6_1: "bf16[25216, 768]", arg7_1: "bf16[25216, 768]", arg8_1: "bf16[768, 192]", arg9_1: "bf16[25216, 192]", arg10_1: "f32[192]", arg11_1: "f32[128, 197, 192]", arg12_1: "f32[128, 197, 1]", arg13_1: "bf16[192, 192]", arg14_1: "bf16[128, 3, 197, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        mm: "bf16[128, 192]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[128, 192]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        full: "f32[128, 197, 192]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "f32[128, 197, 192]" = torch.ops.aten.select_scatter.default(full, convert_element_type, 1, 0);  full = convert_element_type = None
        mul: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(select_scatter, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, 192)
        sum_1: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = None
        sum_2: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(arg3_1, sum_2);  sum_2 = None
        sub: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        mul_5: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(select_scatter, arg3_1);  arg3_1 = None
        sum_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[192]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None
        convert_element_type_1: "bf16[128, 197, 192]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        view: "bf16[25216, 192]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        mm_1: "bf16[25216, 768]" = torch.ops.aten.mm.default(view, arg5_1);  arg5_1 = None
        permute: "bf16[192, 25216]" = torch.ops.aten.permute.default(view, [1, 0])
        mm_2: "bf16[192, 768]" = torch.ops.aten.mm.default(permute, arg6_1);  permute = arg6_1 = None
        sum_5: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32);  view = None
        view_1: "f32[192]" = torch.ops.aten.view.default(sum_5, _shape_param_2);  sum_5 = _shape_param_2 = None
        convert_element_type_2: "bf16[192]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        view_2: "bf16[128, 197, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_3);  mm_1 = _shape_param_3 = None
        convert_element_type_3: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_4: "f32[192]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        convert_element_type_5: "f32[128, 197, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        view_3: "bf16[128, 197, 768]" = torch.ops.aten.view.default(arg7_1, _shape_param_4);  arg7_1 = _shape_param_4 = None
        convert_element_type_6: "f32[128, 197, 768]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_6: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476)
        erf: "f32[128, 197, 768]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_8: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_6)
        mul_9: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_8, -0.5);  mul_8 = None
        exp: "f32[128, 197, 768]" = torch.ops.aten.exp.default(mul_9);  mul_9 = None
        mul_10: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_11: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_10);  convert_element_type_6 = mul_10 = None
        add_1: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(mul_7, mul_11);  mul_7 = mul_11 = None
        mul_12: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, add_1);  convert_element_type_5 = add_1 = None
        convert_element_type_7: "bf16[128, 197, 768]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        view_4: "bf16[25216, 768]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_5);  convert_element_type_7 = _shape_param_5 = None
        mm_3: "bf16[25216, 192]" = torch.ops.aten.mm.default(view_4, arg8_1);  arg8_1 = None
        permute_1: "bf16[768, 25216]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_4: "bf16[768, 192]" = torch.ops.aten.mm.default(permute_1, arg9_1);  permute_1 = arg9_1 = None
        sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[768]" = torch.ops.aten.view.default(sum_6, _shape_param_6);  sum_6 = _shape_param_6 = None
        convert_element_type_8: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[128, 197, 192]" = torch.ops.aten.view.default(mm_3, _shape_param_7);  mm_3 = _shape_param_7 = None
        convert_element_type_9: "f32[128, 197, 192]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_10: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_11: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        mul_13: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_9, arg10_1);  arg10_1 = None
        mul_14: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul_13, 192)
        sum_7: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_13, [2], True)
        mul_15: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul_13, arg11_1);  mul_13 = None
        sum_8: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True);  mul_15 = None
        mul_16: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(arg11_1, sum_8);  sum_8 = None
        sub_2: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(mul_14, sum_7);  mul_14 = sum_7 = None
        sub_3: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(sub_2, mul_16);  sub_2 = mul_16 = None
        mul_17: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(arg12_1, sub_3);  arg12_1 = sub_3 = None
        mul_18: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_9, arg11_1);  arg11_1 = None
        sum_9: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_18, [0, 1]);  mul_18 = None
        sum_10: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [0, 1]);  convert_element_type_9 = None
        add_2: "f32[128, 197, 192]" = torch.ops.aten.add.Tensor(mul_4, mul_17);  mul_4 = mul_17 = None
        convert_element_type_12: "bf16[128, 197, 192]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
        view_7: "bf16[25216, 192]" = torch.ops.aten.view.default(convert_element_type_12, _shape_param_8);  convert_element_type_12 = _shape_param_8 = None
        mm_5: "bf16[25216, 192]" = torch.ops.aten.mm.default(view_7, arg13_1);  arg13_1 = None
        permute_2: "bf16[192, 25216]" = torch.ops.aten.permute.default(view_7, [1, 0])
        permute_3: "bf16[128, 197, 3, 64]" = torch.ops.aten.permute.default(arg14_1, [0, 2, 1, 3]);  arg14_1 = None
        view_8: "bf16[128, 197, 192]" = torch.ops.aten.view.default(permute_3, _shape_param_9);  permute_3 = _shape_param_9 = None
        view_9: "bf16[25216, 192]" = torch.ops.aten.view.default(view_8, _shape_param_10);  view_8 = _shape_param_10 = None
        mm_6: "bf16[192, 192]" = torch.ops.aten.mm.default(permute_2, view_9);  permute_2 = view_9 = None
        sum_11: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True, dtype = torch.float32);  view_7 = None
        view_10: "f32[192]" = torch.ops.aten.view.default(sum_11, _shape_param_11);  sum_11 = _shape_param_11 = None
        convert_element_type_13: "bf16[192]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        view_11: "bf16[128, 197, 192]" = torch.ops.aten.view.default(mm_5, _shape_param_12);  mm_5 = _shape_param_12 = None
        convert_element_type_14: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_15: "f32[192]" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        view_12: "bf16[128, 197, 3, 64]" = torch.ops.aten.view.default(view_11, _shape_param_13);  view_11 = _shape_param_13 = None
        permute_4: "bf16[128, 3, 197, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        return (sum_3, sum_4, convert_element_type_3, convert_element_type_4, convert_element_type_10, convert_element_type_11, sum_9, sum_10, add_2, convert_element_type_14, convert_element_type_15, permute_4)



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

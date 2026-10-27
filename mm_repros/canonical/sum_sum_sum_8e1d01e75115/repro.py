"""
Standalone repro captured via capture_hook.
Label: deit_base_distilled_patch16_224
Pattern hash: 8e1d01e75115
Shape hash: 839f124e
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 768]", arg2_1: "bf16[128, 768]", arg3_1: "bf16[1000, 768]", arg4_1: "bf16[128, 768]", arg5_1: "f32[768]", arg6_1: "f32[128, 198, 768]", arg7_1: "f32[128, 198, 1]", arg8_1: "bf16[768, 3072]", arg9_1: "bf16[25344, 3072]", arg10_1: "bf16[25344, 3072]", arg11_1: "bf16[3072, 768]", arg12_1: "bf16[25344, 768]", arg13_1: "f32[768]", arg14_1: "f32[128, 198, 768]", arg15_1: "f32[128, 198, 1]", arg16_1: "bf16[768, 768]", arg17_1: "bf16[128, 12, 198, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        div: "bf16[128, 1000]" = torch.ops.aten.div.Tensor(arg0_1, 2);  arg0_1 = None
        mm: "bf16[128, 768]" = torch.ops.aten.mm.default(div, arg1_1);  arg1_1 = None
        permute: "bf16[1000, 128]" = torch.ops.aten.permute.default(div, [1, 0])
        mm_1: "bf16[1000, 768]" = torch.ops.aten.mm.default(permute, arg2_1);  arg2_1 = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div, [0], True, dtype = torch.float32)
        view: "f32[1000]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type: "bf16[1000]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_1: "f32[128, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_2: "f32[1000, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_3: "f32[1000]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32)
        mm_2: "bf16[128, 768]" = torch.ops.aten.mm.default(div, arg3_1);  div = arg3_1 = None
        mm_3: "bf16[1000, 768]" = torch.ops.aten.mm.default(permute, arg4_1);  permute = arg4_1 = None
        convert_element_type_4: "f32[128, 768]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_5: "f32[1000, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_6: "f32[1000]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        full: "f32[128, 198, 768]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        select_scatter: "f32[128, 198, 768]" = torch.ops.aten.select_scatter.default(full, convert_element_type_1, 1, 1);  convert_element_type_1 = None
        select_scatter_1: "f32[128, 198, 768]" = torch.ops.aten.select_scatter.default(full, convert_element_type_4, 1, 0);  full = convert_element_type_4 = None
        add: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        mul: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(add, arg5_1);  arg5_1 = None
        mul_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_2: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = None
        sum_3: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(arg6_1, sum_3);  sum_3 = None
        sub: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_2);  mul_1 = sum_2 = None
        sub_1: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(arg7_1, sub_1);  arg7_1 = sub_1 = None
        mul_5: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(add, arg6_1);  arg6_1 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(add, [0, 1]);  add = None
        convert_element_type_7: "bf16[128, 198, 768]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        view_1: "bf16[25344, 768]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_2);  convert_element_type_7 = _shape_param_2 = None
        mm_4: "bf16[25344, 3072]" = torch.ops.aten.mm.default(view_1, arg8_1);  arg8_1 = None
        permute_1: "bf16[768, 25344]" = torch.ops.aten.permute.default(view_1, [1, 0])
        mm_5: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_1, arg9_1);  permute_1 = arg9_1 = None
        sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32);  view_1 = None
        view_2: "f32[768]" = torch.ops.aten.view.default(sum_6, _shape_param_3);  sum_6 = _shape_param_3 = None
        convert_element_type_8: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        view_3: "bf16[128, 198, 3072]" = torch.ops.aten.view.default(mm_4, _shape_param_4);  mm_4 = _shape_param_4 = None
        convert_element_type_9: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_10: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        convert_element_type_11: "f32[128, 198, 3072]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        view_4: "bf16[128, 198, 3072]" = torch.ops.aten.view.default(arg10_1, _shape_param_5);  arg10_1 = _shape_param_5 = None
        convert_element_type_12: "f32[128, 198, 3072]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_6: "f32[128, 198, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 0.7071067811865476)
        erf: "f32[128, 198, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_1: "f32[128, 198, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[128, 198, 3072]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_8: "f32[128, 198, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_12, convert_element_type_12)
        mul_9: "f32[128, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_8, -0.5);  mul_8 = None
        exp: "f32[128, 198, 3072]" = torch.ops.aten.exp.default(mul_9);  mul_9 = None
        mul_10: "f32[128, 198, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_11: "f32[128, 198, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_12, mul_10);  convert_element_type_12 = mul_10 = None
        add_2: "f32[128, 198, 3072]" = torch.ops.aten.add.Tensor(mul_7, mul_11);  mul_7 = mul_11 = None
        mul_12: "f32[128, 198, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_11, add_2);  convert_element_type_11 = add_2 = None
        convert_element_type_13: "bf16[128, 198, 3072]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        view_5: "bf16[25344, 3072]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_6);  convert_element_type_13 = _shape_param_6 = None
        mm_6: "bf16[25344, 768]" = torch.ops.aten.mm.default(view_5, arg11_1);  arg11_1 = None
        permute_2: "bf16[3072, 25344]" = torch.ops.aten.permute.default(view_5, [1, 0])
        mm_7: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_2, arg12_1);  permute_2 = arg12_1 = None
        sum_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32);  view_5 = None
        view_6: "f32[3072]" = torch.ops.aten.view.default(sum_7, _shape_param_7);  sum_7 = _shape_param_7 = None
        convert_element_type_14: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        view_7: "bf16[128, 198, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_8);  mm_6 = _shape_param_8 = None
        convert_element_type_15: "f32[128, 198, 768]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        convert_element_type_16: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_17: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        mul_13: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, arg13_1);  arg13_1 = None
        mul_14: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_13, 768)
        sum_8: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_13, [2], True)
        mul_15: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_13, arg14_1);  mul_13 = None
        sum_9: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True);  mul_15 = None
        mul_16: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(arg14_1, sum_9);  sum_9 = None
        sub_2: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(mul_14, sum_8);  mul_14 = sum_8 = None
        sub_3: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(sub_2, mul_16);  sub_2 = mul_16 = None
        mul_17: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(arg15_1, sub_3);  arg15_1 = sub_3 = None
        mul_18: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, arg14_1);  arg14_1 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_18, [0, 1]);  mul_18 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_15, [0, 1]);  convert_element_type_15 = None
        add_3: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(mul_4, mul_17);  mul_4 = mul_17 = None
        convert_element_type_18: "bf16[128, 198, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        view_8: "bf16[25344, 768]" = torch.ops.aten.view.default(convert_element_type_18, _shape_param_9);  convert_element_type_18 = _shape_param_9 = None
        mm_8: "bf16[25344, 768]" = torch.ops.aten.mm.default(view_8, arg16_1);  arg16_1 = None
        permute_3: "bf16[768, 25344]" = torch.ops.aten.permute.default(view_8, [1, 0])
        permute_4: "bf16[128, 198, 12, 64]" = torch.ops.aten.permute.default(arg17_1, [0, 2, 1, 3]);  arg17_1 = None
        view_9: "bf16[128, 198, 768]" = torch.ops.aten.view.default(permute_4, _shape_param_10);  permute_4 = _shape_param_10 = None
        view_10: "bf16[25344, 768]" = torch.ops.aten.view.default(view_9, _shape_param_11);  view_9 = _shape_param_11 = None
        mm_9: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_3, view_10);  permute_3 = view_10 = None
        sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_11: "f32[768]" = torch.ops.aten.view.default(sum_12, _shape_param_12);  sum_12 = _shape_param_12 = None
        convert_element_type_19: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        view_12: "bf16[128, 198, 768]" = torch.ops.aten.view.default(mm_8, _shape_param_13);  mm_8 = _shape_param_13 = None
        convert_element_type_20: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_21: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_19, torch.float32);  convert_element_type_19 = None
        view_13: "bf16[128, 198, 12, 64]" = torch.ops.aten.view.default(view_12, _shape_param_14);  view_12 = _shape_param_14 = None
        permute_5: "bf16[128, 12, 198, 64]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        return (convert_element_type_2, convert_element_type_3, convert_element_type_5, convert_element_type_6, sum_4, sum_5, convert_element_type_9, convert_element_type_10, convert_element_type_16, convert_element_type_17, sum_10, sum_11, add_3, convert_element_type_20, convert_element_type_21, permute_5)



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

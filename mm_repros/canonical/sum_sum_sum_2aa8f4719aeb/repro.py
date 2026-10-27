"""
Standalone repro captured via capture_hook.
Label: beit_base_patch16_224
Pattern hash: 2aa8f4719aeb
Shape hash: cab3126e
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 768]", arg2_1: "f32[768]", arg3_1: "f32[128, 768]", arg4_1: "f32[128, 1]", arg5_1: "f32[768]", arg6_1: "bf16[25216, 768]", arg7_1: "bf16[768, 3072]", arg8_1: "bf16[25216, 3072]", arg9_1: "bf16[25216, 3072]", arg10_1: "bf16[3072, 768]", arg11_1: "bf16[25216, 768]", arg12_1: "f32[768]", arg13_1: "f32[128, 197, 768]", arg14_1: "f32[128, 197, 1]", arg15_1: "f32[768]", arg16_1: "bf16[25216, 768]", arg17_1: "bf16[768, 768]", arg18_1: "bf16[128, 12, 197, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # No stacktrace found for following nodes
        mm: "bf16[128, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[128, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mul: "f32[128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        mul_2: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = None
        sum_2: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [1], True);  mul_2 = None
        mul_3: "f32[128, 768]" = torch.ops.aten.mul.Tensor(arg3_1, sum_2);  sum_2 = None
        sub: "f32[128, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[128, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        mul_5: "f32[128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, arg3_1);  arg3_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0]);  mul_5 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0]);  convert_element_type = None
        unsqueeze: "f32[128, 1, 768]" = torch.ops.aten.unsqueeze.default(mul_4, 1);  mul_4 = None
        expand: "f32[128, 196, 768]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        div: "f32[128, 196, 768]" = torch.ops.aten.div.Scalar(expand, 196);  expand = None
        full: "f32[128, 197, 768]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        slice_scatter: "f32[128, 197, 768]" = torch.ops.aten.slice_scatter.default(full, div, 1, 1, 9223372036854775807);  full = div = None
        mul_6: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter, arg5_1);  arg5_1 = None
        view: "bf16[128, 197, 768]" = torch.ops.aten.view.default(arg6_1, _shape_param_2);  arg6_1 = _shape_param_2 = None
        mul_7: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter, view);  view = None
        sum_5: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1], True, dtype = torch.float32);  mul_7 = None
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_3);  sum_5 = _shape_param_3 = None
        convert_element_type_1: "bf16[128, 197, 768]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        view_2: "bf16[25216, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_4);  convert_element_type_1 = _shape_param_4 = None
        mm_1: "bf16[25216, 3072]" = torch.ops.aten.mm.default(view_2, arg7_1);  arg7_1 = None
        permute: "bf16[768, 25216]" = torch.ops.aten.permute.default(view_2, [1, 0])
        mm_2: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute, arg8_1);  permute = arg8_1 = None
        sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32);  view_2 = None
        view_3: "f32[768]" = torch.ops.aten.view.default(sum_6, _shape_param_5);  sum_6 = _shape_param_5 = None
        convert_element_type_2: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        view_4: "bf16[128, 197, 3072]" = torch.ops.aten.view.default(mm_1, _shape_param_6);  mm_1 = _shape_param_6 = None
        convert_element_type_3: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_4: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        convert_element_type_5: "f32[128, 197, 3072]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        view_5: "bf16[128, 197, 3072]" = torch.ops.aten.view.default(arg9_1, _shape_param_7);  arg9_1 = _shape_param_7 = None
        convert_element_type_6: "f32[128, 197, 3072]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        mul_8: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476)
        erf: "f32[128, 197, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add: "f32[128, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_10: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_6)
        mul_11: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_10, -0.5);  mul_10 = None
        exp: "f32[128, 197, 3072]" = torch.ops.aten.exp.default(mul_11);  mul_11 = None
        mul_12: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_13: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_12);  convert_element_type_6 = mul_12 = None
        add_1: "f32[128, 197, 3072]" = torch.ops.aten.add.Tensor(mul_9, mul_13);  mul_9 = mul_13 = None
        mul_14: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_5, add_1);  convert_element_type_5 = add_1 = None
        convert_element_type_7: "bf16[128, 197, 3072]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        view_6: "bf16[25216, 3072]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_8);  convert_element_type_7 = _shape_param_8 = None
        mm_3: "bf16[25216, 768]" = torch.ops.aten.mm.default(view_6, arg10_1);  arg10_1 = None
        permute_1: "bf16[3072, 25216]" = torch.ops.aten.permute.default(view_6, [1, 0])
        mm_4: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_1, arg11_1);  permute_1 = arg11_1 = None
        sum_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_6, [0], True, dtype = torch.float32);  view_6 = None
        view_7: "f32[3072]" = torch.ops.aten.view.default(sum_7, _shape_param_9);  sum_7 = _shape_param_9 = None
        convert_element_type_8: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        view_8: "bf16[128, 197, 768]" = torch.ops.aten.view.default(mm_3, _shape_param_10);  mm_3 = _shape_param_10 = None
        convert_element_type_9: "f32[128, 197, 768]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        convert_element_type_10: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_11: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        mul_15: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, arg12_1);  arg12_1 = None
        mul_16: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_15, 768)
        sum_8: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True)
        mul_17: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_15, arg13_1);  mul_15 = None
        sum_9: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [2], True);  mul_17 = None
        mul_18: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(arg13_1, sum_9);  sum_9 = None
        sub_2: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(mul_16, sum_8);  mul_16 = sum_8 = None
        sub_3: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(sub_2, mul_18);  sub_2 = mul_18 = None
        mul_19: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(arg14_1, sub_3);  arg14_1 = sub_3 = None
        mul_20: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, arg13_1);  arg13_1 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_20, [0, 1]);  mul_20 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [0, 1]);  convert_element_type_9 = None
        add_2: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(slice_scatter, mul_19);  slice_scatter = mul_19 = None
        mul_21: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(add_2, arg15_1);  arg15_1 = None
        view_9: "bf16[128, 197, 768]" = torch.ops.aten.view.default(arg16_1, _shape_param_11);  arg16_1 = _shape_param_11 = None
        mul_22: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(add_2, view_9);  view_9 = None
        sum_12: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_22, [0, 1], True, dtype = torch.float32);  mul_22 = None
        view_10: "f32[768]" = torch.ops.aten.view.default(sum_12, _shape_param_12);  sum_12 = _shape_param_12 = None
        convert_element_type_12: "bf16[128, 197, 768]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        view_11: "bf16[25216, 768]" = torch.ops.aten.view.default(convert_element_type_12, _shape_param_13);  convert_element_type_12 = _shape_param_13 = None
        mm_5: "bf16[25216, 768]" = torch.ops.aten.mm.default(view_11, arg17_1);  arg17_1 = None
        permute_2: "bf16[768, 25216]" = torch.ops.aten.permute.default(view_11, [1, 0])
        permute_3: "bf16[128, 197, 12, 64]" = torch.ops.aten.permute.default(arg18_1, [0, 2, 1, 3]);  arg18_1 = None
        view_12: "bf16[128, 197, 768]" = torch.ops.aten.view.default(permute_3, _shape_param_14);  permute_3 = _shape_param_14 = None
        view_13: "bf16[25216, 768]" = torch.ops.aten.view.default(view_12, _shape_param_15);  view_12 = _shape_param_15 = None
        mm_6: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_2, view_13);  permute_2 = view_13 = None
        sum_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_14: "f32[768]" = torch.ops.aten.view.default(sum_13, _shape_param_16);  sum_13 = _shape_param_16 = None
        convert_element_type_13: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_14, torch.bfloat16);  view_14 = None
        view_15: "bf16[128, 197, 768]" = torch.ops.aten.view.default(mm_5, _shape_param_17);  mm_5 = _shape_param_17 = None
        convert_element_type_14: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_15: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        view_16: "bf16[128, 197, 12, 64]" = torch.ops.aten.view.default(view_15, _shape_param_18);  view_15 = _shape_param_18 = None
        permute_4: "bf16[128, 12, 197, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        return (sum_3, sum_4, view_1, convert_element_type_3, convert_element_type_4, convert_element_type_10, convert_element_type_11, sum_10, sum_11, add_2, view_10, convert_element_type_14, convert_element_type_15, permute_4)



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

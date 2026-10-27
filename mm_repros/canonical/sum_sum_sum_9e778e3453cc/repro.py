"""
Standalone repro captured via capture_hook.
Label: BlenderbotForCausalLM
Pattern hash: 9e778e3453cc
Shape hash: 0ac8d0fc
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
    def forward(self, arg0_1: "f32[32, 128, 2560]", arg1_1: "b8[32, 128, 2560]", arg2_1: "bf16[2560, 10240]", arg3_1: "bf16[4096, 10240]", arg4_1: "bf16[4096, 10240]", arg5_1: "bf16[10240, 2560]", arg6_1: "bf16[4096, 2560]", arg7_1: "f32[2560]", arg8_1: "bf16[4096, 2560]", arg9_1: "b8[32, 128, 2560]", arg10_1: "f32[32, 128, 2560]", arg11_1: "f32[32, 128, 1]", arg12_1: "f32[32, 128, 1]", arg13_1: "bf16[2560, 2560]", arg14_1: "bf16[32, 32, 128, 80]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16)
        convert_element_type_1: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[4096, 2560]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        mm: "bf16[4096, 10240]" = torch.ops.aten.mm.default(view, arg2_1);  arg2_1 = None
        permute: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view, [1, 0])
        mm_1: "bf16[2560, 10240]" = torch.ops.aten.mm.default(permute, arg3_1);  permute = arg3_1 = None
        sum_1: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32);  view = None
        view_1: "f32[2560]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        convert_element_type_2: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        view_2: "bf16[32, 128, 10240]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type_3: "f32[2560, 10240]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_4: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        convert_element_type_5: "f32[32, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        view_3: "bf16[32, 128, 10240]" = torch.ops.aten.view.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        convert_element_type_6: "f32[32, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_2: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476)
        erf: "f32[32, 128, 10240]" = torch.ops.aten.erf.default(mul_2);  mul_2 = None
        add: "f32[32, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_3: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_4: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_6)
        mul_5: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_4, -0.5);  mul_4 = None
        exp: "f32[32, 128, 10240]" = torch.ops.aten.exp.default(mul_5);  mul_5 = None
        mul_6: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_7: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_6);  convert_element_type_6 = mul_6 = None
        add_1: "f32[32, 128, 10240]" = torch.ops.aten.add.Tensor(mul_3, mul_7);  mul_3 = mul_7 = None
        mul_8: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_5, add_1);  convert_element_type_5 = add_1 = None
        convert_element_type_7: "bf16[32, 128, 10240]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_4: "bf16[4096, 10240]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_4);  convert_element_type_7 = _shape_param_4 = None
        mm_2: "bf16[4096, 2560]" = torch.ops.aten.mm.default(view_4, arg5_1);  arg5_1 = None
        permute_1: "bf16[10240, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_3: "bf16[10240, 2560]" = torch.ops.aten.mm.default(permute_1, arg6_1);  permute_1 = arg6_1 = None
        sum_2: "f32[1, 10240]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[10240]" = torch.ops.aten.view.default(sum_2, _shape_param_5);  sum_2 = _shape_param_5 = None
        convert_element_type_8: "bf16[10240]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(mm_2, _shape_param_6);  mm_2 = _shape_param_6 = None
        convert_element_type_9: "f32[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_10: "f32[10240, 2560]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_11: "f32[10240]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        mul_9: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_9, arg7_1);  arg7_1 = None
        mul_10: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_9, 2560)
        sum_3: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True)
        view_7: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(arg8_1, _shape_param_7);  arg8_1 = _shape_param_7 = None
        mul_11: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(arg9_1, view_7);  view_7 = None
        mul_12: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_11, 1.1111111111111112);  mul_11 = None
        add_2: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(arg10_1, mul_12);  arg10_1 = mul_12 = None
        sub: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(add_2, arg11_1);  add_2 = arg11_1 = None
        mul_13: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, arg12_1);  sub = None
        mul_14: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_9, mul_13);  mul_9 = None
        sum_4: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_14, [2], True);  mul_14 = None
        mul_15: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_13, sum_4);  sum_4 = None
        sub_1: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_10, sum_3);  mul_10 = sum_3 = None
        sub_2: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_1, mul_15);  sub_1 = mul_15 = None
        div: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(arg12_1, 2560);  arg12_1 = None
        mul_16: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_17: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_13);  mul_13 = None
        sum_5: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_17, [0, 1]);  mul_17 = None
        sum_6: "f32[2560]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [0, 1]);  convert_element_type_9 = None
        add_3: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(arg0_1, mul_16);  arg0_1 = mul_16 = None
        convert_element_type_12: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        convert_element_type_13: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        mul_18: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_19: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_12, mul_18);  convert_element_type_12 = mul_18 = None
        view_8: "bf16[4096, 2560]" = torch.ops.aten.view.default(mul_19, _shape_param_8);  mul_19 = _shape_param_8 = None
        mm_4: "bf16[4096, 2560]" = torch.ops.aten.mm.default(view_8, arg13_1);  arg13_1 = None
        permute_2: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view_8, [1, 0])
        permute_3: "bf16[32, 128, 32, 80]" = torch.ops.aten.permute.default(arg14_1, [0, 2, 1, 3]);  arg14_1 = None
        view_9: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(permute_3, _shape_param_9);  permute_3 = _shape_param_9 = None
        view_10: "bf16[4096, 2560]" = torch.ops.aten.view.default(view_9, _shape_param_10);  view_9 = _shape_param_10 = None
        mm_5: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_2, view_10);  permute_2 = view_10 = None
        sum_7: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_11: "f32[2560]" = torch.ops.aten.view.default(sum_7, _shape_param_11);  sum_7 = _shape_param_11 = None
        convert_element_type_14: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        view_12: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(mm_4, _shape_param_12);  mm_4 = _shape_param_12 = None
        convert_element_type_15: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_16: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        view_13: "bf16[32, 128, 32, 80]" = torch.ops.aten.view.default(view_12, _shape_param_13);  view_12 = _shape_param_13 = None
        permute_4: "bf16[32, 32, 128, 80]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_10, convert_element_type_11, sum_5, sum_6, add_3, convert_element_type_15, convert_element_type_16, permute_4)



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

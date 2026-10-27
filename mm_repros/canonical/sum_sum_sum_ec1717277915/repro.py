"""
Standalone repro captured via capture_hook.
Label: mobilevit_s
Pattern hash: ec1717277915
Shape hash: ab5fa714
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
    def forward(self, arg0_1: "bf16[128, 192, 16, 16]", arg1_1: "f32[192]", arg2_1: "bf16[512, 64, 192]", arg3_1: "f32[512, 64, 1]", arg4_1: "f32[512, 64, 1]", arg5_1: "bf16[192, 384]", arg6_1: "bf16[32768, 384]", arg7_1: "bf16[32768, 384]", arg8_1: "bf16[384, 192]", arg9_1: "bf16[32768, 192]", arg10_1: "f32[192]", arg11_1: "bf16[512, 64, 192]", arg12_1: "f32[512, 64, 1]", arg13_1: "f32[512, 64, 1]", arg14_1: "bf16[192, 192]", arg15_1: "bf16[512, 4, 64, 48]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 192, 16, 16]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        clone: "f32[128, 192, 16, 16]" = torch.ops.aten.clone.default(convert_element_type, memory_format = torch.contiguous_format);  convert_element_type = None
        view: "f32[196608, 2, 8, 2]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        permute: "f32[196608, 8, 2, 2]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        clone_1: "f32[196608, 8, 2, 2]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "f32[128, 192, 64, 4]" = torch.ops.aten.view.default(clone_1, _shape_param_1);  clone_1 = _shape_param_1 = None
        permute_1: "f32[128, 4, 64, 192]" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
        clone_2: "f32[128, 4, 64, 192]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "f32[512, 64, 192]" = torch.ops.aten.view.default(clone_2, _shape_param_2);  clone_2 = _shape_param_2 = None
        mul: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_2, arg1_1);  arg1_1 = None
        mul_1: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul, 192)
        sum_1: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        convert_element_type_1: "f32[512, 64, 192]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg3_1);  convert_element_type_1 = arg3_1 = None
        mul_2: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = None
        mul_3: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(arg4_1, 192);  arg4_1 = None
        mul_5: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_2, mul_2);  mul_2 = None
        sum_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_4: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_2, [0, 1]);  view_2 = None
        convert_element_type_2: "bf16[512, 64, 192]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_3: "bf16[32768, 192]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_3);  _shape_param_3 = None
        mm: "bf16[32768, 384]" = torch.ops.aten.mm.default(view_3, arg5_1);  arg5_1 = None
        permute_2: "bf16[192, 32768]" = torch.ops.aten.permute.default(view_3, [1, 0])
        mm_1: "bf16[192, 384]" = torch.ops.aten.mm.default(permute_2, arg6_1);  permute_2 = arg6_1 = None
        sum_5: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32);  view_3 = None
        view_4: "f32[192]" = torch.ops.aten.view.default(sum_5, _shape_param_4);  sum_5 = _shape_param_4 = None
        convert_element_type_3: "bf16[192]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        view_5: "bf16[512, 64, 384]" = torch.ops.aten.view.default(mm, _shape_param_5);  mm = _shape_param_5 = None
        convert_element_type_4: "f32[192, 384]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_5: "f32[192]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_6: "f32[512, 64, 384]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        view_6: "bf16[512, 64, 384]" = torch.ops.aten.view.default(arg7_1, _shape_param_6);  arg7_1 = _shape_param_6 = None
        convert_element_type_7: "f32[512, 64, 384]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        sigmoid: "f32[512, 64, 384]" = torch.ops.aten.sigmoid.default(convert_element_type_7)
        mul_7: "f32[512, 64, 384]" = torch.ops.aten.mul.Tensor(convert_element_type_6, sigmoid);  convert_element_type_6 = None
        sub_3: "f32[512, 64, 384]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_8: "f32[512, 64, 384]" = torch.ops.aten.mul.Tensor(convert_element_type_7, sub_3);  convert_element_type_7 = sub_3 = None
        add: "f32[512, 64, 384]" = torch.ops.aten.add.Tensor(mul_8, 1);  mul_8 = None
        mul_9: "f32[512, 64, 384]" = torch.ops.aten.mul.Tensor(mul_7, add);  mul_7 = add = None
        convert_element_type_8: "bf16[512, 64, 384]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_7: "bf16[32768, 384]" = torch.ops.aten.view.default(convert_element_type_8, _shape_param_7);  convert_element_type_8 = _shape_param_7 = None
        mm_2: "bf16[32768, 192]" = torch.ops.aten.mm.default(view_7, arg8_1);  arg8_1 = None
        permute_3: "bf16[384, 32768]" = torch.ops.aten.permute.default(view_7, [1, 0])
        mm_3: "bf16[384, 192]" = torch.ops.aten.mm.default(permute_3, arg9_1);  permute_3 = arg9_1 = None
        sum_6: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True, dtype = torch.float32);  view_7 = None
        view_8: "f32[384]" = torch.ops.aten.view.default(sum_6, _shape_param_8);  sum_6 = _shape_param_8 = None
        convert_element_type_9: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        view_9: "bf16[512, 64, 192]" = torch.ops.aten.view.default(mm_2, _shape_param_9);  mm_2 = _shape_param_9 = None
        convert_element_type_10: "f32[512, 64, 192]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        convert_element_type_11: "f32[384, 192]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_12: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
        mul_10: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_10, arg10_1);  arg10_1 = None
        mul_11: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_10, 192)
        sum_7: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_10, [2], True)
        convert_element_type_13: "f32[512, 64, 192]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float32);  arg11_1 = None
        sub_4: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(convert_element_type_13, arg12_1);  convert_element_type_13 = arg12_1 = None
        mul_12: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_4, arg13_1);  sub_4 = None
        mul_13: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_10, mul_12);  mul_10 = None
        sum_8: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_13, [2], True);  mul_13 = None
        mul_14: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_12, sum_8);  sum_8 = None
        sub_5: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(mul_11, sum_7);  mul_11 = sum_7 = None
        sub_6: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(sub_5, mul_14);  sub_5 = mul_14 = None
        div_1: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(arg13_1, 192);  arg13_1 = None
        mul_15: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(div_1, sub_6);  div_1 = sub_6 = None
        mul_16: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_10, mul_12);  mul_12 = None
        sum_9: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_16, [0, 1]);  mul_16 = None
        sum_10: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_10, [0, 1]);  convert_element_type_10 = None
        convert_element_type_14: "bf16[512, 64, 192]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        add_1: "bf16[512, 64, 192]" = torch.ops.aten.add.Tensor(convert_element_type_2, convert_element_type_14);  convert_element_type_2 = convert_element_type_14 = None
        view_10: "bf16[32768, 192]" = torch.ops.aten.view.default(add_1, _shape_param_10);  _shape_param_10 = None
        mm_4: "bf16[32768, 192]" = torch.ops.aten.mm.default(view_10, arg14_1);  arg14_1 = None
        permute_4: "bf16[192, 32768]" = torch.ops.aten.permute.default(view_10, [1, 0])
        permute_5: "bf16[512, 64, 4, 48]" = torch.ops.aten.permute.default(arg15_1, [0, 2, 1, 3]);  arg15_1 = None
        view_11: "bf16[512, 64, 192]" = torch.ops.aten.view.default(permute_5, _shape_param_11);  permute_5 = _shape_param_11 = None
        view_12: "bf16[32768, 192]" = torch.ops.aten.view.default(view_11, _shape_param_12);  view_11 = _shape_param_12 = None
        mm_5: "bf16[192, 192]" = torch.ops.aten.mm.default(permute_4, view_12);  permute_4 = view_12 = None
        sum_11: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_10, [0], True, dtype = torch.float32);  view_10 = None
        view_13: "f32[192]" = torch.ops.aten.view.default(sum_11, _shape_param_13);  sum_11 = _shape_param_13 = None
        convert_element_type_15: "bf16[192]" = torch.ops.prims.convert_element_type.default(view_13, torch.bfloat16);  view_13 = None
        view_14: "bf16[512, 64, 192]" = torch.ops.aten.view.default(mm_4, _shape_param_14);  mm_4 = _shape_param_14 = None
        convert_element_type_16: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_17: "f32[192]" = torch.ops.prims.convert_element_type.default(convert_element_type_15, torch.float32);  convert_element_type_15 = None
        view_15: "bf16[512, 64, 4, 48]" = torch.ops.aten.view.default(view_14, _shape_param_15);  view_14 = _shape_param_15 = None
        permute_6: "bf16[512, 4, 64, 48]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        return (sum_3, sum_4, convert_element_type_4, convert_element_type_5, convert_element_type_11, convert_element_type_12, sum_9, sum_10, add_1, convert_element_type_16, convert_element_type_17, permute_6)



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

"""
Standalone repro captured via capture_hook.
Label: deit_tiny_patch16_224.fb_in1k
Pattern hash: 842eec146fcb
Shape hash: baa0b231
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
    def forward(self, arg0_1: "bf16[128, 3, 197, 64]", arg1_1: "bf16[128, 3, 197, 64]", arg2_1: "bf16[128, 3, 197, 64]", arg3_1: "bf16[576, 192]", arg4_1: "bf16[25216, 192]", arg5_1: "f32[192]", arg6_1: "f32[128, 197, 192]", arg7_1: "f32[1, 197, 192]", arg8_1: "f32[128, 197, 1]", arg9_1: "f32[128, 197, 1]", arg10_1: "f32[128, 197, 192]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        cat: "bf16[384, 3, 197, 64]" = torch.ops.aten.cat.default([arg0_1, arg1_1, arg2_1]);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[3, 128, 3, 197, 64]" = torch.ops.aten.view.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        permute: "bf16[128, 197, 3, 3, 64]" = torch.ops.aten.permute.default(view, [1, 3, 0, 2, 4]);  view = None
        clone: "bf16[128, 197, 3, 3, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[128, 197, 576]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[25216, 576]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        mm: "bf16[25216, 192]" = torch.ops.aten.mm.default(view_2, arg3_1);  arg3_1 = None
        permute_1: "bf16[576, 25216]" = torch.ops.aten.permute.default(view_2, [1, 0])
        mm_1: "bf16[576, 192]" = torch.ops.aten.mm.default(permute_1, arg4_1);  permute_1 = arg4_1 = None
        sum_1: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32);  view_2 = None
        view_3: "f32[576]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type: "bf16[576]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        view_4: "bf16[128, 197, 192]" = torch.ops.aten.view.default(mm, _shape_param_4);  mm = _shape_param_4 = None
        convert_element_type_1: "f32[128, 197, 192]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        convert_element_type_2: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_3: "f32[576]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        mul: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_1, arg5_1);  arg5_1 = None
        mul_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, 192)
        sum_2: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        add: "f32[128, 197, 192]" = torch.ops.aten.add.Tensor(arg6_1, arg7_1);  arg6_1 = arg7_1 = None
        sub: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(add, arg8_1);  add = arg8_1 = None
        mul_2: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(sub, arg9_1);  sub = None
        mul_3: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_3: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul_2, sum_3);  sum_3 = None
        sub_1: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(mul_1, sum_2);  mul_1 = sum_2 = None
        sub_2: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[128, 197, 1]" = torch.ops.aten.div.Tensor(arg9_1, 192);  arg9_1 = None
        mul_5: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_2);  mul_2 = None
        sum_4: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_5: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [0, 1]);  convert_element_type_1 = None
        add_1: "f32[128, 197, 192]" = torch.ops.aten.add.Tensor(arg10_1, mul_5);  arg10_1 = mul_5 = None
        sum_6: "f32[1, 197, 192]" = torch.ops.aten.sum.dim_IntList(add_1, [0], True, dtype = torch.float32)
        slice_1: "f32[128, 1, 192]" = torch.ops.aten.slice.Tensor(add_1, 1, 0, 1)
        slice_2: "f32[128, 196, 192]" = torch.ops.aten.slice.Tensor(add_1, 1, 1, 197);  add_1 = None
        convert_element_type_4: "bf16[128, 196, 192]" = torch.ops.prims.convert_element_type.default(slice_2, torch.bfloat16);  slice_2 = None
        sum_7: "f32[1, 1, 192]" = torch.ops.aten.sum.dim_IntList(slice_1, [0], True, dtype = torch.float32);  slice_1 = None
        permute_2: "bf16[128, 192, 196]" = torch.ops.aten.permute.default(convert_element_type_4, [0, 2, 1]);  convert_element_type_4 = None
        view_5: "bf16[128, 192, 14, 14]" = torch.ops.aten.view.default(permute_2, _shape_param_5);  permute_2 = _shape_param_5 = None
        sum_8: "bf16[192]" = torch.ops.aten.sum.dim_IntList(view_5, [0, 2, 3])
        convert_element_type_5: "f32[192]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        return (convert_element_type_2, convert_element_type_3, sum_4, sum_5, sum_6, sum_7, view_5, convert_element_type_5)



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

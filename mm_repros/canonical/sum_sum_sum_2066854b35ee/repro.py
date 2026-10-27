"""
Standalone repro captured via capture_hook.
Label: mobilevit_s
Pattern hash: 2066854b35ee
Shape hash: 6231a92e
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
    def forward(self, arg0_1: "bf16[512, 4, 64, 48]", arg1_1: "bf16[512, 4, 64, 48]", arg2_1: "bf16[512, 4, 64, 48]", arg3_1: "bf16[576, 192]", arg4_1: "bf16[32768, 192]", arg5_1: "f32[192]", arg6_1: "bf16[512, 64, 192]", arg7_1: "f32[512, 64, 1]", arg8_1: "f32[512, 64, 1]", arg9_1: "bf16[512, 64, 192]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        cat: "bf16[1536, 4, 64, 48]" = torch.ops.aten.cat.default([arg0_1, arg1_1, arg2_1]);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[3, 512, 4, 64, 48]" = torch.ops.aten.view.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        permute: "bf16[512, 64, 3, 4, 48]" = torch.ops.aten.permute.default(view, [1, 3, 0, 2, 4]);  view = None
        clone: "bf16[512, 64, 3, 4, 48]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[512, 64, 576]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[32768, 576]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        mm: "bf16[32768, 192]" = torch.ops.aten.mm.default(view_2, arg3_1);  arg3_1 = None
        permute_1: "bf16[576, 32768]" = torch.ops.aten.permute.default(view_2, [1, 0])
        mm_1: "bf16[576, 192]" = torch.ops.aten.mm.default(permute_1, arg4_1);  permute_1 = arg4_1 = None
        sum_1: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32);  view_2 = None
        view_3: "f32[576]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type: "bf16[576]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        view_4: "bf16[512, 64, 192]" = torch.ops.aten.view.default(mm, _shape_param_4);  mm = _shape_param_4 = None
        convert_element_type_1: "f32[512, 64, 192]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        convert_element_type_2: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_3: "f32[576]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        mul: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_1, arg5_1);  arg5_1 = None
        mul_1: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul, 192)
        sum_2: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        convert_element_type_4: "f32[512, 64, 192]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        sub: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(convert_element_type_4, arg7_1);  convert_element_type_4 = arg7_1 = None
        mul_2: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub, arg8_1);  sub = None
        mul_3: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_3: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_2, sum_3);  sum_3 = None
        sub_1: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(mul_1, sum_2);  mul_1 = sum_2 = None
        sub_2: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(arg8_1, 192);  arg8_1 = None
        mul_5: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_2);  mul_2 = None
        sum_4: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_5: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [0, 1]);  convert_element_type_1 = None
        convert_element_type_5: "bf16[512, 64, 192]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        add: "bf16[512, 64, 192]" = torch.ops.aten.add.Tensor(arg9_1, convert_element_type_5);  arg9_1 = convert_element_type_5 = None
        view_5: "bf16[128, 4, 64, 192]" = torch.ops.aten.view.default(add, _shape_param_5);  add = _shape_param_5 = None
        permute_2: "bf16[128, 192, 64, 4]" = torch.ops.aten.permute.default(view_5, [0, 3, 2, 1]);  view_5 = None
        clone_1: "bf16[128, 192, 64, 4]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_6: "bf16[196608, 8, 2, 2]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        permute_3: "bf16[196608, 2, 8, 2]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_2: "bf16[196608, 2, 8, 2]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_7: "bf16[128, 192, 16, 16]" = torch.ops.aten.view.default(clone_2, _shape_param_7);  clone_2 = _shape_param_7 = None
        return (convert_element_type_2, convert_element_type_3, sum_4, sum_5, view_7)



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

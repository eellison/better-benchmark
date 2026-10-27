"""
Standalone repro captured via capture_hook.
Label: BlenderbotForCausalLM
Pattern hash: f6432c0161c8
Shape hash: 7d89d30d
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
    def forward(self, arg0_1: "bf16[32, 32, 128, 80]", arg1_1: "bf16[32, 32, 128, 80]", arg2_1: "bf16[2560, 2560]", arg3_1: "bf16[4096, 2560]", arg4_1: "bf16[2560, 2560]", arg5_1: "bf16[32, 32, 128, 80]", arg6_1: "bf16[2560, 2560]", arg7_1: "f32[2560]", arg8_1: "f32[32, 128, 2560]", arg9_1: "f32[32, 128, 1]", arg10_1: "f32[32, 128, 1]", arg11_1: "f32[32, 128, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        permute: "bf16[32, 128, 32, 80]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        permute_1: "bf16[32, 128, 32, 80]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        view_1: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(permute_1, _shape_param_1);  permute_1 = _shape_param_1 = None
        view_2: "bf16[4096, 2560]" = torch.ops.aten.view.default(view, _shape_param_2);  view = _shape_param_2 = None
        mm: "bf16[4096, 2560]" = torch.ops.aten.mm.default(view_2, arg2_1);  arg2_1 = None
        permute_2: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        mm_1: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_2, arg3_1);  permute_2 = None
        sum_1: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32);  view_2 = None
        view_3: "f32[2560]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        view_4: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(mm, _shape_param_4);  mm = _shape_param_4 = None
        convert_element_type_1: "f32[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        convert_element_type_2: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_3: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_5: "bf16[4096, 2560]" = torch.ops.aten.view.default(view_1, _shape_param_5);  view_1 = _shape_param_5 = None
        mm_2: "bf16[4096, 2560]" = torch.ops.aten.mm.default(view_5, arg4_1);  arg4_1 = None
        permute_3: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view_5, [1, 0])
        mm_3: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_3, arg3_1);  permute_3 = None
        sum_2: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32);  view_5 = None
        view_6: "f32[2560]" = torch.ops.aten.view.default(sum_2, _shape_param_6);  sum_2 = _shape_param_6 = None
        convert_element_type_4: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        view_7: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(mm_2, _shape_param_7);  mm_2 = _shape_param_7 = None
        convert_element_type_5: "f32[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        add: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_5);  convert_element_type_1 = convert_element_type_5 = None
        convert_element_type_6: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_7: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        permute_4: "bf16[32, 128, 32, 80]" = torch.ops.aten.permute.default(arg5_1, [0, 2, 1, 3]);  arg5_1 = None
        view_8: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(permute_4, _shape_param_8);  permute_4 = _shape_param_8 = None
        view_9: "bf16[4096, 2560]" = torch.ops.aten.view.default(view_8, _shape_param_9);  view_8 = _shape_param_9 = None
        mm_4: "bf16[4096, 2560]" = torch.ops.aten.mm.default(view_9, arg6_1);  arg6_1 = None
        permute_5: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view_9, [1, 0])
        mm_5: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_5, arg3_1);  permute_5 = arg3_1 = None
        sum_3: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_9, [0], True, dtype = torch.float32);  view_9 = None
        view_10: "f32[2560]" = torch.ops.aten.view.default(sum_3, _shape_param_10);  sum_3 = _shape_param_10 = None
        convert_element_type_8: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        view_11: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(mm_4, _shape_param_11);  mm_4 = _shape_param_11 = None
        convert_element_type_9: "f32[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        add_1: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(add, convert_element_type_9);  add = convert_element_type_9 = None
        convert_element_type_10: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_11: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        mul: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(add_1, arg7_1);  arg7_1 = None
        mul_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, 2560)
        sum_4: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        sub: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(arg8_1, arg9_1);  arg8_1 = arg9_1 = None
        mul_2: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, arg10_1);  sub = None
        mul_3: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_5: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_2, sum_5);  sum_5 = None
        sub_1: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_1, sum_4);  mul_1 = sum_4 = None
        sub_2: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(arg10_1, 2560);  arg10_1 = None
        mul_5: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(add_1, mul_2);  mul_2 = None
        sum_6: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_7: "f32[2560]" = torch.ops.aten.sum.dim_IntList(add_1, [0, 1]);  add_1 = None
        add_2: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(arg11_1, mul_5);  arg11_1 = mul_5 = None
        return (convert_element_type_2, convert_element_type_3, convert_element_type_6, convert_element_type_7, convert_element_type_10, convert_element_type_11, sum_6, sum_7, add_2)



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

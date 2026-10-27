"""
Standalone repro captured via capture_hook.
Label: OPTForCausalLM
Pattern hash: 3dcc86c10d94
Shape hash: 9580f737
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
    def forward(self, arg0_1: "f32[4, 2048, 768]", arg1_1: "b8[8192, 768]", arg2_1: "bf16[768, 3072]", arg3_1: "bf16[8192, 3072]", arg4_1: "bf16[3072, 768]", arg5_1: "bf16[8192, 768]", arg6_1: "f32[768]", arg7_1: "bf16[8192, 768]", arg8_1: "b8[4, 2048, 768]", arg9_1: "f32[4, 2048, 768]", arg10_1: "f32[8192, 1]", arg11_1: "f32[8192, 1]", arg12_1: "bf16[768, 768]", arg13_1: "bf16[4, 12, 2048, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        view: "f32[8192, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[8192, 768]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16)
        convert_element_type_1: "bf16[8192, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        mm: "bf16[8192, 3072]" = torch.ops.aten.mm.default(mul_1, arg2_1);  arg2_1 = None
        permute: "bf16[768, 8192]" = torch.ops.aten.permute.default(mul_1, [1, 0])
        mm_1: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute, arg3_1);  permute = None
        sum_1: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1, [0], True, dtype = torch.float32);  mul_1 = None
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        convert_element_type_2: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_3: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_4: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        le: "b8[8192, 3072]" = torch.ops.aten.le.Scalar(arg3_1, 0);  arg3_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[8192, 3072]" = torch.ops.aten.where.self(le, full, mm);  le = full = mm = None
        mm_2: "bf16[8192, 768]" = torch.ops.aten.mm.default(where, arg4_1);  arg4_1 = None
        permute_1: "bf16[3072, 8192]" = torch.ops.aten.permute.default(where, [1, 0])
        mm_3: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_1, arg5_1);  permute_1 = arg5_1 = None
        sum_2: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(where, [0], True, dtype = torch.float32);  where = None
        view_2: "f32[3072]" = torch.ops.aten.view.default(sum_2, _shape_param_2);  sum_2 = _shape_param_2 = None
        convert_element_type_5: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_6: "f32[8192, 768]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_7: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_8: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        mul_2: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, arg6_1);  arg6_1 = None
        mul_3: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_3: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [1], True)
        view_3: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(arg7_1, _shape_param_3);  arg7_1 = _shape_param_3 = None
        mul_4: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(arg8_1, view_3);  view_3 = None
        mul_5: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None
        add: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(arg9_1, mul_5);  arg9_1 = mul_5 = None
        view_4: "f32[8192, 768]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        sub: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(view_4, arg10_1);  view_4 = arg10_1 = None
        mul_6: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(sub, arg11_1);  sub = None
        mul_7: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_2, mul_6);  mul_2 = None
        sum_4: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [1], True);  mul_7 = None
        mul_8: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_6, sum_4);  sum_4 = None
        sub_1: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_2: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(sub_1, mul_8);  sub_1 = mul_8 = None
        div: "f32[8192, 1]" = torch.ops.aten.div.Tensor(arg11_1, 768);  arg11_1 = None
        mul_9: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_10: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_6);  mul_6 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_10, [0]);  mul_10 = None
        sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0]);  convert_element_type_6 = None
        add_1: "f32[8192, 768]" = torch.ops.aten.add.Tensor(view, mul_9);  view = mul_9 = None
        view_5: "f32[4, 2048, 768]" = torch.ops.aten.view.default(add_1, _shape_param_5);  add_1 = _shape_param_5 = None
        convert_element_type_9: "bf16[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16)
        convert_element_type_10: "bf16[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        mul_11: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_12: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_11);  convert_element_type_9 = mul_11 = None
        view_6: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_12, _shape_param_6);  mul_12 = _shape_param_6 = None
        mm_4: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_6, arg12_1);  arg12_1 = None
        permute_2: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_6, [1, 0])
        permute_3: "bf16[4, 2048, 12, 64]" = torch.ops.aten.permute.default(arg13_1, [0, 2, 1, 3]);  arg13_1 = None
        view_7: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(permute_3, _shape_param_7);  permute_3 = _shape_param_7 = None
        view_8: "bf16[8192, 768]" = torch.ops.aten.view.default(view_7, _shape_param_8);  view_7 = _shape_param_8 = None
        mm_5: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_2, view_8);  permute_2 = view_8 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_6, [0], True, dtype = torch.float32);  view_6 = None
        view_9: "f32[768]" = torch.ops.aten.view.default(sum_7, _shape_param_9);  sum_7 = _shape_param_9 = None
        convert_element_type_11: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(mm_4, _shape_param_10);  mm_4 = _shape_param_10 = None
        convert_element_type_12: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_13: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        view_11: "bf16[4, 2048, 12, 64]" = torch.ops.aten.view.default(view_10, _shape_param_11);  view_10 = _shape_param_11 = None
        permute_4: "bf16[4, 12, 2048, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_7, convert_element_type_8, sum_5, sum_6, view_5, convert_element_type_12, convert_element_type_13, permute_4)



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

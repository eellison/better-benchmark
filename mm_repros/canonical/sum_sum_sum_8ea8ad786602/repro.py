"""
Standalone repro captured via capture_hook.
Label: GPT2ForSequenceClassification
Pattern hash: 8ea8ad786602
Shape hash: aacca4e3
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
    def forward(self, arg0_1: "bf16[8, 12, 1024, 64]", arg1_1: "bf16[8, 12, 1024, 64]", arg2_1: "bf16[8, 12, 1024, 64]", arg3_1: "bf16[2304, 768]", arg4_1: "bf16[768, 8192]", arg5_1: "f32[768]", arg6_1: "f32[8, 1024, 768]", arg7_1: "f32[1, 1024, 768]", arg8_1: "b8[8, 1024, 768]", arg9_1: "f32[8, 1024, 1]", arg10_1: "f32[8, 1024, 1]", arg11_1: "f32[8, 1024, 768]", arg12_1: "i64[1, 1024]", arg13_1: "i64[8, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        permute_1: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        view_1: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(permute_1, _shape_param_1);  permute_1 = _shape_param_1 = None
        permute_2: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        view_2: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(permute_2, _shape_param_2);  permute_2 = _shape_param_2 = None
        cat: "bf16[8, 1024, 2304]" = torch.ops.aten.cat.default([view, view_2, view_1], 2);  view = view_2 = view_1 = None
        view_3: "bf16[8192, 2304]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_3, arg3_1);  arg3_1 = None
        mm_1: "bf16[768, 2304]" = torch.ops.aten.mm.default(arg4_1, view_3);  arg4_1 = None
        sum_1: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32);  view_3 = None
        view_4: "f32[2304]" = torch.ops.aten.view.default(sum_1, _shape_param_4);  sum_1 = _shape_param_4 = None
        convert_element_type: "bf16[2304]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_1: "f32[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_2: "f32[8192, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_3: "f32[768, 2304]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        view_5: "f32[8, 1024, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_5);  convert_element_type_2 = _shape_param_5 = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_5, arg5_1);  arg5_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_2: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        add: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(arg6_1, arg7_1);  arg6_1 = arg7_1 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg8_1, add);  add = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_3, arg9_1);  mul_3 = arg9_1 = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, arg10_1);  sub = None
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, mul_4);  mul = None
        sum_3: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [2], True);  mul_5 = None
        mul_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_3);  sum_3 = None
        sub_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_2);  mul_1 = sum_2 = None
        sub_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_1, mul_6);  sub_1 = mul_6 = None
        div: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(arg10_1, 768);  arg10_1 = None
        mul_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_5, mul_4);  mul_4 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_8, [0, 1]);  mul_8 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_5, [0, 1]);  view_5 = None
        add_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(arg11_1, mul_7);  arg11_1 = mul_7 = None
        convert_element_type_4: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        mul_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_1, mul_9);  add_1 = mul_9 = None
        sum_6: "f32[1, 1024, 768]" = torch.ops.aten.sum.dim_IntList(mul_10, [0], True, dtype = torch.float32)
        full: "b8[1, 1024, 1]" = torch.ops.aten.full.default(_shape_param_6, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        full_1: "f32[1024, 768]" = torch.ops.aten.full.default(_shape_param_7, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_7 = None
        _unsafe_masked_index_put_accumulate: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, full, [arg12_1], sum_6);  full_1 = full = arg12_1 = sum_6 = None
        ge: "b8[8, 1024]" = torch.ops.aten.ge.Scalar(arg13_1, 0)
        lt: "b8[8, 1024]" = torch.ops.aten.lt.Scalar(arg13_1, 50257)
        bitwise_and: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg13_1, -1)
        bitwise_and_1: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_2: "f32[50257, 768]" = torch.ops.aten.full.default(_shape_param_8, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_8 = None
        _unsafe_masked_index_put_accumulate_1: "f32[50257, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze, [arg13_1], mul_10);  full_2 = unsqueeze = arg13_1 = mul_10 = None
        return (convert_element_type_1, convert_element_type_3, sum_4, sum_5, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1)



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

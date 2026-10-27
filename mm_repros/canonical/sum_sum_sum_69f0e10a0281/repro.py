"""
Standalone repro captured via capture_hook.
Label: YituTechConvBert
Pattern hash: 69f0e10a0281
Shape hash: 63b160de
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
    def forward(self, arg0_1: "bf16[16384, 30522]", arg1_1: "bf16[16384, 768]", arg2_1: "bf16[32, 6, 512, 64]", arg3_1: "bf16[32, 6, 512, 64]", arg4_1: "bf16[32, 6, 512, 64]", arg5_1: "f32[98304, 9, 1]", arg6_1: "bf16[98304, 64, 1]", arg7_1: "f32[32, 384, 520, 1]", arg8_1: "i64[9, 512, 1, 1]", arg9_1: "i64[1, 1]", arg10_1: "bf16[384, 768]", arg11_1: "bf16[16384, 768]", arg12_1: "f32[32, 512, 768]", arg13_1: "bf16[32, 384, 512]", arg14_1: "f32[384, 1]", arg15_1: "bf16[32, 512, 384]", arg16_1: "bf16[384, 768]", arg17_1: "bf16[32, 768, 512]", arg18_1: "bf16[384, 768]", arg19_1: "bf16[384, 768]", arg20_1: "b8[32, 512, 768]", arg21_1: "f32[768]", arg22_1: "f32[32, 512, 768]", arg23_1: "f32[32, 512, 1]", arg24_1: "i64[1, 512]", arg25_1: "i64[1, 512]", arg26_1: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23):
        # No stacktrace found for following nodes
        permute: "bf16[30522, 16384]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[30522, 768]" = torch.ops.aten.mm.default(permute, arg1_1);  permute = arg1_1 = None
        convert_element_type: "f32[30522, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        permute_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        permute_2: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1, 3]);  arg3_1 = None
        permute_3: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg4_1, [0, 2, 1, 3]);  arg4_1 = None
        convert_element_type_1: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        expand: "bf16[98304, 9, 1]" = torch.ops.aten.expand.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        permute_4: "bf16[98304, 1, 9]" = torch.ops.aten.permute.default(expand, [0, 2, 1]);  expand = None
        convert_element_type_2: "f32[98304, 64, 1]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        convert_element_type_3: "f32[98304, 1, 9]" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        mul: "f32[98304, 64, 9]" = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type_3);  convert_element_type_2 = convert_element_type_3 = None
        convert_element_type_4: "bf16[98304, 64, 9]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        view: "bf16[32, 512, 384, 9]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_1);  convert_element_type_4 = _shape_param_1 = None
        view_1: "bf16[32, 512, 3456]" = torch.ops.aten.view.default(view, _shape_param_2);  view = _shape_param_2 = None
        permute_5: "bf16[32, 3456, 512]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        convert_element_type_5: "f32[32, 3456, 512]" = torch.ops.prims.convert_element_type.default(permute_5, torch.float32);  permute_5 = None
        view_2: "f32[32, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        permute_6: "f32[32, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 2, 4, 3, 5]);  view_2 = None
        index_put: "f32[32, 384, 520, 1]" = torch.ops.aten.index_put.default(arg7_1, [None, None, arg8_1, arg9_1], permute_6, True);  arg7_1 = arg8_1 = arg9_1 = permute_6 = None
        constant_pad_nd: "f32[32, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(index_put, _shape_param_4, 0.0);  index_put = _shape_param_4 = None
        convert_element_type_6: "bf16[32, 384, 512, 1]" = torch.ops.prims.convert_element_type.default(constant_pad_nd, torch.bfloat16);  constant_pad_nd = None
        squeeze: "bf16[32, 384, 512]" = torch.ops.aten.squeeze.dim(convert_element_type_6, -1);  convert_element_type_6 = None
        permute_7: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(squeeze, [0, 2, 1]);  squeeze = None
        clone: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_3: "bf16[16384, 384]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        mm_1: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_3, arg10_1);  arg10_1 = None
        permute_8: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_3, [1, 0])
        mm_2: "bf16[384, 768]" = torch.ops.aten.mm.default(permute_8, arg11_1);  permute_8 = None
        sum_1: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32);  view_3 = None
        view_4: "f32[384]" = torch.ops.aten.view.default(sum_1, _shape_param_6);  sum_1 = _shape_param_6 = None
        convert_element_type_7: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        view_5: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_7);  mm_1 = _shape_param_7 = None
        convert_element_type_8: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg12_1, convert_element_type_8);  arg12_1 = convert_element_type_8 = None
        convert_element_type_9: "f32[384, 768]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_10: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        add_1: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(arg13_1, arg14_1);  arg13_1 = arg14_1 = None
        convert_element_type_11: "bf16[32, 384, 512]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        permute_9: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(convert_element_type_11, [0, 2, 1]);  convert_element_type_11 = None
        mul_1: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(arg15_1, permute_9);  arg15_1 = permute_9 = None
        clone_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_6: "bf16[32, 512, 384]" = torch.ops.aten.view.default(clone_1, _shape_param_8);  clone_1 = _shape_param_8 = None
        clone_2: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        _unsafe_view: "bf16[32, 512, 384]" = torch.ops.aten._unsafe_view.default(clone_2, _shape_param_9);  clone_2 = _shape_param_9 = None
        clone_3: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_7: "bf16[32, 512, 384]" = torch.ops.aten.view.default(clone_3, _shape_param_10);  clone_3 = _shape_param_10 = None
        add_2: "bf16[32, 512, 384]" = torch.ops.aten.add.Tensor(mul_1, view_7);  mul_1 = view_7 = None
        view_8: "bf16[16384, 384]" = torch.ops.aten.view.default(add_2, _shape_param_11);  add_2 = _shape_param_11 = None
        mm_3: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_8, arg16_1);  arg16_1 = None
        permute_10: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_4: "bf16[384, 768]" = torch.ops.aten.mm.default(permute_10, arg11_1);  permute_10 = None
        sum_2: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[384]" = torch.ops.aten.view.default(sum_2, _shape_param_12);  sum_2 = _shape_param_12 = None
        convert_element_type_12: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_3, _shape_param_13);  mm_3 = _shape_param_13 = None
        convert_element_type_13: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_13);  add = convert_element_type_13 = None
        convert_element_type_14: "f32[384, 768]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_15: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        convert_element_type_16: "f32[32, 768, 512]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        permute_11: "f32[32, 512, 768]" = torch.ops.aten.permute.default(convert_element_type_16, [0, 2, 1]);  convert_element_type_16 = None
        add_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_3, permute_11);  add_3 = permute_11 = None
        view_11: "bf16[16384, 384]" = torch.ops.aten.view.default(view_6, _shape_param_14);  view_6 = _shape_param_14 = None
        mm_5: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_11, arg18_1);  arg18_1 = None
        permute_12: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_6: "bf16[384, 768]" = torch.ops.aten.mm.default(permute_12, arg11_1);  permute_12 = None
        sum_3: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_12: "f32[384]" = torch.ops.aten.view.default(sum_3, _shape_param_15);  sum_3 = _shape_param_15 = None
        convert_element_type_17: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        view_13: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_5, _shape_param_16);  mm_5 = _shape_param_16 = None
        convert_element_type_18: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_18);  add_4 = convert_element_type_18 = None
        convert_element_type_19: "f32[384, 768]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_20: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_17, torch.float32);  convert_element_type_17 = None
        clone_4: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(_unsafe_view, memory_format = torch.contiguous_format);  _unsafe_view = None
        view_14: "bf16[16384, 384]" = torch.ops.aten.view.default(clone_4, _shape_param_17);  clone_4 = _shape_param_17 = None
        mm_7: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_14, arg19_1);  arg19_1 = None
        permute_13: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_14, [1, 0])
        mm_8: "bf16[384, 768]" = torch.ops.aten.mm.default(permute_13, arg11_1);  permute_13 = arg11_1 = None
        sum_4: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_14, [0], True, dtype = torch.float32);  view_14 = None
        view_15: "f32[384]" = torch.ops.aten.view.default(sum_4, _shape_param_18);  sum_4 = _shape_param_18 = None
        convert_element_type_21: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_15, torch.bfloat16);  view_15 = None
        view_16: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_7, _shape_param_19);  mm_7 = _shape_param_19 = None
        convert_element_type_22: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_16, torch.float32);  view_16 = None
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_22);  add_5 = convert_element_type_22 = None
        convert_element_type_23: "f32[384, 768]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        convert_element_type_24: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_21, torch.float32);  convert_element_type_21 = None
        convert_element_type_25: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.float32);  arg20_1 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_6, mul_2);  add_6 = mul_2 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg21_1);  arg21_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, 768)
        sum_5: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True)
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg22_1);  mul_4 = None
        sum_6: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_6, [2], True);  mul_6 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg22_1, sum_6);  sum_6 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_5, sum_5);  mul_5 = sum_5 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_7);  sub = mul_7 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg23_1, sub_1);  arg23_1 = sub_1 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg22_1);  arg22_1 = None
        sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_9, [0, 1]);  mul_9 = None
        sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1]);  mul_3 = None
        sum_9: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_8, [0], True, dtype = torch.float32)
        expand_1: "i64[32, 512]" = torch.ops.aten.expand.default(arg24_1, _shape_param_20);  arg24_1 = _shape_param_20 = None
        ge: "b8[32, 512]" = torch.ops.aten.ge.Scalar(expand_1, 0)
        lt: "b8[32, 512]" = torch.ops.aten.lt.Scalar(expand_1, 2)
        bitwise_and: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[32, 512]" = torch.ops.aten.ne.Scalar(expand_1, -1)
        bitwise_and_1: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[2, 768]" = torch.ops.aten.full.default(_shape_param_21, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_21 = None
        _unsafe_masked_index_put_accumulate: "f32[2, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [expand_1], mul_8);  full = unsqueeze = expand_1 = None
        ge_1: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg25_1, 0)
        lt_1: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg25_1, 512)
        bitwise_and_2: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg25_1, -1)
        bitwise_and_3: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_1: "f32[512, 768]" = torch.ops.aten.full.default(_shape_param_22, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_22 = None
        _unsafe_masked_index_put_accumulate_1: "f32[512, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, unsqueeze_1, [arg25_1], sum_9);  full_1 = unsqueeze_1 = arg25_1 = sum_9 = None
        ge_2: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg26_1, 0)
        lt_2: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg26_1, 30522)
        bitwise_and_4: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_2: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg26_1, 0)
        bitwise_and_5: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_2);  bitwise_and_4 = ne_2 = None
        unsqueeze_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_2: "f32[30522, 768]" = torch.ops.aten.full.default(_shape_param_23, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_23 = None
        _unsafe_masked_index_put_accumulate_2: "f32[30522, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze_2, [arg26_1], mul_8);  full_2 = unsqueeze_2 = arg26_1 = mul_8 = None
        add_7: "f32[30522, 768]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate_2);  convert_element_type = _unsafe_masked_index_put_accumulate_2 = None
        return (convert_element_type_9, convert_element_type_10, convert_element_type_14, convert_element_type_15, convert_element_type_19, convert_element_type_20, convert_element_type_23, convert_element_type_24, sum_7, sum_8, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, add_7)



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

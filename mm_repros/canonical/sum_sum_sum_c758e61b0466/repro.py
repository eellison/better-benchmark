"""
Standalone repro captured via capture_hook.
Label: MegatronBertForCausalLM
Pattern hash: c758e61b0466
Shape hash: 861fd349
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
    def forward(self, arg0_1: "bf16[8192, 29056]", arg1_1: "bf16[8192, 1024]", arg2_1: "bf16[16, 16, 512, 64]", arg3_1: "bf16[16, 16, 512, 64]", arg4_1: "bf16[16, 16, 512, 64]", arg5_1: "bf16[1024, 1024]", arg6_1: "bf16[8192, 1024]", arg7_1: "bf16[1024, 1024]", arg8_1: "bf16[1024, 1024]", arg9_1: "f32[1024]", arg10_1: "f32[16, 512, 1024]", arg11_1: "f32[16, 512, 1]", arg12_1: "f32[16, 512, 1024]", arg13_1: "b8[16, 512, 1024]", arg14_1: "i64[1, 512]", arg15_1: "i64[16, 512]", arg16_1: "i64[16, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15):
        # No stacktrace found for following nodes
        permute: "bf16[29056, 8192]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[29056, 1024]" = torch.ops.aten.mm.default(permute, arg1_1);  permute = arg1_1 = None
        convert_element_type: "f32[29056, 1024]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        permute_1: "bf16[16, 512, 16, 64]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        permute_2: "bf16[16, 512, 16, 64]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1, 3]);  arg3_1 = None
        permute_3: "bf16[16, 512, 16, 64]" = torch.ops.aten.permute.default(arg4_1, [0, 2, 1, 3]);  arg4_1 = None
        clone: "bf16[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        mm_1: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_1, arg5_1);  arg5_1 = None
        permute_4: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0])
        mm_2: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_4, arg6_1);  permute_4 = None
        sum_1: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32);  view_1 = None
        view_2: "f32[1024]" = torch.ops.aten.view.default(sum_1, _shape_param_2);  sum_1 = _shape_param_2 = None
        convert_element_type_1: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        view_3: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm_1, _shape_param_3);  mm_1 = _shape_param_3 = None
        convert_element_type_2: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        convert_element_type_3: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_4: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        view_4: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(permute_2, _shape_param_4);  permute_2 = _shape_param_4 = None
        clone_1: "bf16[16, 512, 1024]" = torch.ops.aten.clone.default(view_4, memory_format = torch.contiguous_format);  view_4 = None
        view_5: "bf16[8192, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        mm_3: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_5, arg7_1);  arg7_1 = None
        permute_5: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_5, [1, 0])
        mm_4: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_5, arg6_1);  permute_5 = None
        sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32);  view_5 = None
        view_6: "f32[1024]" = torch.ops.aten.view.default(sum_2, _shape_param_6);  sum_2 = _shape_param_6 = None
        convert_element_type_5: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        view_7: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm_3, _shape_param_7);  mm_3 = _shape_param_7 = None
        convert_element_type_6: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        add: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_2, convert_element_type_6);  convert_element_type_2 = convert_element_type_6 = None
        convert_element_type_7: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_8: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        clone_2: "bf16[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_8: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_8);  clone_2 = _shape_param_8 = None
        view_9: "bf16[8192, 1024]" = torch.ops.aten.view.default(view_8, _shape_param_9);  view_8 = _shape_param_9 = None
        mm_5: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_9, arg8_1);  arg8_1 = None
        permute_6: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_9, [1, 0])
        mm_6: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_6, arg6_1);  permute_6 = arg6_1 = None
        sum_3: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_9, [0], True, dtype = torch.float32);  view_9 = None
        view_10: "f32[1024]" = torch.ops.aten.view.default(sum_3, _shape_param_10);  sum_3 = _shape_param_10 = None
        convert_element_type_9: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        view_11: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm_5, _shape_param_11);  mm_5 = _shape_param_11 = None
        convert_element_type_10: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        add_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add, convert_element_type_10);  add = convert_element_type_10 = None
        convert_element_type_11: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_12: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
        mul: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_1, arg9_1);  arg9_1 = None
        mul_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sum_4: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, arg10_1);  mul = None
        sum_5: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg10_1, sum_5);  sum_5 = None
        sub: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1, sum_4);  mul_1 = sum_4 = None
        sub_1: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg11_1, sub_1);  arg11_1 = sub_1 = None
        mul_5: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_1, arg10_1);  arg10_1 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_1, [0, 1]);  add_1 = None
        add_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(arg12_1, mul_4);  arg12_1 = mul_4 = None
        convert_element_type_13: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        mul_6: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_7: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_2, mul_6);  add_2 = mul_6 = None
        sum_8: "f32[1, 512, 1024]" = torch.ops.aten.sum.dim_IntList(mul_7, [0], True, dtype = torch.float32)
        ge: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg14_1, 0)
        lt: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg14_1, 512)
        bitwise_and: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg14_1, -1)
        bitwise_and_1: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[512, 1024]" = torch.ops.aten.full.default(_shape_param_12, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_12 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg14_1], sum_8);  full = unsqueeze = arg14_1 = sum_8 = None
        full_1: "b8[16, 512, 1]" = torch.ops.aten.full.default(_shape_param_13, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_13 = None
        full_2: "f32[2, 1024]" = torch.ops.aten.full.default(_shape_param_14, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_14 = None
        _unsafe_masked_index_put_accumulate_1: "f32[2, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, full_1, [arg15_1], mul_7);  full_2 = full_1 = arg15_1 = None
        ge_1: "b8[16, 512]" = torch.ops.aten.ge.Scalar(arg16_1, 0)
        lt_1: "b8[16, 512]" = torch.ops.aten.lt.Scalar(arg16_1, 29056)
        bitwise_and_2: "b8[16, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[16, 512]" = torch.ops.aten.ne.Scalar(arg16_1, 0)
        bitwise_and_3: "b8[16, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[16, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_3: "f32[29056, 1024]" = torch.ops.aten.full.default(_shape_param_15, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_15 = None
        _unsafe_masked_index_put_accumulate_2: "f32[29056, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_3, unsqueeze_1, [arg16_1], mul_7);  full_3 = unsqueeze_1 = arg16_1 = mul_7 = None
        add_3: "f32[29056, 1024]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate_2);  convert_element_type = _unsafe_masked_index_put_accumulate_2 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_7, convert_element_type_8, convert_element_type_11, convert_element_type_12, sum_6, sum_7, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, add_3)



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

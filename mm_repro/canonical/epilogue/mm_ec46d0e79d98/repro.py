"""Curated MM repro. Pattern hash: ec46d0e79d98."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[30522, 16384]", arg1_1: "bf16[16384, 768]", arg2_1: "bf16[16384, 768]", arg3_1: "f32[32, 512, 768]", arg4_1: "bf16[16384, 768]", arg5_1: "bf16[16384, 768]", arg6_1: "b8[32, 512, 768]", arg7_1: "f32[768]", arg8_1: "f32[32, 512, 768]", arg9_1: "f32[32, 512, 1]", arg10_1: "i64[32, 512]", arg11_1: "i64[32, 512]", arg12_1: "i64[32, 512]", arg13_1: "i64[32, 512]", arg14_1: "i64[32, 512]", arg15_1: "i64[32, 512]", arg16_1: "i64[32, 512]", arg17_1: "i64[1, 512]", arg18_1: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        mm: "bf16[30522, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[30522, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        convert_element_type_1: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg3_1, convert_element_type_1);  arg3_1 = convert_element_type_1 = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        convert_element_type_2: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        view_2: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg5_1, _shape_param_2);  arg5_1 = _shape_param_2 = None
        convert_element_type_3: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_3);  add_1 = convert_element_type_3 = None
        convert_element_type_4: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_2, mul);  add_2 = mul = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg7_1);  arg7_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg8_1);  mul_2 = None
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg8_1, sum_2);  sum_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_1);  mul_3 = sum_1 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_5);  sub = mul_5 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg9_1, sub_1);  arg9_1 = sub_1 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg8_1);  arg8_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1, [0, 1]);  mul_1 = None
        sum_5: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_6, [0], True, dtype = torch.float32)
        full: "b8[32, 512, 1]" = torch.ops.aten.full.default(_shape_param_3, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        full_1: "f32[2, 768]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        _unsafe_masked_index_put_accumulate: "f32[2, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, full, [arg10_1], mul_6);  full_1 = full = arg10_1 = None
        ge: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg11_1, 0)
        lt: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg11_1, 1024)
        bitwise_and: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg11_1, -1)
        bitwise_and_1: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_2: "f32[1024, 768]" = torch.ops.aten.full.default(_shape_param_5, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        _unsafe_masked_index_put_accumulate_1: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze, [arg11_1], mul_6);  unsqueeze = arg11_1 = None
        ge_1: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg12_1, 0)
        lt_1: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg12_1, 1024)
        bitwise_and_2: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg12_1, -1)
        bitwise_and_3: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        _unsafe_masked_index_put_accumulate_2: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze_1, [arg12_1], mul_6);  unsqueeze_1 = arg12_1 = None
        ge_2: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg13_1, 0)
        lt_2: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg13_1, 1024)
        bitwise_and_4: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_2: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg13_1, -1)
        bitwise_and_5: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_2);  bitwise_and_4 = ne_2 = None
        unsqueeze_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        _unsafe_masked_index_put_accumulate_3: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze_2, [arg13_1], mul_6);  unsqueeze_2 = arg13_1 = None
        ge_3: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg14_1, 0)
        lt_3: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg14_1, 1024)
        bitwise_and_6: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_3, lt_3);  ge_3 = lt_3 = None
        ne_3: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg14_1, -1)
        bitwise_and_7: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_6, ne_3);  bitwise_and_6 = ne_3 = None
        unsqueeze_3: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_7, -1);  bitwise_and_7 = None
        _unsafe_masked_index_put_accumulate_4: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze_3, [arg14_1], mul_6);  unsqueeze_3 = arg14_1 = None
        ge_4: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg15_1, 0)
        lt_4: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg15_1, 1024)
        bitwise_and_8: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_4, lt_4);  ge_4 = lt_4 = None
        ne_4: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg15_1, -1)
        bitwise_and_9: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_8, ne_4);  bitwise_and_8 = ne_4 = None
        unsqueeze_4: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_9, -1);  bitwise_and_9 = None
        _unsafe_masked_index_put_accumulate_5: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze_4, [arg15_1], mul_6);  unsqueeze_4 = arg15_1 = None
        add_3: "f32[1024, 768]" = torch.ops.aten.add.Tensor(_unsafe_masked_index_put_accumulate_3, _unsafe_masked_index_put_accumulate_5);  _unsafe_masked_index_put_accumulate_3 = _unsafe_masked_index_put_accumulate_5 = None
        ge_5: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg16_1, 0)
        lt_5: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg16_1, 1024)
        bitwise_and_10: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_5, lt_5);  ge_5 = lt_5 = None
        ne_5: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg16_1, -1)
        bitwise_and_11: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_10, ne_5);  bitwise_and_10 = ne_5 = None
        unsqueeze_5: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_11, -1);  bitwise_and_11 = None
        _unsafe_masked_index_put_accumulate_6: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze_5, [arg16_1], mul_6);  full_2 = unsqueeze_5 = arg16_1 = None
        add_4: "f32[1024, 768]" = torch.ops.aten.add.Tensor(_unsafe_masked_index_put_accumulate_4, _unsafe_masked_index_put_accumulate_6);  _unsafe_masked_index_put_accumulate_4 = _unsafe_masked_index_put_accumulate_6 = None
        ge_6: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg17_1, 0)
        lt_6: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg17_1, 512)
        bitwise_and_12: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_6, lt_6);  ge_6 = lt_6 = None
        ne_6: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg17_1, -1)
        bitwise_and_13: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_12, ne_6);  bitwise_and_12 = ne_6 = None
        unsqueeze_6: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_13, -1);  bitwise_and_13 = None
        full_3: "f32[512, 768]" = torch.ops.aten.full.default(_shape_param_6, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        _unsafe_masked_index_put_accumulate_7: "f32[512, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_3, unsqueeze_6, [arg17_1], sum_5);  full_3 = unsqueeze_6 = arg17_1 = sum_5 = None
        ge_7: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg18_1, 0)
        lt_7: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg18_1, 30522)
        bitwise_and_14: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_7, lt_7);  ge_7 = lt_7 = None
        ne_7: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg18_1, 0)
        bitwise_and_15: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_14, ne_7);  bitwise_and_14 = ne_7 = None
        unsqueeze_7: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_15, -1);  bitwise_and_15 = None
        full_4: "f32[30522, 768]" = torch.ops.aten.full.default(_shape_param_7, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_7 = None
        _unsafe_masked_index_put_accumulate_8: "f32[30522, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_7, [arg18_1], mul_6);  full_4 = unsqueeze_7 = arg18_1 = mul_6 = None
        add_5: "f32[30522, 768]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate_8);  convert_element_type = _unsafe_masked_index_put_accumulate_8 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, _unsafe_masked_index_put_accumulate_2, add_3, add_4, _unsafe_masked_index_put_accumulate_7, add_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

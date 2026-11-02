"""Curated MM repro. Pattern hash: 12f2f044d8bd."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32128, 8192]", arg1_1: "bf16[8192, 512]", arg2_1: "bf16[8192, 512]", arg3_1: "bf16[8192, 512]", arg4_1: "bf16[8192, 512]", arg5_1: "f32[512]", arg6_1: "b8[8, 1024, 512]", arg7_1: "f32[8, 1024, 512]", arg8_1: "f32[8, 1024, 1]", arg9_1: "f32[8, 1024, 512]", arg10_1: "i64[8, 1024]", arg11_1: "bf16[8192, 512]", arg12_1: "bf16[8192, 512]", arg13_1: "bf16[8192, 512]", arg14_1: "f32[512]", arg15_1: "b8[8, 1024, 512]", arg16_1: "f32[8, 1024, 1]", arg17_1: "f32[8, 1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        mm: "bf16[32128, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[32128, 512]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        convert_element_type_1: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        convert_element_type_2: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_2);  convert_element_type_1 = convert_element_type_2 = None
        view_2: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(arg4_1, _shape_param_2);  arg4_1 = _shape_param_2 = None
        convert_element_type_3: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_1: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add, convert_element_type_3);  add = convert_element_type_3 = None
        mul: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_1, arg5_1);  arg5_1 = None
        mul_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg6_1, arg7_1)
        mul_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None
        mul_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_2, arg8_1)
        mul_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_1, mul_3);  add_1 = mul_3 = None
        sum_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_4, [0, 1], True, dtype = torch.float32);  mul_4 = None
        view_3: "f32[512]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        mul_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul, mul_2)
        mul_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul, arg8_1);  mul = None
        sum_2: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [2], True, dtype = torch.float32);  mul_5 = None
        add_2: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(arg9_1, mul_6);  arg9_1 = mul_6 = None
        pow_1: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg8_1, 3);  arg8_1 = None
        mul_7: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_8: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_7, pow_1);  mul_7 = pow_1 = None
        expand: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_8, _shape_param_4);  mul_8 = _shape_param_4 = None
        div: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand, 512);  expand = None
        pow_2: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_2, 1.0);  mul_2 = None
        mul_9: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        mul_10: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div, mul_9);  div = mul_9 = None
        add_3: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_2, mul_10);  add_2 = mul_10 = None
        convert_element_type_4: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        mul_11: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_12: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_3, mul_11);  add_3 = mul_11 = None
        ge: "b8[8, 1024]" = torch.ops.aten.ge.Scalar(arg10_1, 0)
        lt: "b8[8, 1024]" = torch.ops.aten.lt.Scalar(arg10_1, 32128)
        bitwise_and: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg10_1, -1)
        bitwise_and_1: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[32128, 512]" = torch.ops.aten.full.default(_shape_param_5, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        _unsafe_masked_index_put_accumulate: "f32[32128, 512]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg10_1], mul_12);  mul_12 = None
        add_4: "f32[32128, 512]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate);  convert_element_type = _unsafe_masked_index_put_accumulate = None
        view_4: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(arg11_1, _shape_param_6);  arg11_1 = _shape_param_6 = None
        convert_element_type_5: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        view_5: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(arg12_1, _shape_param_7);  arg12_1 = _shape_param_7 = None
        convert_element_type_6: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        add_5: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_5, convert_element_type_6);  convert_element_type_5 = convert_element_type_6 = None
        view_6: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(arg13_1, _shape_param_8);  arg13_1 = _shape_param_8 = None
        convert_element_type_7: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        add_6: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_7);  add_5 = convert_element_type_7 = None
        mul_13: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_6, arg14_1);  arg14_1 = None
        mul_14: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg15_1, arg7_1);  arg7_1 = None
        mul_15: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        mul_16: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_15, arg16_1)
        mul_17: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_6, mul_16);  add_6 = mul_16 = None
        sum_3: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_17, [0, 1], True, dtype = torch.float32);  mul_17 = None
        view_7: "f32[512]" = torch.ops.aten.view.default(sum_3, _shape_param_9);  sum_3 = _shape_param_9 = None
        mul_18: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_13, mul_15)
        mul_19: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_13, arg16_1);  mul_13 = None
        sum_4: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True, dtype = torch.float32);  mul_18 = None
        add_7: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(arg17_1, mul_19);  arg17_1 = mul_19 = None
        pow_3: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg16_1, 3);  arg16_1 = None
        mul_20: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_4, -0.5);  sum_4 = None
        mul_21: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_20, pow_3);  mul_20 = pow_3 = None
        expand_1: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_21, _shape_param_10);  mul_21 = _shape_param_10 = None
        div_1: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_1, 512);  expand_1 = None
        pow_4: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_15, 1.0);  mul_15 = None
        mul_22: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_4, 2.0);  pow_4 = None
        mul_23: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_1, mul_22);  div_1 = mul_22 = None
        add_8: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_7, mul_23);  add_7 = mul_23 = None
        convert_element_type_8: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.float32);  arg15_1 = None
        mul_24: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_25: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_8, mul_24);  add_8 = mul_24 = None
        _unsafe_masked_index_put_accumulate_1: "f32[32128, 512]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg10_1], mul_25);  full = unsqueeze = arg10_1 = mul_25 = None
        add_9: "f32[32128, 512]" = torch.ops.aten.add.Tensor(add_4, _unsafe_masked_index_put_accumulate_1);  add_4 = _unsafe_masked_index_put_accumulate_1 = None
        return (view_3, view_7, add_9)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

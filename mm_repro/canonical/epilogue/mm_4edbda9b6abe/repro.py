"""Curated MM repro. Pattern hash: 4edbda9b6abe."""
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
    def forward(self, arg0_1: "bf16[29056, 8192]", arg1_1: "bf16[8192, 1024]", arg2_1: "bf16[8192, 1024]", arg3_1: "bf16[8192, 1024]", arg4_1: "bf16[8192, 1024]", arg5_1: "f32[1024]", arg6_1: "f32[16, 512, 1024]", arg7_1: "f32[16, 512, 1]", arg8_1: "f32[16, 512, 1024]", arg9_1: "b8[16, 512, 1024]", arg10_1: "i64[1, 512]", arg11_1: "i64[16, 512]", arg12_1: "i64[16, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        mm: "bf16[29056, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[29056, 1024]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        convert_element_type_1: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        convert_element_type_2: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_2);  convert_element_type_1 = convert_element_type_2 = None
        view_2: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(arg4_1, _shape_param_2);  arg4_1 = _shape_param_2 = None
        convert_element_type_3: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add, convert_element_type_3);  add = convert_element_type_3 = None
        mul: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_1, arg5_1);  arg5_1 = None
        mul_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sum_1: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = None
        sum_2: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg6_1, sum_2);  sum_2 = None
        sub: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg7_1, sub_1);  arg7_1 = sub_1 = None
        mul_5: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_1, arg6_1);  arg6_1 = None
        sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_1, [0, 1]);  add_1 = None
        add_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(arg8_1, mul_4);  arg8_1 = mul_4 = None
        convert_element_type_4: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        mul_6: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_7: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_2, mul_6);  add_2 = mul_6 = None
        sum_5: "f32[1, 512, 1024]" = torch.ops.aten.sum.dim_IntList(mul_7, [0], True, dtype = torch.float32)
        ge: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg10_1, 0)
        lt: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg10_1, 512)
        bitwise_and: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg10_1, -1)
        bitwise_and_1: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[512, 1024]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg10_1], sum_5);  full = unsqueeze = arg10_1 = sum_5 = None
        full_1: "b8[16, 512, 1]" = torch.ops.aten.full.default(_shape_param_4, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        full_2: "f32[2, 1024]" = torch.ops.aten.full.default(_shape_param_5, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        _unsafe_masked_index_put_accumulate_1: "f32[2, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, full_1, [arg11_1], mul_7);  full_2 = full_1 = arg11_1 = None
        ge_1: "b8[16, 512]" = torch.ops.aten.ge.Scalar(arg12_1, 0)
        lt_1: "b8[16, 512]" = torch.ops.aten.lt.Scalar(arg12_1, 29056)
        bitwise_and_2: "b8[16, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[16, 512]" = torch.ops.aten.ne.Scalar(arg12_1, 0)
        bitwise_and_3: "b8[16, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[16, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_3: "f32[29056, 1024]" = torch.ops.aten.full.default(_shape_param_6, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        _unsafe_masked_index_put_accumulate_2: "f32[29056, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_3, unsqueeze_1, [arg12_1], mul_7);  full_3 = unsqueeze_1 = arg12_1 = mul_7 = None
        add_3: "f32[29056, 1024]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate_2);  convert_element_type = _unsafe_masked_index_put_accumulate_2 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, add_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

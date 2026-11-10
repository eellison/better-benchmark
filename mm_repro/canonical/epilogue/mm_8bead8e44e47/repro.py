"""Curated MM repro. Pattern hash: 8bead8e44e47."""
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
    def forward(self, arg0_1: "bf16[30522, 32768]", arg1_1: "bf16[32768, 768]", arg2_1: "bf16[32768, 768]", arg3_1: "f32[256, 128, 768]", arg4_1: "bf16[32768, 768]", arg5_1: "bf16[32768, 768]", arg6_1: "b8[256, 128, 768]", arg7_1: "f32[768]", arg8_1: "f32[256, 128, 768]", arg9_1: "f32[1, 128, 768]", arg10_1: "f32[256, 128, 1]", arg11_1: "f32[256, 128, 1]", arg12_1: "i64[1, 512]", arg13_1: "i64[256, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[30522, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[30522, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view: "bf16[256, 128, 768]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        convert_element_type_1: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(arg3_1, convert_element_type_1);  arg3_1 = convert_element_type_1 = None
        view_1: "bf16[256, 128, 768]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        convert_element_type_2: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add_1: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        view_2: "bf16[256, 128, 768]" = torch.ops.aten.view.default(arg5_1, _shape_param_2);  arg5_1 = _shape_param_2 = None
        convert_element_type_3: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_2: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_3);  add_1 = convert_element_type_3 = None
        convert_element_type_4: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        mul: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_2, mul);  add_2 = mul = None
        mul_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg7_1);  arg7_1 = None
        mul_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_1: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        add_3: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(arg8_1, arg9_1);  arg8_1 = arg9_1 = None
        sub: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, arg10_1);  add_3 = arg10_1 = None
        mul_4: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub, arg11_1);  sub = None
        mul_5: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, mul_4);  mul_2 = None
        sum_2: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [2], True);  mul_5 = None
        mul_6: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_2);  sum_2 = None
        sub_1: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_1);  mul_3 = sum_1 = None
        sub_2: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_1, mul_6);  sub_1 = mul_6 = None
        div: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(arg11_1, 768);  arg11_1 = None
        mul_7: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_8: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_1, mul_4);  mul_4 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_8, [0, 1]);  mul_8 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1, [0, 1]);  mul_1 = None
        sum_5: "f32[1, 128, 768]" = torch.ops.aten.sum.dim_IntList(mul_7, [0], True, dtype = torch.float32)
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg12_1, 1, 0, 128);  arg12_1 = None
        ge: "b8[1, 128]" = torch.ops.aten.ge.Scalar(slice_1, 0)
        lt: "b8[1, 128]" = torch.ops.aten.lt.Scalar(slice_1, 512)
        bitwise_and: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 128]" = torch.ops.aten.ne.Scalar(slice_1, -1)
        bitwise_and_1: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[512, 768]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [slice_1], sum_5);  full = unsqueeze = slice_1 = sum_5 = None
        ge_1: "b8[256, 128]" = torch.ops.aten.ge.Scalar(arg13_1, 0)
        lt_1: "b8[256, 128]" = torch.ops.aten.lt.Scalar(arg13_1, 30522)
        bitwise_and_2: "b8[256, 128]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[256, 128]" = torch.ops.aten.ne.Scalar(arg13_1, 0)
        bitwise_and_3: "b8[256, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[256, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_1: "f32[30522, 768]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        _unsafe_masked_index_put_accumulate_1: "f32[30522, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, unsqueeze_1, [arg13_1], mul_7);  full_1 = unsqueeze_1 = arg13_1 = mul_7 = None
        add_4: "f32[30522, 768]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate_1);  convert_element_type = _unsafe_masked_index_put_accumulate_1 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, add_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

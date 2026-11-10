"""Curated MM repro. Pattern hash: e50c9658dcc1."""
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
    def forward(self, arg0_1: "bf16[50264, 16384]", arg1_1: "bf16[16384, 768]", arg2_1: "bf16[16384, 768]", arg3_1: "f32[768]", arg4_1: "f32[32, 512, 768]", arg5_1: "f32[1, 512, 768]", arg6_1: "b8[32, 512, 768]", arg7_1: "f32[32, 512, 1]", arg8_1: "f32[32, 512, 1]", arg9_1: "f32[32, 512, 768]", arg10_1: "i64[1, 512]", arg11_1: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[50264, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "bf16[50257, 768]" = torch.ops.aten.slice.Tensor(mm, 0, 0, -7);  mm = None
        convert_element_type: "f32[50257, 768]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        convert_element_type_1: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        view: "f32[32, 512, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg3_1);  arg3_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg6_1, add);  add = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_3, arg7_1);  mul_3 = arg7_1 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, arg8_1);  sub = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, mul_4);  mul = None
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [2], True);  mul_5 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_2);  sum_2 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_1, mul_6);  sub_1 = mul_6 = None
        div: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(arg8_1, 768);  arg8_1 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, mul_4);  mul_4 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_8, [0, 1]);  mul_8 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(view, [0, 1]);  view = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg9_1, mul_7);  arg9_1 = mul_7 = None
        convert_element_type_2: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_1, mul_9);  add_1 = mul_9 = None
        sum_5: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_10, [0], True, dtype = torch.float32)
        full: "b8[1, 512, 1]" = torch.ops.aten.full.default(_shape_param_1, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        full_1: "f32[1024, 768]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        _unsafe_masked_index_put_accumulate: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, full, [arg10_1], sum_5);  full_1 = full = arg10_1 = sum_5 = None
        ge: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg11_1, 0)
        lt: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg11_1, 50257)
        bitwise_and: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg11_1, -1)
        bitwise_and_1: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_2: "f32[50257, 768]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate_1: "f32[50257, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze, [arg11_1], mul_10);  full_2 = unsqueeze = arg11_1 = mul_10 = None
        add_2: "f32[50257, 768]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate_1);  convert_element_type = _unsafe_masked_index_put_accumulate_1 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, add_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""Curated MM repro. Pattern hash: c9525fdd5c1d."""
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
    def forward(self, arg0_1: "bf16[8192, 2304]", arg1_1: "bf16[2304, 768]", arg2_1: "f32[768]", arg3_1: "f32[8, 1024, 768]", arg4_1: "f32[1, 1024, 768]", arg5_1: "b8[8, 1024, 768]", arg6_1: "f32[8, 1024, 1]", arg7_1: "f32[8, 1024, 1]", arg8_1: "f32[8, 1024, 768]", arg9_1: "i64[1, 1024]", arg10_1: "i64[8, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[8192, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view: "f32[8, 1024, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view, arg2_1);  arg2_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        add: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg5_1, add);  add = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_3, arg6_1);  mul_3 = arg6_1 = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, arg7_1);  sub = None
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, mul_4);  mul = None
        sum_2: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [2], True);  mul_5 = None
        mul_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_2);  sum_2 = None
        sub_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_1, mul_6);  sub_1 = mul_6 = None
        div: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(arg7_1, 768);  arg7_1 = None
        mul_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view, mul_4);  mul_4 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_8, [0, 1]);  mul_8 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(view, [0, 1]);  view = None
        add_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(arg8_1, mul_7);  arg8_1 = mul_7 = None
        convert_element_type_1: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        mul_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_1, mul_9);  add_1 = mul_9 = None
        sum_5: "f32[1, 1024, 768]" = torch.ops.aten.sum.dim_IntList(mul_10, [0], True, dtype = torch.float32)
        full: "b8[1, 1024, 1]" = torch.ops.aten.full.default(_shape_param_1, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        full_1: "f32[1024, 768]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        _unsafe_masked_index_put_accumulate: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, full, [arg9_1], sum_5);  full_1 = full = arg9_1 = sum_5 = None
        ge: "b8[8, 1024]" = torch.ops.aten.ge.Scalar(arg10_1, 0)
        lt: "b8[8, 1024]" = torch.ops.aten.lt.Scalar(arg10_1, 50257)
        bitwise_and: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg10_1, -1)
        bitwise_and_1: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_2: "f32[50257, 768]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate_1: "f32[50257, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze, [arg10_1], mul_10);  full_2 = unsqueeze = arg10_1 = mul_10 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

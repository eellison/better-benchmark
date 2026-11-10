"""Curated MM repro. Pattern hash: b7c24d3ba62b."""
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
    def forward(self, arg0_1: "bf16[128, 16384]", arg1_1: "bf16[16384, 4096]", arg2_1: "bf16[128, 4096]", arg3_1: "bf16[128, 4096]", arg4_1: "bf16[128, 4096]", arg5_1: "f32[4096]", arg6_1: "f32[1, 128, 4096]", arg7_1: "f32[1, 128, 1]", arg8_1: "f32[1, 128, 1]", arg9_1: "f32[1, 128, 4096]", arg10_1: "i64[1, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        view_2: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        view_3: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        convert_element_type_3: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        add_2: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_3);  add_1 = convert_element_type_3 = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_2, arg5_1);  arg5_1 = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, 4096)
        sum_1: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(arg6_1, arg7_1);  arg6_1 = arg7_1 = None
        mul_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, arg8_1);  sub = None
        mul_3: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(arg8_1, 4096);  arg8_1 = None
        mul_5: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_2, mul_2);  mul_2 = None
        sum_3: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_2, [0, 1]);  add_2 = None
        add_3: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(arg9_1, mul_5);  arg9_1 = mul_5 = None
        ge: "b8[1, 128]" = torch.ops.aten.ge.Scalar(arg10_1, 0)
        lt: "b8[1, 128]" = torch.ops.aten.lt.Scalar(arg10_1, 50400)
        bitwise_and: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 128]" = torch.ops.aten.ne.Scalar(arg10_1, -1)
        bitwise_and_1: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[50400, 4096]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        _unsafe_masked_index_put_accumulate: "f32[50400, 4096]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg10_1], add_3);  full = unsqueeze = arg10_1 = add_3 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

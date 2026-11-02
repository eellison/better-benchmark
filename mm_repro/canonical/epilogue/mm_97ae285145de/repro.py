"""Curated MM repro. Pattern hash: 97ae285145de."""
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
    def forward(self, arg0_1: "bf16[50264, 4096]", arg1_1: "bf16[4096, 2048]", arg2_1: "bf16[4096, 2048]", arg3_1: "bf16[4096, 2048]", arg4_1: "bf16[4096, 2048]", arg5_1: "f32[2048]", arg6_1: "f32[32, 128, 2048]", arg7_1: "f32[1, 128, 2048]", arg8_1: "f32[32, 128, 1]", arg9_1: "f32[32, 128, 1]", arg10_1: "f32[32, 128, 2048]", arg11_1: "i64[1, 128]", arg12_1: "i64[32, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mm: "bf16[50264, 2048]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "bf16[50257, 2048]" = torch.ops.aten.slice.Tensor(mm, 0, 0, -7);  mm = None
        convert_element_type: "f32[50257, 2048]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        view: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        convert_element_type_1: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        convert_element_type_2: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_2);  convert_element_type_1 = convert_element_type_2 = None
        view_2: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(arg4_1, _shape_param_2);  arg4_1 = _shape_param_2 = None
        convert_element_type_3: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_1: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add, convert_element_type_3);  add = convert_element_type_3 = None
        mul: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_1, arg5_1);  arg5_1 = None
        mul_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, 2048)
        sum_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        add_2: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(arg6_1, arg7_1);  arg6_1 = arg7_1 = None
        sub: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_2, arg8_1);  add_2 = arg8_1 = None
        mul_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub, arg9_1);  sub = None
        mul_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(arg9_1, 2048);  arg9_1 = None
        mul_5: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_1, mul_2);  mul_2 = None
        sum_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_4: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_1, [0, 1]);  add_1 = None
        add_3: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(arg10_1, mul_5);  arg10_1 = mul_5 = None
        sum_5: "f32[1, 128, 2048]" = torch.ops.aten.sum.dim_IntList(add_3, [0], True, dtype = torch.float32)
        full: "b8[1, 128, 1]" = torch.ops.aten.full.default(_shape_param_3, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        full_1: "f32[2048, 2048]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        _unsafe_masked_index_put_accumulate: "f32[2048, 2048]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, full, [arg11_1], sum_5);  full_1 = full = arg11_1 = sum_5 = None
        ge: "b8[32, 128]" = torch.ops.aten.ge.Scalar(arg12_1, 0)
        lt: "b8[32, 128]" = torch.ops.aten.lt.Scalar(arg12_1, 50257)
        bitwise_and: "b8[32, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg12_1, -1)
        bitwise_and_1: "b8[32, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[32, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_2: "f32[50257, 2048]" = torch.ops.aten.full.default(_shape_param_5, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        _unsafe_masked_index_put_accumulate_1: "f32[50257, 2048]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze, [arg12_1], add_3);  full_2 = unsqueeze = arg12_1 = add_3 = None
        add_4: "f32[50257, 2048]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate_1);  convert_element_type = _unsafe_masked_index_put_accumulate_1 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, add_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

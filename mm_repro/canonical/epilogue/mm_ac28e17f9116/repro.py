"""Curated MM repro. Pattern hash: ac28e17f9116."""
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
    def forward(self, arg0_1: "bf16[30522, 32768]", arg1_1: "bf16[32768, 128]", arg2_1: "bf16[32768, 128]", arg3_1: "b8[64, 512, 128]", arg4_1: "f32[128]", arg5_1: "f32[64, 512, 128]", arg6_1: "f32[64, 512, 1]", arg7_1: "i64[1, 512]", arg8_1: "i64[1, 512]", arg9_1: "i64[64, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[30522, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[30522, 128]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view: "bf16[64, 512, 128]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        convert_element_type_1: "f32[64, 512, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_2: "f32[64, 512, 128]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        mul: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul);  convert_element_type_1 = mul = None
        mul_2: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, arg4_1);  arg4_1 = None
        mul_3: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_2, 128)
        sum_1: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  mul_2 = None
        sum_2: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(arg5_1, sum_2);  sum_2 = None
        sub: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_3, sum_1);  mul_3 = sum_1 = None
        sub_1: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(sub, mul_5);  sub = mul_5 = None
        mul_6: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(arg6_1, sub_1);  arg6_1 = sub_1 = None
        mul_7: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, arg5_1);  arg5_1 = None
        sum_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_4: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1, [0, 1]);  mul_1 = None
        sum_5: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_6, [0], True, dtype = torch.float32)
        ge: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg7_1, 0)
        lt: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg7_1, 512)
        bitwise_and: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg7_1, -1)
        bitwise_and_1: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[512, 128]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 128]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg7_1], sum_5);  full = unsqueeze = arg7_1 = sum_5 = None
        expand: "i64[64, 512]" = torch.ops.aten.expand.default(arg8_1, _shape_param_2);  arg8_1 = _shape_param_2 = None
        ge_1: "b8[64, 512]" = torch.ops.aten.ge.Scalar(expand, 0)
        lt_1: "b8[64, 512]" = torch.ops.aten.lt.Scalar(expand, 2)
        bitwise_and_2: "b8[64, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[64, 512]" = torch.ops.aten.ne.Scalar(expand, -1)
        bitwise_and_3: "b8[64, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[64, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_1: "f32[2, 128]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate_1: "f32[2, 128]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, unsqueeze_1, [expand], mul_6);  full_1 = unsqueeze_1 = expand = None
        ge_2: "b8[64, 512]" = torch.ops.aten.ge.Scalar(arg9_1, 0)
        lt_2: "b8[64, 512]" = torch.ops.aten.lt.Scalar(arg9_1, 30522)
        bitwise_and_4: "b8[64, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_2: "b8[64, 512]" = torch.ops.aten.ne.Scalar(arg9_1, 0)
        bitwise_and_5: "b8[64, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_2);  bitwise_and_4 = ne_2 = None
        unsqueeze_2: "b8[64, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_2: "f32[30522, 128]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        _unsafe_masked_index_put_accumulate_2: "f32[30522, 128]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze_2, [arg9_1], mul_6);  full_2 = unsqueeze_2 = arg9_1 = mul_6 = None
        add: "f32[30522, 128]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate_2);  convert_element_type = _unsafe_masked_index_put_accumulate_2 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, add)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

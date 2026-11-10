"""Curated MM repro. Pattern hash: ff2173253f4a."""
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "bf16[1024, 1024]", arg2_1: "f32[1024]", arg3_1: "bf16[8192, 1024]", arg4_1: "b8[64, 128, 1024]", arg5_1: "f32[64, 128, 1024]", arg6_1: "f32[64, 128, 1]", arg7_1: "f32[64, 128, 1]", arg8_1: "f32[64, 128, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sum_1: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        view_1: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        mul_2: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(arg4_1, view_1);  view_1 = None
        mul_3: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        add: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(arg5_1, mul_3);  arg5_1 = mul_3 = None
        sub: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add, arg6_1);  add = arg6_1 = None
        mul_4: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, arg7_1);  sub = None
        mul_5: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, mul_4);  mul = None
        sum_2: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [2], True);  mul_5 = None
        mul_6: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_4, sum_2);  sum_2 = None
        sub_1: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_1, mul_6);  sub_1 = mul_6 = None
        div: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(arg7_1, 1024);  arg7_1 = None
        mul_7: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_8: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_4);  mul_4 = None
        sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_8, [0, 1]);  mul_8 = None
        sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1]);  convert_element_type = None
        add_1: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(arg8_1, mul_7);  arg8_1 = mul_7 = None
        convert_element_type_1: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16)
        convert_element_type_2: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        mul_9: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_10: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_9);  convert_element_type_1 = mul_9 = None
        view_2: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_10, _shape_param_2);  mul_10 = _shape_param_2 = None
        permute: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_5: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[1024]" = torch.ops.aten.view.default(sum_5, _shape_param_3);  sum_5 = _shape_param_3 = None
        convert_element_type_3: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_4: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (sum_3, sum_4, add_1, view_2, permute, convert_element_type_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

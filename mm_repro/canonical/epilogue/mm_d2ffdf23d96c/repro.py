"""Curated MM repro. Pattern hash: d2ffdf23d96c."""
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
    def forward(self, arg0_1: "bf16[8192, 3072]", arg1_1: "bf16[3072, 768]", arg2_1: "f32[768]", arg3_1: "bf16[8192, 768]", arg4_1: "b8[4, 2048, 768]", arg5_1: "f32[4, 2048, 768]", arg6_1: "f32[8192, 1]", arg7_1: "f32[8192, 1]", arg8_1: "f32[8192, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[8192, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mul: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        view: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        mul_2: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(arg4_1, view);  view = None
        mul_3: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        add: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(arg5_1, mul_3);  arg5_1 = mul_3 = None
        view_1: "f32[8192, 768]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        sub: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(view_1, arg6_1);  view_1 = arg6_1 = None
        mul_4: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(sub, arg7_1);  sub = None
        mul_5: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul, mul_4);  mul = None
        sum_2: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [1], True);  mul_5 = None
        mul_6: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_2);  sum_2 = None
        sub_1: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(sub_1, mul_6);  sub_1 = mul_6 = None
        div: "f32[8192, 1]" = torch.ops.aten.div.Tensor(arg7_1, 768);  arg7_1 = None
        mul_7: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_8: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_4);  mul_4 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_8, [0]);  mul_8 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0]);  convert_element_type = None
        add_1: "f32[8192, 768]" = torch.ops.aten.add.Tensor(arg8_1, mul_7);  arg8_1 = mul_7 = None
        view_2: "f32[4, 2048, 768]" = torch.ops.aten.view.default(add_1, _shape_param_2);  add_1 = _shape_param_2 = None
        convert_element_type_1: "bf16[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16)
        convert_element_type_2: "bf16[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        mul_9: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_10: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_9);  convert_element_type_1 = mul_9 = None
        view_3: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_10, _shape_param_3);  mul_10 = _shape_param_3 = None
        permute: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_3, [1, 0])
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32)
        view_4: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_4);  sum_5 = _shape_param_4 = None
        convert_element_type_3: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_4: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (sum_3, sum_4, view_2, view_3, permute, convert_element_type_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

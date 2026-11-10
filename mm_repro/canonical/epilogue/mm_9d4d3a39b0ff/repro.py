"""Curated MM repro. Pattern hash: 9d4d3a39b0ff."""
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
    def forward(self, arg0_1: "bf16[32768, 384]", arg1_1: "bf16[384, 192]", arg2_1: "f32[192]", arg3_1: "bf16[512, 64, 192]", arg4_1: "f32[512, 64, 1]", arg5_1: "f32[512, 64, 1]", arg6_1: "bf16[512, 64, 192]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[32768, 192]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[512, 64, 192]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[512, 64, 192]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul, 192)
        sum_1: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        convert_element_type_1: "f32[512, 64, 192]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul_2: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = None
        mul_3: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(arg5_1, 192);  arg5_1 = None
        mul_5: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_2);  mul_2 = None
        sum_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_4: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1]);  convert_element_type = None
        convert_element_type_2: "bf16[512, 64, 192]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        add: "bf16[512, 64, 192]" = torch.ops.aten.add.Tensor(arg6_1, convert_element_type_2);  arg6_1 = convert_element_type_2 = None
        view_1: "bf16[32768, 192]" = torch.ops.aten.view.default(add, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[192, 32768]" = torch.ops.aten.permute.default(view_1, [1, 0])
        sum_5: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32)
        view_2: "f32[192]" = torch.ops.aten.view.default(sum_5, _shape_param_2);  sum_5 = _shape_param_2 = None
        convert_element_type_3: "bf16[192]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_4: "f32[192]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (sum_3, sum_4, add, view_1, permute, convert_element_type_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

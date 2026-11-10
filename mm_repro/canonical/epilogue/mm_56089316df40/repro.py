"""Curated MM repro. Pattern hash: 56089316df40."""
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
    def forward(self, arg0_1: "bf16[175360, 2304]", arg1_1: "bf16[2304, 768]", arg2_1: "f32[768]", arg3_1: "f32[128, 1370, 768]", arg4_1: "f32[1, 1370, 768]", arg5_1: "f32[128, 1370, 1]", arg6_1: "f32[128, 1370, 1]", arg7_1: "f32[128, 1370, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[175360, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 1370, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        add: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        sub: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(add, arg5_1);  add = arg5_1 = None
        mul_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(sub, arg6_1);  sub = None
        mul_3: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[128, 1370, 1]" = torch.ops.aten.div.Tensor(arg6_1, 768);  arg6_1 = None
        mul_5: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_2);  mul_2 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1]);  convert_element_type = None
        add_1: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(arg7_1, mul_5);  arg7_1 = mul_5 = None
        sum_5: "f32[1, 1370, 768]" = torch.ops.aten.sum.dim_IntList(add_1, [0], True, dtype = torch.float32)
        slice_1: "f32[128, 1, 768]" = torch.ops.aten.slice.Tensor(add_1, 1, 0, 1)
        slice_2: "f32[128, 1369, 768]" = torch.ops.aten.slice.Tensor(add_1, 1, 1, 1370);  add_1 = None
        convert_element_type_1: "bf16[128, 1369, 768]" = torch.ops.prims.convert_element_type.default(slice_2, torch.bfloat16);  slice_2 = None
        sum_6: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_1, [0], True, dtype = torch.float32);  slice_1 = None
        permute: "bf16[128, 768, 1369]" = torch.ops.aten.permute.default(convert_element_type_1, [0, 2, 1]);  convert_element_type_1 = None
        view_1: "bf16[128, 768, 37, 37]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        sum_7: "bf16[768]" = torch.ops.aten.sum.dim_IntList(view_1, [0, 2, 3])
        convert_element_type_2: "f32[768]" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None
        return (sum_3, sum_4, sum_5, sum_6, view_1, convert_element_type_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

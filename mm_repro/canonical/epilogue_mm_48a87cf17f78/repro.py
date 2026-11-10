"""Curated MM repro. Pattern hash: 48a87cf17f78."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 192]", arg2_1: "f32[192]", arg3_1: "f32[128, 197, 192]", arg4_1: "f32[128, 197, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[128, 192]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[128, 192]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        full: "f32[128, 197, 192]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "f32[128, 197, 192]" = torch.ops.aten.select_scatter.default(full, convert_element_type, 1, 0);  full = convert_element_type = None
        mul: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(select_scatter, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, 192)
        sum_1: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = None
        sum_2: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(arg3_1, sum_2);  sum_2 = None
        sub: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        mul_5: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(select_scatter, arg3_1);  arg3_1 = None
        sum_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[192]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None
        convert_element_type_1: "bf16[128, 197, 192]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        view: "bf16[25216, 192]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        permute: "bf16[192, 25216]" = torch.ops.aten.permute.default(view, [1, 0])
        sum_5: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32)
        view_1: "f32[192]" = torch.ops.aten.view.default(sum_5, _shape_param_2);  sum_5 = _shape_param_2 = None
        convert_element_type_2: "bf16[192]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_3: "f32[192]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        return (mul_4, sum_3, sum_4, view, permute, convert_element_type_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

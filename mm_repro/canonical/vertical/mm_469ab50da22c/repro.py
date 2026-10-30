"""Curated MM repro. Pattern hash: 469ab50da22c."""
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
    def forward(self, arg0_1: "f32[16, 2]", arg1_1: "f32[16, 2]", arg2_1: "bf16[2, 768]", arg3_1: "bf16[16, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        exp: "f32[16, 2]" = torch.ops.aten.exp.default(arg0_1);  arg0_1 = None
        sum_1: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(arg1_1, [1], True)
        mul: "f32[16, 2]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub: "f32[16, 2]" = torch.ops.aten.sub.Tensor(arg1_1, mul);  arg1_1 = mul = None
        convert_element_type: "bf16[16, 2]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        mm: "bf16[16, 768]" = torch.ops.aten.mm.default(convert_element_type, arg2_1);  arg2_1 = None
        permute: "bf16[2, 16]" = torch.ops.aten.permute.default(convert_element_type, [1, 0])
        mm_1: "bf16[2, 768]" = torch.ops.aten.mm.default(permute, arg3_1);  permute = arg3_1 = None
        sum_2: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0], True, dtype = torch.float32);  convert_element_type = None
        view: "f32[2]" = torch.ops.aten.view.default(sum_2, _shape_param_0);  sum_2 = _shape_param_0 = None
        convert_element_type_1: "bf16[2]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_2: "f32[16, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_3: "f32[2, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_4: "f32[2]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        return (convert_element_type_2, convert_element_type_3, convert_element_type_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

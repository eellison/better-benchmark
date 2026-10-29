"""Curated MM repro. Pattern hash: 00455bdaf9ef."""
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
    def forward(self, arg0_1: "f32[16, 2]", arg1_1: "f32[16, 2]", arg2_1: "bf16[2, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        exp: "f32[16, 2]" = torch.ops.aten.exp.default(arg0_1);  arg0_1 = None
        sum_1: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(arg1_1, [1], True)
        mul: "f32[16, 2]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub: "f32[16, 2]" = torch.ops.aten.sub.Tensor(arg1_1, mul);  arg1_1 = mul = None
        convert_element_type: "bf16[16, 2]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        mm: "bf16[16, 768]" = torch.ops.aten.mm.default(convert_element_type, arg2_1);  convert_element_type = arg2_1 = None
        convert_element_type_1: "f32[16, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        full: "f32[16, 128, 768]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "f32[16, 128, 768]" = torch.ops.aten.select_scatter.default(full, convert_element_type_1, 1, 0);  full = convert_element_type_1 = None
        return select_scatter

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

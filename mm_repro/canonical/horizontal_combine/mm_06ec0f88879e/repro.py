"""Curated MM repro. Pattern hash: 06ec0f88879e."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 768]", arg2_1: "bf16[1000, 768]", arg3_1: "f32[768]", arg4_1: "f32[128, 198, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        div: "bf16[128, 1000]" = torch.ops.aten.div.Tensor(arg0_1, 2);  arg0_1 = None
        mm: "bf16[128, 768]" = torch.ops.aten.mm.default(div, arg1_1);  arg1_1 = None
        convert_element_type: "f32[128, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mm_1: "bf16[128, 768]" = torch.ops.aten.mm.default(div, arg2_1);  div = arg2_1 = None
        convert_element_type_1: "f32[128, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        full: "f32[128, 198, 768]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "f32[128, 198, 768]" = torch.ops.aten.select_scatter.default(full, convert_element_type, 1, 1);  convert_element_type = None
        select_scatter_1: "f32[128, 198, 768]" = torch.ops.aten.select_scatter.default(full, convert_element_type_1, 1, 0);  full = convert_element_type_1 = None
        add: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        mul: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        mul_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        mul_2: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(add, arg4_1);  arg4_1 = None
        return (add, mul, mul_1, mul_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

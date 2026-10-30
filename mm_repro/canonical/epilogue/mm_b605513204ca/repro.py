"""Curated MM repro. Pattern hash: b605513204ca."""
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096, 16384]", arg2_1: "f32[4096, 16384]", arg3_1: "f32[4096, 16384]", arg4_1: "f32[4096, 16384]", arg5_1: "f32[4096, 16384]"):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 16384]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        add: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        add_1: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        add_2: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_1, arg4_1);  add_1 = arg4_1 = None
        add_3: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_2, arg5_1);  add_2 = arg5_1 = None
        return add_3

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

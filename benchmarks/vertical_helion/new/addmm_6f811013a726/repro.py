"""Curated MM repro. Pattern hash: 6f811013a726."""
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
    def forward(self, arg0_1: "bf16[16, 16]", arg1_1: "bf16[16]", arg2_1: "bf16[128, 16]", arg3_1: "bf16[16, 16]", arg4_1: "bf16[16]", arg5_1: "bf16[16, 1]", arg6_1: "bf16[1]"):
        # No stacktrace found for following nodes
        addmm: "bf16[128, 16]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        tanh: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm);  addmm = None
        addmm_1: "bf16[128, 16]" = torch.ops.aten.addmm.default(arg4_1, tanh, arg3_1);  arg4_1 = tanh = arg3_1 = None
        tanh_1: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm_1);  addmm_1 = None
        addmm_2: "bf16[128, 1]" = torch.ops.aten.addmm.default(arg6_1, tanh_1, arg5_1);  arg6_1 = tanh_1 = arg5_1 = None
        return addmm_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

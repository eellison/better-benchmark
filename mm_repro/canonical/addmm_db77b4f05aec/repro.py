"""Curated MM repro. Pattern hash: db77b4f05aec."""
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
    def forward(self, arg0_1: "bf16[512, 65]", arg1_1: "bf16[65]", arg2_1: "bf16[96, 512]"):
        # No stacktrace found for following nodes
        addmm: "bf16[96, 65]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        sigmoid: "bf16[96, 65]" = torch.ops.aten.sigmoid.default(addmm);  addmm = None
        return sigmoid

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

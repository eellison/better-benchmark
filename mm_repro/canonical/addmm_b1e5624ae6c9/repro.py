"""Curated MM repro. Pattern hash: b1e5624ae6c9."""
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
    def forward(self, arg0_1: "bf16[768, 3072]", arg1_1: "bf16[3072]", arg2_1: "bf16[8192, 768]"):
        # No stacktrace found for following nodes
        addmm: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        relu: "bf16[8192, 3072]" = torch.ops.aten.relu.default(addmm);  addmm = None
        return relu

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

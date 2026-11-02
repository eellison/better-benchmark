"""Curated MM repro. Pattern hash: ba69b95f1d16."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8192, 3072]", arg1_1: "bf16[768]", arg2_1: "bf16[3072, 768]"):
        # No stacktrace found for following nodes
        relu: "bf16[8192, 3072]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        addmm: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg1_1, relu, arg2_1);  arg1_1 = relu = arg2_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

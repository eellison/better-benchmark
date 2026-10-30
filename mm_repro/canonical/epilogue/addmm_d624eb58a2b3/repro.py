"""Curated MM repro. Pattern hash: d624eb58a2b3."""
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
    def forward(self, arg0_1: "bf16[768, 1000]", arg1_1: "bf16[1000]", arg2_1: "bf16[128, 768]", arg3_1: "bf16[128, 1000]"):
        # No stacktrace found for following nodes
        addmm: "bf16[128, 1000]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        add: "bf16[128, 1000]" = torch.ops.aten.add.Tensor(arg3_1, addmm);  arg3_1 = addmm = None
        div: "bf16[128, 1000]" = torch.ops.aten.div.Tensor(add, 2);  add = None
        return div

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

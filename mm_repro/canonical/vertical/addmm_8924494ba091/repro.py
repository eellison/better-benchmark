"""Curated MM repro. Pattern hash: 8924494ba091."""
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
    def forward(self, arg0_1: "bf16[128, 64, 8, 8]", arg1_1: "bf16[64, 10]", arg2_1: "bf16[10]", _shape_param_0):
        # No stacktrace found for following nodes
        mean: "bf16[128, 64, 1, 1]" = torch.ops.aten.mean.dim(arg0_1, [2, 3], True);  arg0_1 = None
        view: "bf16[128, 64]" = torch.ops.aten.view.default(mean, _shape_param_0);  mean = _shape_param_0 = None
        addmm: "bf16[128, 10]" = torch.ops.aten.addmm.default(arg2_1, view, arg1_1);  arg2_1 = view = arg1_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

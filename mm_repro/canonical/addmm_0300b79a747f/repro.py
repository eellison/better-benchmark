"""Curated MM repro. Pattern hash: 0300b79a747f."""
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
    def forward(self, arg0_1: "bf16[512, 1280, 1, 1]", arg1_1: "bf16[1280, 1000]", arg2_1: "bf16[1000]", _shape_param_0):
        # No stacktrace found for following nodes
        relu: "bf16[512, 1280, 1, 1]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        view: "bf16[512, 1280]" = torch.ops.aten.view.default(relu, _shape_param_0);  relu = _shape_param_0 = None
        addmm: "bf16[512, 1000]" = torch.ops.aten.addmm.default(arg2_1, view, arg1_1);  arg2_1 = view = arg1_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

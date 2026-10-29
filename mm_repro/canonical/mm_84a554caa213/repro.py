"""Curated MM repro. Pattern hash: 84a554caa213."""
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
    def forward(self, arg0_1: "bf16[8192, 240]", arg1_1: "bf16[240, 240]", arg2_1: "bf16[512, 4, 16, 64]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 240]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view_1: "bf16[512, 16, 4, 60]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute: "bf16[512, 4, 16, 60]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        slice_scatter: "bf16[512, 4, 16, 64]" = torch.ops.aten.slice_scatter.default(arg2_1, permute, -1, 0, 60);  arg2_1 = permute = None
        return slice_scatter

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

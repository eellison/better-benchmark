"""Curated MM repro. Pattern hash: fb53a6aa5b33."""
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
    def forward(self, arg0_1: "bf16[384, 384]", arg1_1: "bf16[1, 384]", arg2_1: "bf16[384, 384]", arg3_1: "bf16[384]", arg4_1: "bf16[1, 384]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[1, 384]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view_1: "bf16[1, 1, 6, 64]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute_1: "bf16[1, 6, 1, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        addmm: "bf16[1, 384]" = torch.ops.aten.addmm.default(arg3_1, arg4_1, arg2_1);  arg3_1 = arg4_1 = arg2_1 = None
        view_3: "bf16[1, 1, 6, 64]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        permute_3: "bf16[1, 6, 1, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        return (permute_1, permute_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

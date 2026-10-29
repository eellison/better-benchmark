"""Curated MM repro. Pattern hash: 13ea7485f8b8."""
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
    def forward(self, arg0_1: "bf16[1, 1, 768]", arg1_1: "bf16[768, 768]", arg2_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        expand: "bf16[128, 1, 768]" = torch.ops.aten.expand.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view: "bf16[128, 768]" = torch.ops.aten.view.default(expand, _shape_param_1);  expand = _shape_param_1 = None
        mm: "bf16[128, 768]" = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_1: "bf16[128, 1, 768]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        add: "bf16[128, 1, 768]" = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        view_2: "bf16[128, 1, 12, 64]" = torch.ops.aten.view.default(add, _shape_param_3);  add = _shape_param_3 = None
        permute_1: "bf16[128, 12, 1, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        return permute_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

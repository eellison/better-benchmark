"""Curated MM repro. Pattern hash: 658088d75f6d."""
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096, 4096]", arg2_1: "bf16[8, 512, 4096]", arg3_1: "bf16[8, 512, 4096]", arg4_1: "bf16[8, 512, 4096]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        add: "bf16[8, 512, 4096]" = torch.ops.aten.add.Tensor(arg2_1, view);  arg2_1 = view = None
        add_1: "bf16[8, 512, 4096]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        add_2: "bf16[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_1, arg4_1);  add_1 = arg4_1 = None
        view_1: "bf16[4096, 4096]" = torch.ops.aten.view.default(add_2, _shape_param_1);  add_2 = _shape_param_1 = None
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_1, [1, 0])
        return (view_1, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

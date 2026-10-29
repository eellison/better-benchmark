"""Curated MM repro. Pattern hash: 6a074e57046c."""
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
    def forward(self, arg0_1: "bf16[64, 1000, 64]", arg1_1: "bf16[64, 64, 1000]", arg2_1: "bf16[1, 1, 1000, 1000]", arg3_1: "bf16[1, 64, 1000, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        bmm: "bf16[64, 1000, 1000]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        mul: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.mul.Tensor(view, 0.125);  view = None
        add: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.add.Tensor(mul, arg2_1);  mul = arg2_1 = None
        cat: "bf16[1, 64, 1000, 1001]" = torch.ops.aten.cat.default([add, arg3_1], -1);  add = arg3_1 = None
        return cat

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

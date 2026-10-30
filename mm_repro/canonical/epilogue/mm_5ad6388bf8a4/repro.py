"""Curated MM repro. Pattern hash: 5ad6388bf8a4."""
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
    def forward(self, arg0_1: "f32[768, 768]", arg1_1: "f32[16384, 768]", arg2_1: "f32[768]", arg3_1: "f32[32, 512, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "f32[16384, 768]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg2_1);  arg2_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg3_1);  arg3_1 = None
        return (view, mul, mul_1, mul_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

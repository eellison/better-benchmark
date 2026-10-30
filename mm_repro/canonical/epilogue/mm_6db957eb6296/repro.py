"""Curated MM repro. Pattern hash: 6db957eb6296."""
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[1000, 4096]", arg2_1: "bf16[1, 1000, 128]", arg3_1: "bf16[1, 1000, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[1000, 4096]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view_1: "bf16[1, 1000, 32, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute_1: "bf16[1, 32, 1000, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        unsqueeze: "bf16[1, 1, 1000, 128]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        mul: "bf16[1, 32, 1000, 128]" = torch.ops.aten.mul.Tensor(permute_1, unsqueeze);  unsqueeze = None
        slice_1: "bf16[1, 32, 1000, 64]" = torch.ops.aten.slice.Tensor(permute_1, 3, 64, 9223372036854775807)
        neg: "bf16[1, 32, 1000, 64]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 32, 1000, 64]" = torch.ops.aten.slice.Tensor(permute_1, 3, 0, 64);  permute_1 = None
        cat: "bf16[1, 32, 1000, 128]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        unsqueeze_1: "bf16[1, 1, 1000, 128]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        mul_1: "bf16[1, 32, 1000, 128]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_1);  cat = unsqueeze_1 = None
        add: "bf16[1, 32, 1000, 128]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        return add

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

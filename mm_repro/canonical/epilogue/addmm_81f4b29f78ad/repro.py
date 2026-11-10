"""Curated MM repro. Pattern hash: 81f4b29f78ad."""
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
    def forward(self, arg0_1: "bf16[4096]", arg1_1: "bf16[1000, 2880]", arg2_1: "bf16[2880, 4096]", arg3_1: "bf16[1, 1000, 32]", arg4_1: "bf16[1, 1000, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        addmm: "bf16[1000, 4096]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[1, 1000, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[1, 1000, 64, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[1, 64, 1000, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        split = torch.ops.aten.split.Tensor(permute, 32, -1);  permute = None
        getitem: "bf16[1, 64, 1000, 32]" = split[0]
        getitem_1: "bf16[1, 64, 1000, 32]" = split[1];  split = None
        unsqueeze: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(arg3_1, 1);  arg3_1 = None
        mul: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze)
        unsqueeze_1: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(arg4_1, 1);  arg4_1 = None
        mul_1: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_1)
        sub: "bf16[1, 64, 1000, 32]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        mul_2: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze);  getitem_1 = unsqueeze = None
        mul_3: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_1);  getitem = unsqueeze_1 = None
        add: "bf16[1, 64, 1000, 32]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        cat: "bf16[1, 64, 1000, 64]" = torch.ops.aten.cat.default([sub, add], -1);  sub = add = None
        expand: "bf16[1, 64, 1000, 64]" = torch.ops.aten.expand.default(cat, _shape_param_2);  cat = _shape_param_2 = None
        view_2: "bf16[64, 1000, 64]" = torch.ops.aten.view.default(expand, _shape_param_3);  expand = _shape_param_3 = None
        return view_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

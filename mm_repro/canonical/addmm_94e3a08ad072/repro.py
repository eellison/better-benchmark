"""Curated MM repro. Pattern hash: 94e3a08ad072."""
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
    def forward(self, arg0_1: "bf16[2880, 512]", arg1_1: "bf16[512]", arg2_1: "bf16[1000, 2880]", arg3_1: "bf16[1, 1, 1000, 32]", arg4_1: "bf16[1, 1, 1000, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        addmm: "bf16[1000, 512]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view_1: "bf16[1, 1000, 8, 64]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        split = torch.ops.aten.split.Tensor(permute_1, 32, -1);  permute_1 = None
        getitem: "bf16[1, 8, 1000, 32]" = split[0]
        getitem_1: "bf16[1, 8, 1000, 32]" = split[1];  split = None
        mul: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, arg3_1)
        mul_1: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, arg4_1)
        sub: "bf16[1, 8, 1000, 32]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        mul_2: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, arg3_1);  getitem_1 = arg3_1 = None
        mul_3: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, arg4_1);  getitem = arg4_1 = None
        add: "bf16[1, 8, 1000, 32]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        cat: "bf16[1, 8, 1000, 64]" = torch.ops.aten.cat.default([sub, add], -1);  sub = add = None
        unsqueeze: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.view.default(cat, _shape_param_1);  _shape_param_1 = None
        expand: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_2);  unsqueeze = _shape_param_2 = None
        clone: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1, 64, 1000, 64]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        permute_2: "bf16[1, 64, 64, 1000]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2]);  view_2 = None
        view_3: "bf16[64, 64, 1000]" = torch.ops.aten.view.default(permute_2, _shape_param_4);  permute_2 = _shape_param_4 = None
        slice_1: "bf16[1, 8, 127, 64]" = torch.ops.aten.slice.Tensor(cat, 2, -127, 9223372036854775807);  cat = None
        return (view_3, slice_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

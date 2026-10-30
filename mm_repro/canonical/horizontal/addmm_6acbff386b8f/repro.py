"""Curated MM repro. Pattern hash: 6acbff386b8f."""
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
    def forward(self, arg0_1: "bf16[2880, 512]", arg1_1: "bf16[512]", arg2_1: "bf16[1000, 2880]", arg3_1: "bf16[1, 1000, 32]", arg4_1: "bf16[1, 1000, 32]", arg5_1: "bf16[2880, 512]", arg6_1: "bf16[512]", arg7_1: "bf16[1000, 2880]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        addmm: "bf16[1000, 512]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view_1: "bf16[1, 1000, 8, 64]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        split = torch.ops.aten.split.Tensor(permute_1, 32, -1);  permute_1 = None
        getitem: "bf16[1, 8, 1000, 32]" = split[0]
        getitem_1: "bf16[1, 8, 1000, 32]" = split[1];  split = None
        unsqueeze: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(arg3_1, 1);  arg3_1 = None
        unsqueeze_1: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(arg4_1, 1);  arg4_1 = None
        mul: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze)
        mul_1: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_1)
        sub: "bf16[1, 8, 1000, 32]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        mul_2: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze);  getitem_1 = unsqueeze = None
        mul_3: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_1);  getitem = unsqueeze_1 = None
        add: "bf16[1, 8, 1000, 32]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        cat: "bf16[1, 8, 1000, 64]" = torch.ops.aten.cat.default([sub, add], -1);  sub = add = None
        unsqueeze_2: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.view.default(cat, _shape_param_1);  _shape_param_1 = None
        expand: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_2);  unsqueeze_2 = _shape_param_2 = None
        clone: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1, 64, 1000, 64]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        permute_2: "bf16[1, 64, 64, 1000]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2]);  view_2 = None
        view_3: "bf16[64, 64, 1000]" = torch.ops.aten.view.default(permute_2, _shape_param_4);  permute_2 = _shape_param_4 = None
        addmm_1: "bf16[1000, 512]" = torch.ops.aten.addmm.default(arg6_1, arg7_1, arg5_1);  arg6_1 = arg7_1 = arg5_1 = None
        view_5: "bf16[1, 1000, 8, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_5);  addmm_1 = _shape_param_5 = None
        permute_4: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        unsqueeze_3: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(permute_4, 2)
        expand_2: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_3, _shape_param_6);  unsqueeze_3 = _shape_param_6 = None
        clone_1: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_7: "bf16[64, 1000, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        return (cat, view_3, permute_4, view_7)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

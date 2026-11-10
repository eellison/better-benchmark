"""Curated MM repro. Pattern hash: cdc7c2f6dc5e."""
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
    def forward(self, arg0_1: "bf16[512, 16, 1024]", arg1_1: "bf16[1024, 16, 64]", arg2_1: "bf16[16, 64]", arg3_1: "bf16[1024, 16, 64]", arg4_1: "bf16[16, 64]", arg5_1: "bf16[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15):
        # No stacktrace found for following nodes
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  _shape_param_0 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_1);  squeeze = squeeze_1 = None
        view_2: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        permute: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute, _shape_param_3);  permute = _shape_param_3 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg2_1);  arg2_1 = None
        unsqueeze_5: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        permute_1: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_5, [1, 2, 0, 4, 3]);  unsqueeze_5 = None
        squeeze_2: "bf16[8192, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_5);  _shape_param_5 = None
        squeeze_3: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_6);  arg3_1 = _shape_param_6 = None
        mm_1: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_2, squeeze_3);  squeeze_2 = squeeze_3 = None
        view_6: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm_1, _shape_param_7);  mm_1 = _shape_param_7 = None
        permute_2: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 4, 2]);  view_6 = None
        view_7: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_2, _shape_param_8);  permute_2 = _shape_param_8 = None
        unsqueeze_11: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(view_7, _shape_param_9);  view_7 = _shape_param_9 = None
        permute_3: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_11, [1, 2, 4, 0, 3]);  unsqueeze_11 = None
        add_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg4_1);  view_3 = arg4_1 = None
        unsqueeze_12: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add_1, _shape_param_10);  add_1 = _shape_param_10 = None
        squeeze_4: "bf16[8192, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_11);  arg0_1 = _shape_param_11 = None
        squeeze_5: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg5_1, _shape_param_12);  arg5_1 = _shape_param_12 = None
        mm_2: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_5);  squeeze_4 = squeeze_5 = None
        view_10: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm_2, _shape_param_13);  mm_2 = _shape_param_13 = None
        permute_4: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_10, [0, 1, 3, 4, 2]);  view_10 = None
        view_11: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_4, _shape_param_14);  permute_4 = _shape_param_14 = None
        unsqueeze_18: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(view_11, _shape_param_15);  view_11 = _shape_param_15 = None
        permute_5: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_18, [4, 1, 2, 3, 0]);  unsqueeze_18 = None
        return (permute_1, permute_3, unsqueeze_12, permute_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

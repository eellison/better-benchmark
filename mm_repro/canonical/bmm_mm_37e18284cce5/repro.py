"""Curated MM repro. Pattern hash: 37e18284cce5."""
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
    def forward(self, arg0_1: "bf16[1024, 16, 64]", arg1_1: "bf16[8192, 1024]", arg2_1: "bf16[16, 64]", arg3_1: "bf16[1024, 16, 64]", arg4_1: "bf16[8192, 1024]", arg5_1: "bf16[16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        squeeze: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg1_1, squeeze);  arg1_1 = squeeze = None
        view_1: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        permute: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 4, 2]);  view_1 = None
        view_2: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_2, arg2_1);  arg2_1 = None
        unsqueeze_3: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add, _shape_param_3);  add = _shape_param_3 = None
        permute_1: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_3, [1, 2, 0, 4, 3]);  unsqueeze_3 = None
        permute_2: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1, [0, 1, 2, 4, 3]);  permute_1 = None
        view_3: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_2, _shape_param_4);  permute_2 = _shape_param_4 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_5);  arg3_1 = _shape_param_5 = None
        mm_1: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg4_1, squeeze_1);  arg4_1 = squeeze_1 = None
        view_5: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm_1, _shape_param_6);  mm_1 = _shape_param_6 = None
        permute_3: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_5, [0, 1, 3, 4, 2]);  view_5 = None
        view_6: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_7);  permute_3 = _shape_param_7 = None
        unsqueeze_7: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(view_6, _shape_param_8);  view_6 = _shape_param_8 = None
        permute_4: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_7, [1, 2, 4, 0, 3]);  unsqueeze_7 = None
        permute_5: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_4, [0, 1, 4, 3, 2]);  permute_4 = None
        view_7: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_5, _shape_param_9);  permute_5 = _shape_param_9 = None
        bmm: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_3, view_7);  view_3 = view_7 = None
        view_8: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm, _shape_param_10);  bmm = _shape_param_10 = None
        permute_6: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_8, [0, 1, 2, 4, 3]);  view_8 = None
        view_9: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_6, _shape_param_11);  permute_6 = _shape_param_11 = None
        add_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_2, arg5_1);  view_2 = arg5_1 = None
        unsqueeze_8: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add_1, _shape_param_12);  add_1 = _shape_param_12 = None
        permute_7: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_8, [1, 2, 0, 4, 3]);  unsqueeze_8 = None
        permute_8: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_7, [0, 1, 2, 4, 3]);  permute_7 = None
        view_10: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_8, _shape_param_13);  permute_8 = _shape_param_13 = None
        return (view_9, view_10)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

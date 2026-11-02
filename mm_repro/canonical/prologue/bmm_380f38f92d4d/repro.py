"""Curated MM repro. Pattern hash: 380f38f92d4d."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "bf16[16, 64]", arg2_1: "bf16[256, 64, 512]", arg3_1: "bf16[16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        unsqueeze: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        view: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        permute: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view, [0, 1, 3, 4, 2]);  view = None
        view_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_1, arg1_1);  arg1_1 = None
        unsqueeze_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add, 4);  add = None
        permute_1: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_1, [1, 2, 0, 4, 3]);  unsqueeze_1 = None
        permute_2: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1, [0, 1, 2, 4, 3]);  permute_1 = None
        view_2: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_2, _shape_param_2);  permute_2 = _shape_param_2 = None
        bmm: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_2, arg2_1);  view_2 = arg2_1 = None
        add_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_1, arg3_1);  view_1 = arg3_1 = None
        unsqueeze_2: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_1, 4);  add_1 = None
        permute_3: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_2, [1, 2, 0, 4, 3]);  unsqueeze_2 = None
        permute_4: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_3, [0, 1, 2, 4, 3]);  permute_3 = None
        view_3: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_4, _shape_param_3);  permute_4 = _shape_param_3 = None
        return (bmm, view_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

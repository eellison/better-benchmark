"""Curated MM repro. Pattern hash: ec99bb3e35c4."""
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
    def forward(self, arg0_1: "bf16[256, 512, 512]", arg1_1: "bf16[256, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view, [2, 0, 1, 4, 3]);  view = None
        view_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        unsqueeze: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_1, 4);  view_1 = None
        permute_1: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze, [0, 1, 4, 3, 2]);  unsqueeze = None
        permute_2: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 4, 2]);  permute_1 = None
        clone: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_2: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_2, 0);  view_2 = None
        return squeeze

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

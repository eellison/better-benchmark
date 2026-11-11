"""Curated MM repro. Pattern hash: 3c28b5fd9892."""
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
    def forward(self, arg0_1: "bf16[98304, 64, 9]", arg1_1: "bf16[98304, 9, 1]", arg2_1: "bf16[32, 6, 512, 64]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        bmm: "bf16[98304, 64, 1]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[16384, 384]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        clone: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        cat: "bf16[32, 512, 12, 64]" = torch.ops.aten.cat.default([clone, view_1], 2);  clone = view_1 = None
        return cat

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

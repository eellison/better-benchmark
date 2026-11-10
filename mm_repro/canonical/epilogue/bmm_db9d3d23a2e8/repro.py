"""Curated MM repro. Pattern hash: db9d3d23a2e8."""
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
    def forward(self, arg0_1: "bf16[32, 6, 512, 64]", arg1_1: "bf16[98304, 64, 9]", arg2_1: "bf16[98304, 9, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        permute: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        clone: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        bmm: "bf16[98304, 64, 1]" = torch.ops.aten.bmm.default(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        view: "bf16[16384, 384]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        view_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        cat: "bf16[32, 512, 12, 64]" = torch.ops.aten.cat.default([clone, view_1], 2);  clone = view_1 = None
        view_2: "bf16[32, 512, 768]" = torch.ops.aten.view.default(cat, _shape_param_2);  cat = _shape_param_2 = None
        view_3: "bf16[16384, 768]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        return view_3

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

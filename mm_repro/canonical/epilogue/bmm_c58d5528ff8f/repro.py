"""Curated MM repro. Pattern hash: c58d5528ff8f."""
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
    def forward(self, arg0_1: "bf16[768, 49, 49]", arg1_1: "bf16[768, 49, 128]", arg2_1: "bf16[128, 6, 49, 128]", arg3_1: "bf16[128, 6, 49, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[768, 49, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 6, 49, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        cat: "bf16[384, 6, 49, 128]" = torch.ops.aten.cat.default([view, arg2_1, arg3_1]);  view = arg2_1 = arg3_1 = None
        view_1: "bf16[3, 128, 6, 49, 128]" = torch.ops.aten.view.default(cat, _shape_param_1);  cat = _shape_param_1 = None
        permute: "bf16[128, 3, 6, 128, 49]" = torch.ops.aten.permute.default(view_1, [1, 0, 2, 4, 3]);  view_1 = None
        clone: "bf16[128, 3, 6, 128, 49]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[128, 2304, 7, 7]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        return view_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

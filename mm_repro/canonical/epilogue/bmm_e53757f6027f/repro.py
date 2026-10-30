"""Curated MM repro. Pattern hash: e53757f6027f."""
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
    def forward(self, arg0_1: "bf16[4096, 49, 49]", arg1_1: "bf16[4096, 49, 32]", arg2_1: "bf16[128, 32, 49, 32]", arg3_1: "bf16[128, 32, 49, 32]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[4096, 49, 32]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        cat: "bf16[384, 32, 49, 32]" = torch.ops.aten.cat.default([arg2_1, arg3_1, view]);  arg2_1 = arg3_1 = view = None
        view_1: "bf16[3, 128, 32, 49, 32]" = torch.ops.aten.view.default(cat, _shape_param_1);  cat = _shape_param_1 = None
        permute: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.permute.default(view_1, [1, 3, 0, 2, 4]);  view_1 = None
        clone: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "bf16[6272, 3072]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_1: "bf16[3072, 6272]" = torch.ops.aten.permute.default(view_3, [1, 0])
        return (view_3, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

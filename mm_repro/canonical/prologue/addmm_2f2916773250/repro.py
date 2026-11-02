"""Curated MM repro. Pattern hash: 2f2916773250."""
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
    def forward(self, arg0_1: "bf16[512, 4, 256, 40]", arg1_1: "bf16[144]", arg2_1: "bf16[144, 144]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        slice_1: "bf16[512, 4, 256, 36]" = torch.ops.aten.slice.Tensor(arg0_1, -1, 0, 36);  arg0_1 = None
        permute: "bf16[512, 256, 4, 36]" = torch.ops.aten.permute.default(slice_1, [0, 2, 1, 3]);  slice_1 = None
        clone: "bf16[512, 256, 4, 36]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view: "bf16[512, 256, 144]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        view_1: "bf16[131072, 144]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        addmm: "bf16[131072, 144]" = torch.ops.aten.addmm.default(arg1_1, view_1, arg2_1);  arg1_1 = arg2_1 = None
        return (view_1, addmm)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

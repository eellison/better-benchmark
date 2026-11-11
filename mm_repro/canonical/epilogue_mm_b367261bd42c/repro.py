"""Curated MM repro. Pattern hash: b367261bd42c."""
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
    def forward(self, arg0_1: "bf16[16384, 768]", arg1_1: "bf16[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        slice_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_1, 2, 0, 6)
        slice_2: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_1, 2, 6, 12);  view_1 = None
        view_2: "bf16[16384, 384]" = torch.ops.aten.view.default(slice_2, _shape_param_2);  slice_2 = _shape_param_2 = None
        permute: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(slice_1, [0, 2, 1, 3]);  slice_1 = None
        clone: "bf16[32, 6, 512, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        clone_1: "bf16[16384, 384]" = torch.ops.aten.clone.default(view_2, memory_format = torch.contiguous_format);  view_2 = None
        return (clone, clone_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

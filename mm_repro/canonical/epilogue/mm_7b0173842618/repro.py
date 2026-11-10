"""Curated MM repro. Pattern hash: 7b0173842618."""
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
    def forward(self, arg0_1: "bf16[8192, 768]", arg1_1: "bf16[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(view, [1, 0, 2]);  view = None
        view_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        permute_1: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1, [1, 0, 2, 3]);  view_1 = None
        permute_2: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1, [0, 2, 1, 3]);  permute_1 = None
        clone: "bf16[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_2: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        view_3: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        permute_3: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_3, [0, 1, 2, 4, 3]);  view_3 = None
        view_4: "bf16[384, 256, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_4);  permute_3 = _shape_param_4 = None
        return view_4

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

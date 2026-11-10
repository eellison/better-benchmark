"""Curated MM repro. Pattern hash: f29e6ef92666."""
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
    def forward(self, arg0_1: "bf16[4096, 512]", arg1_1: "bf16[512, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 384]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        permute_1: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute, [0, 1, 3, 2]);  permute = None
        expand: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_1, _shape_param_2);  permute_1 = _shape_param_2 = None
        clone: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        permute_2: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1])
        return (view_2, permute_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

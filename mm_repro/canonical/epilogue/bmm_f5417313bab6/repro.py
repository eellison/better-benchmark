"""Curated MM repro. Pattern hash: f5417313bab6."""
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
    def forward(self, arg0_1: "bf16[288, 64, 512]", arg1_1: "bf16[288, 512, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        bmm: "bf16[288, 64, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view, [0, 1, 4, 3, 2]);  view = None
        permute_1: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute, [0, 1, 3, 4, 2]);  permute = None
        squeeze: "bf16[96, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1, 4);  permute_1 = None
        clone: "bf16[96, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        view_1: "bf16[9437184]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        return view_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

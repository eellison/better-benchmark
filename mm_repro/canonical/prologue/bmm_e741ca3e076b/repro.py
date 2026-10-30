"""Curated MM repro. Pattern hash: e741ca3e076b."""
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
    def forward(self, arg0_1: "bf16[16, 16, 512, 1023]", arg1_1: "bf16[16, 16, 1024, 512]", arg2_1: "bf16[256, 64, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[16, 16, 1023, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        slice_scatter: "bf16[16, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(arg1_1, view, 2, 1, 9223372036854775807);  arg1_1 = view = None
        view_2: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.view.default(slice_scatter, _shape_param_1);  slice_scatter = _shape_param_1 = None
        permute: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.permute.default(view_2, [0, 1, 2, 4, 3]);  view_2 = None
        view_3: "bf16[256, 512, 1024]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        bmm: "bf16[256, 64, 1024]" = torch.ops.aten.bmm.default(arg2_1, view_3);  arg2_1 = view_3 = None
        view_4: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.view.default(bmm, _shape_param_3);  bmm = _shape_param_3 = None
        permute_1: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 1, 4, 3, 2]);  view_4 = None
        permute_2: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1, [3, 0, 1, 4, 2]);  permute_1 = None
        squeeze: "bf16[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2, 4);  permute_2 = None
        view_5: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze, _shape_param_4);  squeeze = _shape_param_4 = None
        permute_3: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_5, [0, 1, 4, 2, 3]);  view_5 = None
        clone: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        squeeze_1: "bf16[16384, 1024]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        return squeeze_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

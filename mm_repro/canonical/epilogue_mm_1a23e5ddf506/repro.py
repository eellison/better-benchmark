"""Curated MM repro. Pattern hash: 1a23e5ddf506."""
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
    def forward(self, arg0_1: "bf16[4096, 2048]", arg1_1: "bf16[2048, 2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 2048]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        convert_element_type: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        permute_1: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type, [0, 1, 3, 2]);  convert_element_type = None
        expand: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_1, _shape_param_2);  permute_1 = _shape_param_2 = None
        clone: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        return clone

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

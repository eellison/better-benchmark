"""Curated MM repro. Pattern hash: ebaf65e5eace."""
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
    def forward(self, arg0_1: "bf16[128, 12, 256, 64]", arg1_1: "bf16[128, 12, 256, 64]", arg2_1: "bf16[32768, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        cat: "bf16[256, 12, 256, 64]" = torch.ops.aten.cat.default([arg0_1, arg1_1]);  arg0_1 = arg1_1 = None
        view: "bf16[2, 128, 12, 256, 64]" = torch.ops.aten.view.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        permute: "bf16[128, 256, 2, 12, 64]" = torch.ops.aten.permute.default(view, [1, 3, 0, 2, 4]);  view = None
        clone: "bf16[128, 256, 2, 12, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[32768, 1536]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        permute_1: "bf16[1536, 32768]" = torch.ops.aten.permute.default(view_2, [1, 0]);  view_2 = None
        mm: "bf16[1536, 768]" = torch.ops.aten.mm.default(permute_1, arg2_1);  permute_1 = arg2_1 = None
        convert_element_type: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

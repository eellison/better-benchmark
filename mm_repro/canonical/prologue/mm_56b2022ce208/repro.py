"""Curated MM repro. Pattern hash: 56b2022ce208."""
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
    def forward(self, arg0_1: "f32[8, 1024, 768]", arg1_1: "bf16[768, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        permute: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2]);  arg0_1 = None
        convert_element_type: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute, torch.bfloat16);  permute = None
        clone: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type, memory_format = torch.contiguous_format);  convert_element_type = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(view, arg1_1);  arg1_1 = None
        return (view, mm)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

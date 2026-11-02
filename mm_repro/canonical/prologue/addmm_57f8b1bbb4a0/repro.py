"""Curated MM repro. Pattern hash: 57f8b1bbb4a0."""
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
    def forward(self, arg0_1: "f32[1, 1, 768]", arg1_1: "bf16[768]", arg2_1: "bf16[768, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        expand: "f32[128, 1, 768]" = torch.ops.aten.expand.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[128, 1, 768]" = torch.ops.prims.convert_element_type.default(expand, torch.bfloat16);  expand = None
        view: "bf16[128, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        addmm: "bf16[128, 768]" = torch.ops.aten.addmm.default(arg1_1, view, arg2_1);  arg1_1 = arg2_1 = None
        return (view, addmm)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

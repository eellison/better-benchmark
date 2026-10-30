"""Curated MM repro. Pattern hash: 41b6b1a7d236."""
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
    def forward(self, arg0_1: "bf16[128, 184, 4, 4]", arg1_1: "f32[10]", arg2_1: "f32[184, 10]", _shape_param_0):
        # No stacktrace found for following nodes
        mean: "bf16[128, 184, 1, 1]" = torch.ops.aten.mean.dim(arg0_1, [2, 3], True);  arg0_1 = None
        view: "bf16[128, 184]" = torch.ops.aten.view.default(mean, _shape_param_0);  mean = _shape_param_0 = None
        convert_element_type: "bf16[10]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[184, 10]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        addmm: "bf16[128, 10]" = torch.ops.aten.addmm.default(convert_element_type, view, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        return (view, addmm)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

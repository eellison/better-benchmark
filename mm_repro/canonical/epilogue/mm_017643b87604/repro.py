"""Curated MM repro. Pattern hash: 017643b87604."""
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
    def forward(self, arg0_1: "f32[2560, 8008]", arg1_1: "f32[16, 128, 2560]", arg2_1: "f32[1, 8008]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[2560, 8008]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        view: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        mm: "bf16[2048, 8008]" = torch.ops.aten.mm.default(view, convert_element_type);  view = convert_element_type = None
        view_1: "bf16[16, 128, 8008]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        add: "f32[16, 128, 8008]" = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        view_2: "f32[2048, 8008]" = torch.ops.aten.view.default(add, _shape_param_2);  _shape_param_2 = None
        return (add, view_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

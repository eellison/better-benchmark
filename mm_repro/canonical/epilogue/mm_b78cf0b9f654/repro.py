"""Curated MM repro. Pattern hash: b78cf0b9f654."""
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
    def forward(self, arg0_1: "f32[54]", arg1_1: "f32[384, 54]", arg2_1: "bf16[16384, 384]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[54]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[384, 54]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mm: "bf16[16384, 54]" = torch.ops.aten.mm.default(arg2_1, convert_element_type_1);  arg2_1 = convert_element_type_1 = None
        view: "bf16[32, 512, 54]" = torch.ops.aten.view.default(mm, _shape_param_0);  _shape_param_0 = None
        add: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view, convert_element_type);  view = convert_element_type = None
        view_1: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type_2: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        return (mm, convert_element_type_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

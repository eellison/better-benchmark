"""Curated MM repro. Pattern hash: b956940dad46."""
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
    def forward(self, arg0_1: "f32[768]", arg1_1: "f32[3072, 768]", arg2_1: "bf16[128, 3072]", arg3_1: "bf16[128, 1, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[128, 768]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        view: "bf16[128, 1, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        add: "bf16[128, 1, 768]" = torch.ops.aten.add.Tensor(arg3_1, view);  arg3_1 = view = None
        select: "bf16[128, 768]" = torch.ops.aten.select.int(add, 1, 0);  add = None
        return select

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

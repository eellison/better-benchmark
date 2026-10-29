"""Curated MM repro. Pattern hash: 446628236e20."""
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
    def forward(self, arg0_1: "bf16[2560, 8008]", arg1_1: "bf16[2048, 2560]", arg2_1: "bf16[1, 8008]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[2048, 8008]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view: "bf16[16, 128, 8008]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        add: "bf16[16, 128, 8008]" = torch.ops.aten.add.Tensor(view, arg2_1);  view = arg2_1 = None
        view_1: "bf16[2048, 8008]" = torch.ops.aten.view.default(add, _shape_param_1);  _shape_param_1 = None
        convert_element_type: "f32[2048, 8008]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        return (add, convert_element_type)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

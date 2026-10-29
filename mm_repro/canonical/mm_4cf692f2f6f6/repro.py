"""Curated MM repro. Pattern hash: 4cf692f2f6f6."""
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
    def forward(self, arg0_1: "bf16[1, 1000, 16, 128]", arg1_1: "bf16[2048, 1024]", arg2_1: "bf16[1, 1000, 1024]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_1: "bf16[1000, 2048]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mm: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view_1, arg1_1);  view_1 = arg1_1 = None
        view_2: "bf16[1, 1000, 1024]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        add: "bf16[1, 1000, 1024]" = torch.ops.aten.add.Tensor(arg2_1, view_2);  arg2_1 = view_2 = None
        convert_element_type: "f32[1, 1000, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        return (add, convert_element_type)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

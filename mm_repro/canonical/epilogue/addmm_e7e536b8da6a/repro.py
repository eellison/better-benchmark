"""Curated MM repro. Pattern hash: e7e536b8da6a."""
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
    def forward(self, arg0_1: "bf16[16384, 4096]", arg1_1: "bf16[4096]", arg2_1: "bf16[128, 16384]", arg3_1: "bf16[1, 128, 4096]", arg4_1: "bf16[1, 128, 4096]", _shape_param_0):
        # No stacktrace found for following nodes
        addmm: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        add: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(arg3_1, view);  arg3_1 = view = None
        add_1: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add, arg4_1);  add = arg4_1 = None
        convert_element_type: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        return (add_1, convert_element_type)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: 7b27746a2f36."""
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
    def forward(self, arg0_1: "f32[4096, 4096]", arg1_1: "bf16[128, 4096]", arg2_1: "bf16[4096]", arg3_1: "bf16[128, 16384]", arg4_1: "bf16[16384, 4096]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        addmm: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg2_1, arg3_1, arg4_1);  arg2_1 = arg3_1 = arg4_1 = None
        view_1: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        add: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view, view_1);  view = view_1 = None
        return add

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

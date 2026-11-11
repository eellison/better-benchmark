"""Curated MM repro. Pattern hash: 75a3c82cceee."""
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
    def forward(self, arg0_1: "bf16[8, 1024, 1024]", arg1_1: "bf16[1024, 1024]", arg2_1: "bf16[1024]", arg3_1: "bf16[1024, 1024]", arg4_1: "bf16[1024]", arg5_1: "bf16[1024, 1024]", arg6_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  _shape_param_0 = None
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg2_1, view, arg1_1);  arg2_1 = view = arg1_1 = None
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_1);  _shape_param_1 = None
        addmm_1: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg4_1, view_1, arg3_1);  arg4_1 = view_1 = arg3_1 = None
        view_2: "bf16[8192, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_2);  arg0_1 = _shape_param_2 = None
        addmm_2: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg6_1, view_2, arg5_1);  arg6_1 = view_2 = arg5_1 = None
        return (addmm, addmm_1, addmm_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

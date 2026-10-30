"""Curated MM repro. Pattern hash: 5738d00e2a4c."""
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
    def forward(self, arg0_1: "bf16[1024, 1024]", arg1_1: "bf16[1024]", arg2_1: "bf16[8192, 1024]", arg3_1: "bf16[1024, 1024]", arg4_1: "bf16[1024]", arg5_1: "bf16[8192, 1024]", arg6_1: "bf16[1024, 1024]", arg7_1: "bf16[1024]", arg8_1: "bf16[8192, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view_1: "bf16[8, 1024, 16, 64]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        addmm_1: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg4_1, arg5_1, arg3_1);  arg4_1 = arg5_1 = arg3_1 = None
        view_3: "bf16[8, 1024, 16, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_1);  addmm_1 = _shape_param_1 = None
        permute_3: "bf16[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        addmm_2: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg7_1, arg8_1, arg6_1);  arg7_1 = arg8_1 = arg6_1 = None
        view_5: "bf16[8, 1024, 16, 64]" = torch.ops.aten.view.default(addmm_2, _shape_param_2);  addmm_2 = _shape_param_2 = None
        permute_5: "bf16[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        return (permute_1, permute_3, permute_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: d0266136a52f."""
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
    def forward(self, arg0_1: "bf16[4096]", arg1_1: "bf16[8192, 1024]", arg2_1: "bf16[1024, 4096]", _shape_param_0):
        # No stacktrace found for following nodes
        addmm: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[64, 128, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        relu: "bf16[64, 128, 4096]" = torch.ops.aten.relu.default(view);  view = None
        le: "b8[64, 128, 4096]" = torch.ops.aten.le.Scalar(relu, 0)
        return (relu, le)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""Curated MM repro. Pattern hash: 539cbbb593f7."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[512, 1280, 1, 1]", arg1_1: "bf16[1000]", arg2_1: "bf16[1280, 1000]", _shape_param_0):
        # No stacktrace found for following nodes
        relu: "bf16[512, 1280, 1, 1]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        view: "bf16[512, 1280]" = torch.ops.aten.view.default(relu, _shape_param_0);  _shape_param_0 = None
        addmm: "bf16[512, 1000]" = torch.ops.aten.addmm.default(arg1_1, view, arg2_1);  arg1_1 = arg2_1 = None
        le: "b8[512, 1280, 1, 1]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (view, addmm, le)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

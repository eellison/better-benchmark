"""Curated MM repro. Pattern hash: 035bd65034b0."""
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
    def forward(self, arg0_1: "bf16[8192, 2048]", arg1_1: "bf16[2048, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        relu: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view);  view = None
        view_1: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu, _shape_param_1);  relu = _shape_param_1 = None
        mm: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_1, arg1_1);  view_1 = arg1_1 = None
        return mm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

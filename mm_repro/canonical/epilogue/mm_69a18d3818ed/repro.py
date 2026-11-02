"""Curated MM repro. Pattern hash: 69a18d3818ed."""
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
    def forward(self, arg0_1: "bf16[1024, 8192]", arg1_1: "bf16[8192, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        unsqueeze: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        squeeze: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view, 4);  view = None
        squeeze_1: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze, 3);  squeeze = None
        convert_element_type: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        return convert_element_type

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

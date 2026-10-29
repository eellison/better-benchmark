"""Curated MM repro. Pattern hash: 89b198dde06c."""
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
    def forward(self, arg0_1: "bf16[1, 1500, 6, 64]", arg1_1: "bf16[384, 384]", arg2_1: "bf16[384]", arg3_1: "bf16[1, 1500, 384]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        clone: "bf16[1, 1500, 6, 64]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view_1: "bf16[1500, 384]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        addmm: "bf16[1500, 384]" = torch.ops.aten.addmm.default(arg2_1, view_1, arg1_1);  arg2_1 = view_1 = arg1_1 = None
        view_2: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        add: "bf16[1, 1500, 384]" = torch.ops.aten.add.Tensor(arg3_1, view_2);  arg3_1 = view_2 = None
        clone_1: "bf16[1, 1500, 384]" = torch.ops.aten.clone.default(add, memory_format = torch.contiguous_format)
        return (add, clone_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: 78279de31791."""
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
    def forward(self, arg0_1: "bf16[512, 128]", arg1_1: "bf16[128]", arg2_1: "bf16[32768, 512]", arg3_1: "bf16[128]", arg4_1: "bf16[128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        mul: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view, arg3_1);  view = arg3_1 = None
        add: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul, arg4_1);  mul = arg4_1 = None
        view_1: "bf16[32768, 128]" = torch.ops.aten.view.default(add, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[32768, 128]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        return (view_1, view_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

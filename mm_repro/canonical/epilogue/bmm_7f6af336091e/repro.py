"""Curated MM repro. Pattern hash: 7f6af336091e."""
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
    def forward(self, arg0_1: "f32[16, 128, 256]", arg1_1: "f32[16, 256, 128]", arg2_1: "bf16[1, 1, 128, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        div: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view, 16.0);  view = None
        add: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div, arg2_1);  div = arg2_1 = None
        return add

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

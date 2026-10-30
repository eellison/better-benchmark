"""Curated MM repro. Pattern hash: d1f867c0ec81."""
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
    def forward(self, arg0_1: "f32[98304, 9, 1]", arg1_1: "bf16[98304, 64, 9]", _shape_param_0):
        # No stacktrace found for following nodes
        amax: "f32[98304, 1, 1]" = torch.ops.aten.amax.default(arg0_1, [1], True)
        sub: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(arg0_1, amax);  arg0_1 = None
        exp: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        bmm: "bf16[98304, 64, 1]" = torch.ops.aten.bmm.default(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        return (amax, sum_1, view_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: d5c06549bc59."""
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
    def forward(self, arg0_1: "bf16[384, 54]", arg1_1: "bf16[16384, 384]", arg2_1: "bf16[54]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[16384, 54]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view: "bf16[32, 512, 54]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        add: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view, arg2_1);  view = arg2_1 = None
        view_1: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        amax: "f32[98304, 1, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        return convert_element_type_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

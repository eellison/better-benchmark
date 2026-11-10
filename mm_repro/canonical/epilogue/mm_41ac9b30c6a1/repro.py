"""Curated MM repro. Pattern hash: 41ac9b30c6a1."""
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
    def forward(self, arg0_1: "f32[54]", arg1_1: "bf16[16384, 384]", arg2_1: "bf16[384, 54]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[54]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mm: "bf16[16384, 54]" = torch.ops.aten.mm.default(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        view: "bf16[32, 512, 54]" = torch.ops.aten.view.default(mm, _shape_param_0);  _shape_param_0 = None
        add: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view, convert_element_type);  view = convert_element_type = None
        view_1: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type_1: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        amax: "f32[98304, 1, 1]" = torch.ops.aten.amax.default(convert_element_type_1, [1], True)
        sub: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_1, amax);  convert_element_type_1 = None
        exp: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_2: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand: "bf16[98304, 9, 1]" = torch.ops.aten.expand.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        return (mm, amax, sum_1, expand)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

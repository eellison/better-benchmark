"""Curated MM repro. Pattern hash: 11d22aeb6ef5."""
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
    def forward(self, arg0_1: "bf16[1024, 256, 64]", arg1_1: "bf16[1024, 64, 256]", arg2_1: "bf16[64, 1, 256, 256]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        bmm: "bf16[1024, 256, 256]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[64, 16, 256, 256]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        add: "bf16[64, 16, 256, 256]" = torch.ops.aten.add.Tensor(view, arg2_1);  view = arg2_1 = None
        view_1: "bf16[1024, 256, 256]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type: "f32[1024, 256, 256]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        amax: "f32[1024, 256, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[1024, 256, 256]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[1024, 256, 256]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1024, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1024, 256, 256]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[1024, 256, 256]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        return convert_element_type_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""Curated MM repro. Pattern hash: ad442e3a4df8."""
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
    def forward(self, arg0_1: "bf16[1024, 3072]", arg1_1: "bf16[1000, 1024]", arg2_1: "bf16[1, 1000, 3072]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[1000, 3072]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view: "bf16[1, 1000, 3072]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[1, 1000, 3072]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        neg: "f32[1, 1000, 3072]" = torch.ops.aten.neg.default(convert_element_type)
        exp: "f32[1, 1000, 3072]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[1, 1000, 3072]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[1, 1000, 3072]" = torch.ops.aten.div.Tensor(convert_element_type, add);  convert_element_type = add = None
        convert_element_type_1: "bf16[1, 1000, 3072]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul: "bf16[1, 1000, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, arg2_1);  convert_element_type_1 = arg2_1 = None
        view_1: "bf16[1000, 3072]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        return view_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

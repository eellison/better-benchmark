"""Curated MM repro. Pattern hash: c4282ba906e5."""
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
    def forward(self, arg0_1: "bf16[1, 1000, 1024]", arg1_1: "bf16[1024, 3072]", arg2_1: "bf16[1024, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "bf16[1000, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  _shape_param_0 = None
        mm: "bf16[1000, 3072]" = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_1: "bf16[1, 1000, 3072]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type: "f32[1, 1000, 3072]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        neg: "f32[1, 1000, 3072]" = torch.ops.aten.neg.default(convert_element_type)
        exp: "f32[1, 1000, 3072]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[1, 1000, 3072]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[1, 1000, 3072]" = torch.ops.aten.div.Tensor(convert_element_type, add);  convert_element_type = add = None
        convert_element_type_1: "bf16[1, 1000, 3072]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view_2: "bf16[1000, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_2);  arg0_1 = _shape_param_2 = None
        mm_1: "bf16[1000, 3072]" = torch.ops.aten.mm.default(view_2, arg2_1);  view_2 = arg2_1 = None
        view_3: "bf16[1, 1000, 3072]" = torch.ops.aten.view.default(mm_1, _shape_param_3);  mm_1 = _shape_param_3 = None
        mul: "bf16[1, 1000, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, view_3);  convert_element_type_1 = view_3 = None
        view_4: "bf16[1000, 3072]" = torch.ops.aten.view.default(mul, _shape_param_4);  mul = _shape_param_4 = None
        return view_4

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

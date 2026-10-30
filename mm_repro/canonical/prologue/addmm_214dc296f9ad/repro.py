"""Curated MM repro. Pattern hash: 214dc296f9ad."""
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
    def forward(self, arg0_1: "bf16[512, 1280, 1, 1]", arg1_1: "f32[1000]", arg2_1: "f32[1280, 1000]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[512, 1280, 1, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        add: "f32[512, 1280, 1, 1]" = torch.ops.aten.add.Tensor(convert_element_type, 3)
        clamp_min: "f32[512, 1280, 1, 1]" = torch.ops.aten.clamp_min.default(add, 0);  add = None
        clamp_max: "f32[512, 1280, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul: "f32[512, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type, clamp_max);  convert_element_type = clamp_max = None
        div: "f32[512, 1280, 1, 1]" = torch.ops.aten.div.Tensor(mul, 6);  mul = None
        convert_element_type_1: "bf16[512, 1280, 1, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        convert_element_type_2: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_3: "bf16[1280, 1000]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view: "bf16[512, 1280]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        addmm: "bf16[512, 1000]" = torch.ops.aten.addmm.default(convert_element_type_2, view, convert_element_type_3);  convert_element_type_2 = view = convert_element_type_3 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

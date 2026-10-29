"""Curated MM repro. Pattern hash: 5950fa4ea097."""
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
    def forward(self, arg0_1: "bf16[4, 2048, 12, 64]", arg1_1: "f32[768]", arg2_1: "f32[768, 768]", arg3_1: "b8[4, 2048, 768]", arg4_1: "f32[4, 2048, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view_1: "bf16[8192, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        addmm: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type, view_1, convert_element_type_1);  convert_element_type = view_1 = convert_element_type_1 = None
        view_2: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        mul: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(arg3_1, view_2);  arg3_1 = view_2 = None
        mul_1: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(arg4_1, mul_1);  arg4_1 = mul_1 = None
        view_3: "f32[8192, 768]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        return view_3

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

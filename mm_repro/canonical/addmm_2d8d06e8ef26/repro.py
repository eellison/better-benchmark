"""Curated MM repro. Pattern hash: 2d8d06e8ef26."""
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
    def forward(self, arg0_1: "f32[768, 768]", arg1_1: "f32[768]", arg2_1: "f32[16384, 768]", arg3_1: "b8[32, 512, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        addmm: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view: "f32[32, 512, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg3_1, view);  arg3_1 = view = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        convert_element_type: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_1, torch.complex64)
        return (mul_1, convert_element_type)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

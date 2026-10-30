"""Curated MM repro. Pattern hash: 2824e74817d0."""
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
    def forward(self, arg0_1: "bf16[384, 512, 64]", arg1_1: "bf16[384, 64, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        bmm: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  _shape_param_0 = None
        mul: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view, 0.125)
        convert_element_type: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_1: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul_1: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        abs_1: "f32[32, 12, 512, 512]" = torch.ops.aten.abs.default(convert_element_type)
        ne: "b8[32, 12, 512, 512]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        eq: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type, convert_element_type)
        mul_2: "b8[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(eq, ne);  eq = ne = None
        logical_not: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(mul_2);  mul_2 = None
        return (bmm, convert_element_type, mul_1, logical_not)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

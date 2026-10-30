"""Curated MM repro. Pattern hash: 4b8217e3930f."""
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
    def forward(self, arg0_1: "bf16[2, 4096]", arg1_1: "bf16[128, 8]", arg2_1: "f32[4096]", arg3_1: "f32[1, 128, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        constant_pad_nd: "bf16[8, 4096]" = torch.ops.aten.constant_pad_nd.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg1_1, constant_pad_nd);  arg1_1 = constant_pad_nd = None
        view: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, 4096)
        mul_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type, arg3_1);  arg3_1 = None
        sum_1: "f32[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 1], True);  mul_2 = None
        view_default: "f32[4096]" = torch.ops.aten.view.default(sum_1, _shape_param_2);  sum_1 = _shape_param_2 = None
        sum_2: "f32[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1], True);  convert_element_type = None
        view_default_1: "f32[4096]" = torch.ops.aten.view.default(sum_2, _shape_param_3);  sum_2 = _shape_param_3 = None
        return (mul, mul_1, view_default, view_default_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

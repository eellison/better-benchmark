"""Curated MM repro. Pattern hash: 62c6fa9b5402."""
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
    def forward(self, arg0_1: "bf16[32, 512, 50257]", arg1_1: "bf16[50257, 768]", arg2_1: "f32[32, 512, 768]", arg3_1: "f32[768]", arg4_1: "f32[32, 512, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[16384, 50257]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        constant_pad_nd: "bf16[16384, 50264]" = torch.ops.aten.constant_pad_nd.default(view, _shape_param_1);  view = _shape_param_1 = None
        constant_pad_nd_1: "bf16[50264, 768]" = torch.ops.aten.constant_pad_nd.default(arg1_1, _shape_param_2);  arg1_1 = _shape_param_2 = None
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add, arg4_1);  arg4_1 = None
        return (add, mul, mul_1, mul_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

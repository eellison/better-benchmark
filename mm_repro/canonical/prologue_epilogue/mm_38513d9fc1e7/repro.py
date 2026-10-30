"""Curated MM repro. Pattern hash: 38513d9fc1e7."""
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
    def forward(self, arg0_1: "bf16[32, 128, 50257]", arg1_1: "f32[4096, 1]", arg2_1: "f32[4096, 1]", arg3_1: "f32[4096, 1]", arg4_1: "f32[4096, 50257]", arg5_1: "bf16[32, 128, 50257]", arg6_1: "bf16[50257, 2048]", arg7_1: "f32[2048]", arg8_1: "f32[32, 128, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        convert_element_type: "f32[32, 128, 50257]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "f32[4096, 50257]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        sub: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(view, arg1_1);  view = arg1_1 = None
        sub_1: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(sub, arg2_1);  sub = arg2_1 = None
        exp: "f32[4096, 50257]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        mul: "f32[4096, 50257]" = torch.ops.aten.mul.Tensor(exp, arg3_1);  exp = arg3_1 = None
        sub_2: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(arg4_1, mul);  arg4_1 = mul = None
        view_1: "f32[32, 128, 50257]" = torch.ops.aten.view.default(sub_2, _shape_param_1);  sub_2 = _shape_param_1 = None
        convert_element_type_1: "bf16[32, 128, 50257]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        add: "bf16[32, 128, 50257]" = torch.ops.aten.add.Tensor(arg5_1, convert_element_type_1);  arg5_1 = convert_element_type_1 = None
        view_2: "bf16[4096, 50257]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        constant_pad_nd: "bf16[4096, 50264]" = torch.ops.aten.constant_pad_nd.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        constant_pad_nd_1: "bf16[50264, 2048]" = torch.ops.aten.constant_pad_nd.default(arg6_1, _shape_param_4);  arg6_1 = _shape_param_4 = None
        mm: "bf16[4096, 2048]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        view_3: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm, _shape_param_5);  mm = _shape_param_5 = None
        convert_element_type_2: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_2, arg7_1);  arg7_1 = None
        mul_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_1, 2048)
        mul_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_2, arg8_1);  arg8_1 = None
        return (convert_element_type_2, mul_1, mul_2, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

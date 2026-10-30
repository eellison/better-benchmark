"""Curated MM repro. Pattern hash: 32bccf04e73e."""
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
    def forward(self, arg0_1: "bf16[32768, 768]", arg1_1: "bf16[768, 768]", arg2_1: "f32[256, 128, 768]", arg3_1: "f32[256, 128, 768]", arg4_1: "b8[256, 128, 768]", arg5_1: "f32[768]", arg6_1: "f32[256, 128, 768]", arg7_1: "f32[1, 128, 768]", arg8_1: "f32[256, 128, 1]", arg9_1: "f32[256, 128, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[32768, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        add_1: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        convert_element_type_1: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        mul: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_1, mul);  add_1 = mul = None
        mul_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg5_1);  arg5_1 = None
        mul_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        add_2: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(arg6_1, arg7_1);  arg6_1 = arg7_1 = None
        sub: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(add_2, arg8_1);  add_2 = arg8_1 = None
        mul_4: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub, arg9_1);  sub = arg9_1 = None
        mul_5: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_1, mul_4);  mul_4 = None
        return (mul_1, mul_2, mul_3, mul_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

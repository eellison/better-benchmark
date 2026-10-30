"""Curated MM repro. Pattern hash: 2d5d794e7e16."""
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
    def forward(self, arg0_1: "bf16[4096, 384]", arg1_1: "bf16[384, 512]", arg2_1: "f32[32, 128, 512]", arg3_1: "f32[32, 128, 512]", arg4_1: "f32[32, 128, 512]", arg5_1: "f32[32, 128, 512]", arg6_1: "f32[32, 128, 512]", arg7_1: "f32[32, 128, 512]", arg8_1: "f32[32, 128, 512]", arg9_1: "b8[32, 128, 512]", arg10_1: "f32[512]", arg11_1: "f32[32, 128, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        add_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        add_2: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_1, arg4_1);  add_1 = arg4_1 = None
        add_3: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_2, arg5_1);  add_2 = arg5_1 = None
        add_4: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_3, arg6_1);  add_3 = arg6_1 = None
        add_5: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_4, arg7_1);  add_4 = arg7_1 = None
        add_6: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_5, arg8_1);  add_5 = arg8_1 = None
        convert_element_type_1: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_6, mul);  add_6 = mul = None
        mul_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, arg10_1);  arg10_1 = None
        mul_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, arg11_1);  mul_1 = arg11_1 = None
        return (mul_2, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

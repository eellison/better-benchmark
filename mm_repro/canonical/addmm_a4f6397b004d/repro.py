"""Curated MM repro. Pattern hash: a4f6397b004d."""
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
    def forward(self, arg0_1: "f32[128, 1]", arg1_1: "f32[128, 768]", arg2_1: "f32[128, 1]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[1000]", arg6_1: "f32[768, 1000]"):
        # No stacktrace found for following nodes
        add: "f32[128, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-06);  arg0_1 = None
        rsqrt: "f32[128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 768]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_1: "bf16[768, 1000]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        convert_element_type_2: "bf16[128, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        addmm: "bf16[128, 1000]" = torch.ops.aten.addmm.default(convert_element_type, convert_element_type_2, convert_element_type_1);  convert_element_type = convert_element_type_2 = convert_element_type_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

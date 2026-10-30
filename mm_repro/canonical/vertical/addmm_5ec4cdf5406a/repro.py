"""Curated MM repro. Pattern hash: 5ec4cdf5406a."""
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
    def forward(self, arg0_1: "bf16[128, 1]", arg1_1: "bf16[1, 16]", arg2_1: "bf16[16]", arg3_1: "bf16[16, 16]", arg4_1: "bf16[16]", arg5_1: "bf16[16, 16]", arg6_1: "bf16[16]", arg7_1: "bf16[16, 16]", arg8_1: "bf16[16]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[1, 16]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        mul_1: "f32[128, 16]" = torch.ops.aten.mul.Tensor(mul, 1);  mul = None
        convert_element_type_2: "f32[16]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        add: "f32[128, 16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        convert_element_type_3: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        tanh: "bf16[128, 16]" = torch.ops.aten.tanh.default(convert_element_type_3);  convert_element_type_3 = None
        addmm: "bf16[128, 16]" = torch.ops.aten.addmm.default(arg4_1, tanh, arg3_1);  arg4_1 = tanh = arg3_1 = None
        tanh_1: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm);  addmm = None
        addmm_1: "bf16[128, 16]" = torch.ops.aten.addmm.default(arg6_1, tanh_1, arg5_1);  arg6_1 = tanh_1 = arg5_1 = None
        tanh_2: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm_1);  addmm_1 = None
        addmm_2: "bf16[128, 16]" = torch.ops.aten.addmm.default(arg8_1, tanh_2, arg7_1);  arg8_1 = tanh_2 = arg7_1 = None
        tanh_3: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm_2);  addmm_2 = None
        return tanh_3

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

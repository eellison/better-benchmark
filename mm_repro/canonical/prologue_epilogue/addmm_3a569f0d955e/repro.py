"""Curated MM repro. Pattern hash: 3a569f0d955e."""
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
    def forward(self, arg0_1: "f32[16]", arg1_1: "f32[1, 16]", arg2_1: "f32[128, 1]", arg3_1: "f32[16]", arg4_1: "f32[16, 16]"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[16]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[1, 16]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "bf16[128, 1]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        convert_element_type_3: "f32[16]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_4: "f32[128, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        convert_element_type_5: "f32[1, 16]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        mul: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_4, convert_element_type_5);  convert_element_type_4 = convert_element_type_5 = None
        mul_1: "f32[128, 16]" = torch.ops.aten.mul.Tensor(mul, 1);  mul = None
        mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1);  convert_element_type_3 = None
        add: "f32[128, 16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        convert_element_type_6: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        tanh: "bf16[128, 16]" = torch.ops.aten.tanh.default(convert_element_type_6);  convert_element_type_6 = None
        convert_element_type_7: "bf16[16]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_8: "bf16[16, 16]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        addmm: "bf16[128, 16]" = torch.ops.aten.addmm.default(convert_element_type_7, tanh, convert_element_type_8);  convert_element_type_7 = tanh = convert_element_type_8 = None
        tanh_1: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm);  addmm = None
        return tanh_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

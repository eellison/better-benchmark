"""Curated MM repro. Pattern hash: e9d6848e08b0."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[16]", arg1_1: "f32[16, 1]", arg2_1: "f32[128, 1]", arg3_1: "bf16[16]", arg4_1: "bf16[16, 16]"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[16]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[16, 1]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "bf16[128, 1]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        permute: "bf16[1, 16]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        convert_element_type_3: "f32[16]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_4: "f32[128, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32)
        convert_element_type_5: "f32[1, 16]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        mul: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_4, convert_element_type_5);  convert_element_type_4 = convert_element_type_5 = None
        mul_1: "f32[128, 16]" = torch.ops.aten.mul.Tensor(mul, 1);  mul = None
        mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1);  convert_element_type_3 = None
        add: "f32[128, 16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        convert_element_type_6: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        tanh: "bf16[128, 16]" = torch.ops.aten.tanh.default(convert_element_type_6);  convert_element_type_6 = None
        addmm: "bf16[128, 16]" = torch.ops.aten.addmm.default(arg3_1, tanh, arg4_1);  arg3_1 = arg4_1 = None
        return (convert_element_type_2, tanh, addmm)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

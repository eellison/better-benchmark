"""Curated MM repro. Pattern hash: 8c79f4a2b9fa."""
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
    def forward(self, arg0_1: "f32[16]", arg1_1: "f32[16, 16]", arg2_1: "bf16[128, 16]", arg3_1: "f32[16]", arg4_1: "f32[16, 16]", arg5_1: "f32[1]", arg6_1: "f32[16, 1]"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[16]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[16, 16]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[128, 16]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        tanh: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm);  addmm = None
        convert_element_type_2: "bf16[16]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_3: "bf16[16, 16]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        addmm_1: "bf16[128, 16]" = torch.ops.aten.addmm.default(convert_element_type_2, tanh, convert_element_type_3);  convert_element_type_2 = convert_element_type_3 = None
        tanh_1: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm_1);  addmm_1 = None
        convert_element_type_4: "bf16[1]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_5: "bf16[16, 1]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        addmm_2: "bf16[128, 1]" = torch.ops.aten.addmm.default(convert_element_type_4, tanh_1, convert_element_type_5);  convert_element_type_4 = convert_element_type_5 = None
        return (tanh, tanh_1, addmm_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

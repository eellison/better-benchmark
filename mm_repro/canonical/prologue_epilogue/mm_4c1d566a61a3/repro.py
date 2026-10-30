"""Curated MM repro. Pattern hash: 4c1d566a61a3."""
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
    def forward(self, arg0_1: "bf16[128, 1]", arg1_1: "f32[1, 16]", arg2_1: "bf16[128, 16]", arg3_1: "bf16[16, 16]", arg4_1: "bf16[128, 16]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "bf16[1, 16]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "f32[1, 16]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        mul: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type_2);  convert_element_type = convert_element_type_2 = None
        convert_element_type_3: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_4: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_5: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul_1: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_5, convert_element_type_5);  convert_element_type_5 = None
        sub: "f32[128, 16]" = torch.ops.aten.sub.Tensor(1, mul_1);  mul_1 = None
        mul_2: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_4, sub);  convert_element_type_4 = sub = None
        convert_element_type_6: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        mm: "bf16[128, 16]" = torch.ops.aten.mm.default(convert_element_type_6, arg3_1);  convert_element_type_6 = arg3_1 = None
        convert_element_type_7: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_8: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        mul_3: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_8, convert_element_type_8);  convert_element_type_8 = None
        sub_1: "f32[128, 16]" = torch.ops.aten.sub.Tensor(1, mul_3);  mul_3 = None
        mul_4: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_7, sub_1);  convert_element_type_7 = sub_1 = None
        convert_element_type_9: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        permute_2: "bf16[16, 128]" = torch.ops.aten.permute.default(convert_element_type_9, [1, 0])
        return (convert_element_type_9, permute_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: 39e42e7c5f69."""
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
    def forward(self, arg0_1: "bf16[128, 1]", arg1_1: "f32[1, 16]", arg2_1: "bf16[128, 16]", arg3_1: "bf16[128, 16]"):
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
        permute_2: "bf16[16, 128]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        mm: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_2, arg3_1);  permute_2 = arg3_1 = None
        convert_element_type_7: "f32[16, 16]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type_7

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

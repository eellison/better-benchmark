"""Curated MM repro. Pattern hash: 9523804b6cd1."""
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
    def forward(self, arg0_1: "bf16[128, 16]", arg1_1: "bf16[16, 16]", arg2_1: "bf16[128, 16]"):
        # No stacktrace found for following nodes
        mm: "bf16[128, 16]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_1: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1);  convert_element_type_1 = None
        sub: "f32[128, 16]" = torch.ops.aten.sub.Tensor(1, mul);  mul = None
        mul_1: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type, sub);  convert_element_type = sub = None
        convert_element_type_2: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        permute: "bf16[16, 128]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0])
        return (convert_element_type_2, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

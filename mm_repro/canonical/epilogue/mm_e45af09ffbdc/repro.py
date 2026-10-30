"""Curated MM repro. Pattern hash: e45af09ffbdc."""
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
    def forward(self, arg0_1: "bf16[256, 512]", arg1_1: "bf16[512, 1024]", arg2_1: "b8[256, 1024]", arg3_1: "bf16[256, 1024]"):
        # No stacktrace found for following nodes
        mm: "bf16[256, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 5.000000000000001);  convert_element_type = None
        mul_1: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(mm, mul);  mm = mul = None
        convert_element_type_1: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        convert_element_type_2: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        le: "b8[256, 1024]" = torch.ops.aten.le.Scalar(convert_element_type_2, 0)
        mul_2: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1)
        mul_3: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1.7580993408473766);  mul_2 = None
        mul_4: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        exp: "f32[256, 1024]" = torch.ops.aten.exp.default(mul_4);  mul_4 = None
        mul_5: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(mul_3, exp);  mul_3 = exp = None
        mul_6: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.0507009873554805);  convert_element_type_1 = None
        where: "f32[256, 1024]" = torch.ops.aten.where.self(le, mul_5, mul_6);  le = mul_5 = mul_6 = None
        convert_element_type_3: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        permute: "bf16[1024, 256]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0])
        return (convert_element_type_3, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

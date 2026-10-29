"""Curated MM repro. Pattern hash: e0a3efd98d31."""
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
    def forward(self, arg0_1: "bf16[256, 197951]", arg1_1: "bf16[256, 197951]", arg2_1: "bf16[197951, 512]", arg3_1: "f32[256, 512]", arg4_1: "b8[256, 512]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[256, 197951]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[256, 197951]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        le: "b8[256, 197951]" = torch.ops.aten.le.Scalar(convert_element_type_1, 0)
        mul: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(convert_element_type, 1)
        mul_1: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(mul, 1.7580993408473766);  mul = None
        mul_2: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        exp: "f32[256, 197951]" = torch.ops.aten.exp.default(mul_2);  mul_2 = None
        mul_3: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(mul_1, exp);  mul_1 = exp = None
        mul_4: "f32[256, 197951]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.0507009873554805);  convert_element_type = None
        where: "f32[256, 197951]" = torch.ops.aten.where.self(le, mul_3, mul_4);  le = mul_3 = mul_4 = None
        convert_element_type_2: "bf16[256, 197951]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        constant_pad_nd: "bf16[256, 197952]" = torch.ops.aten.constant_pad_nd.default(convert_element_type_2, [0, 1, 0, 0]);  convert_element_type_2 = None
        constant_pad_nd_1: "bf16[197952, 512]" = torch.ops.aten.constant_pad_nd.default(arg2_1, [0, 0, 0, 1]);  arg2_1 = None
        mm: "bf16[256, 512]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        convert_element_type_3: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mul_5: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1)
        mul_6: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_5, 1.7580993408473766);  mul_5 = None
        mul_7: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_6, arg3_1);  mul_6 = arg3_1 = None
        mul_8: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0507009873554805);  convert_element_type_3 = None
        where_1: "f32[256, 512]" = torch.ops.aten.where.self(arg4_1, mul_7, mul_8);  arg4_1 = mul_7 = mul_8 = None
        convert_element_type_4: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        permute: "bf16[512, 256]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0])
        return (convert_element_type_4, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

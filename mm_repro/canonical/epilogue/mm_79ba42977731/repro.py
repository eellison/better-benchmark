"""Curated MM repro. Pattern hash: 79ba42977731."""
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
    def forward(self, arg0_1: "bf16[32, 1000]", arg1_1: "bf16[1000, 1280]", arg2_1: "b8[32, 1280]", arg3_1: "bf16[32, 1280]"):
        # No stacktrace found for following nodes
        mm: "bf16[32, 1280]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        div: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type, 0.8);  convert_element_type = None
        mul: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(mm, div);  mm = div = None
        convert_element_type_1: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_2: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        le: "b8[32, 1280]" = torch.ops.aten.le.Scalar(convert_element_type_2, -3)
        lt: "b8[32, 1280]" = torch.ops.aten.lt.Scalar(convert_element_type_2, 3)
        div_1: "f32[32, 1280]" = torch.ops.aten.div.Tensor(convert_element_type_2, 3);  convert_element_type_2 = None
        add: "f32[32, 1280]" = torch.ops.aten.add.Tensor(div_1, 0.5);  div_1 = None
        mul_1: "f32[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_1, add);  add = None
        where: "f32[32, 1280]" = torch.ops.aten.where.self(lt, mul_1, convert_element_type_1);  lt = mul_1 = convert_element_type_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 1280]" = torch.ops.aten.where.self(le, full, where);  le = full = where = None
        convert_element_type_3: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        permute: "bf16[1280, 32]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0])
        return (convert_element_type_3, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

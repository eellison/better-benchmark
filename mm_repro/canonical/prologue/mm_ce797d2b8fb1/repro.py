"""Curated MM repro. Pattern hash: ce797d2b8fb1."""
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
    def forward(self, arg0_1: "f32[8, 1024, 768]", arg1_1: "f32[8, 1024, 768]", arg2_1: "f32[8, 1024, 1]", arg3_1: "f32[8, 1024, 1]", arg4_1: "f32[8, 1024, 1]", arg5_1: "b8[8, 1024, 768]", arg6_1: "bf16[8192, 3072]", _shape_param_0):
        # No stacktrace found for following nodes
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg0_1, 768);  arg0_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul, arg3_1);  mul = arg3_1 = None
        sub_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub, mul_1);  sub = mul_1 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        convert_element_type: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        convert_element_type_1: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        mul_3: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_4: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_3);  convert_element_type = mul_3 = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_4, _shape_param_0);  mul_4 = _shape_param_0 = None
        permute: "bf16[768, 8192]" = torch.ops.aten.permute.default(view, [1, 0]);  view = None
        mm: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute, arg6_1);  permute = arg6_1 = None
        convert_element_type_2: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

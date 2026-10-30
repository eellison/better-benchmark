"""Curated MM repro. Pattern hash: cca6d5f064de."""
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
    def forward(self, arg0_1: "f32[512, 16, 240]", arg1_1: "f32[512, 16, 240]", arg2_1: "f32[512, 16, 1]", arg3_1: "f32[512, 16, 1]", arg4_1: "f32[512, 16, 1]", arg5_1: "bf16[8192, 480]", _shape_param_0):
        # No stacktrace found for following nodes
        mul: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(arg0_1, 240);  arg0_1 = None
        mul_1: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        sub: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(mul, arg3_1);  mul = arg3_1 = None
        sub_1: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(sub, mul_1);  sub = mul_1 = None
        div: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(arg4_1, 240);  arg4_1 = None
        mul_2: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(div, sub_1);  div = sub_1 = None
        convert_element_type: "bf16[512, 16, 240]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        view: "bf16[8192, 240]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        permute: "bf16[240, 8192]" = torch.ops.aten.permute.default(view, [1, 0]);  view = None
        mm: "bf16[240, 480]" = torch.ops.aten.mm.default(permute, arg5_1);  permute = arg5_1 = None
        convert_element_type_1: "f32[240, 480]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

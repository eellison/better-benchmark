"""Curated MM repro. Pattern hash: 11c55b7d2484."""
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096, 1024]", arg2_1: "bf16[4096, 4096]", arg3_1: "bf16[4096, 1024]", arg4_1: "f32[1024]", arg5_1: "bf16[4096, 1024]", arg6_1: "b8[32, 128, 1024]", arg7_1: "f32[32, 128, 1024]", arg8_1: "f32[32, 128, 1]", arg9_1: "f32[32, 128, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mm_1: "bf16[4096, 1024]" = torch.ops.aten.mm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        view: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[4096, 1024]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        mul: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, arg4_1);  arg4_1 = None
        view_1: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg5_1, _shape_param_1);  arg5_1 = _shape_param_1 = None
        mul_1: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(arg6_1, view_1);  arg6_1 = view_1 = None
        mul_2: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None
        add: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(arg7_1, mul_2);  arg7_1 = mul_2 = None
        sub: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add, arg8_1);  add = arg8_1 = None
        mul_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, arg9_1);  sub = arg9_1 = None
        mul_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_3);  mul_3 = None
        return (convert_element_type, convert_element_type_1, mul, mul_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: af9dde4d9375."""
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
    def forward(self, arg0_1: "bf16[32, 512, 12, 64]", arg1_1: "bf16[32, 512, 12, 64]", arg2_1: "bf16[32, 512, 12, 64]", arg3_1: "bf16[2304, 768]", arg4_1: "f32[768]", arg5_1: "f32[32, 512, 768]", arg6_1: "f32[1, 512, 768]", arg7_1: "b8[32, 512, 768]", arg8_1: "f32[32, 512, 1]", arg9_1: "f32[32, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        view_2: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        cat: "bf16[32, 512, 2304]" = torch.ops.aten.cat.default([view, view_2, view_1], 2);  view = view_2 = view_1 = None
        view_3: "bf16[16384, 2304]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_3, arg3_1);  view_3 = arg3_1 = None
        convert_element_type: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view_4: "f32[32, 512, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_4);  convert_element_type = _shape_param_4 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_4, arg4_1);  arg4_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg5_1, arg6_1);  arg5_1 = arg6_1 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg7_1, add);  arg7_1 = add = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_3, arg8_1);  mul_3 = arg8_1 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, arg9_1);  sub = arg9_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_4, mul_4);  mul_4 = None
        return (view_4, mul, mul_1, mul_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

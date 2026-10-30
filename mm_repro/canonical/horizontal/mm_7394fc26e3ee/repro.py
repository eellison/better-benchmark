"""Curated MM repro. Pattern hash: 7394fc26e3ee."""
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
    def forward(self, arg0_1: "f32[1, 128, 4096]", arg1_1: "f32[1, 128, 1]", arg2_1: "f32[1, 128, 1]", arg3_1: "bf16[4096]", arg4_1: "bf16[4096]", arg5_1: "bf16[4096, 4096]", arg6_1: "bf16[4096, 4096]", arg7_1: "bf16[4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(arg2_1, 1e-05);  arg2_1 = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  _shape_param_0 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(view, arg5_1);  view = arg5_1 = None
        view_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        slice_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_2, 3, 0, 64)
        slice_2: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_2, 3, 64, 9223372036854775807);  view_2 = None
        view_3: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  _shape_param_2 = None
        mm_1: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_3, arg6_1);  view_3 = arg6_1 = None
        view_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(mm_1, _shape_param_3);  mm_1 = _shape_param_3 = None
        view_6: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type, _shape_param_4);  convert_element_type = _shape_param_4 = None
        mm_2: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_6, arg7_1);  view_6 = arg7_1 = None
        view_8: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(mm_2, _shape_param_5);  mm_2 = _shape_param_5 = None
        return (slice_1, slice_2, view_5, view_8)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

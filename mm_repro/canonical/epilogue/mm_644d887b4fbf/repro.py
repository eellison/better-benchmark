"""Curated MM repro. Pattern hash: 644d887b4fbf."""
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
    def forward(self, arg0_1: "bf16[128, 4096]", arg1_1: "bf16[4096, 4096]", arg2_1: "f32[1, 128, 4096]", arg3_1: "f32[1, 128, 4096]", arg4_1: "f32[1, 128, 4096]", arg5_1: "f32[4096]", arg6_1: "bf16[1, 128, 4096]", arg7_1: "f32[1, 128, 4096]", arg8_1: "f32[1, 128, 1]", arg9_1: "f32[1, 128, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        add_2: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_1, arg4_1);  add_1 = arg4_1 = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_2, arg5_1);  arg5_1 = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, 4096)
        add_3: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(arg6_1, arg7_1);  arg6_1 = arg7_1 = None
        sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_3, arg8_1);  add_3 = arg8_1 = None
        mul_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, arg9_1);  sub = arg9_1 = None
        mul_3: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_2, mul_2);  mul_2 = None
        return (add_2, mul, mul_1, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

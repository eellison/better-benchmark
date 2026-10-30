"""Curated MM repro. Pattern hash: 498412873f0a."""
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
    def forward(self, arg0_1: "bf16[64, 128, 16, 64]", arg1_1: "bf16[1024, 1024]", arg2_1: "f32[64, 128, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[64, 128, 1024]", arg5_1: "f32[64, 128, 1]", arg6_1: "f32[64, 128, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_1, arg1_1);  view_1 = arg1_1 = None
        view_2: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        mul: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        mul_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sub: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        mul_2: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, arg6_1);  sub = arg6_1 = None
        mul_3: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add, mul_2);  mul_2 = None
        return (add, mul, mul_1, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

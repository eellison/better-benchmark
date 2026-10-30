"""Curated MM repro. Pattern hash: d67f542839b4."""
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
    def forward(self, arg0_1: "f32[32, 512, 32000]", arg1_1: "f32[16384, 1]", arg2_1: "f32[16384, 1]", arg3_1: "f32[16384, 1]", arg4_1: "f32[16384, 32000]", arg5_1: "f32[32, 512, 32000]", arg6_1: "f32[16384, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "f32[16384, 32000]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        sub: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(view, arg1_1);  view = arg1_1 = None
        sub_1: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(sub, arg2_1);  sub = arg2_1 = None
        exp: "f32[16384, 32000]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        mul: "f32[16384, 32000]" = torch.ops.aten.mul.Tensor(exp, arg3_1);  exp = arg3_1 = None
        sub_2: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(arg4_1, mul);  arg4_1 = mul = None
        view_1: "f32[32, 512, 32000]" = torch.ops.aten.view.default(sub_2, _shape_param_1);  sub_2 = _shape_param_1 = None
        add: "f32[32, 512, 32000]" = torch.ops.aten.add.Tensor(arg5_1, view_1);  arg5_1 = view_1 = None
        view_2: "f32[16384, 32000]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        permute: "f32[32000, 16384]" = torch.ops.aten.permute.default(view_2, [1, 0]);  view_2 = None
        mm: "f32[32000, 768]" = torch.ops.aten.mm.default(permute, arg6_1);  permute = arg6_1 = None
        return mm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

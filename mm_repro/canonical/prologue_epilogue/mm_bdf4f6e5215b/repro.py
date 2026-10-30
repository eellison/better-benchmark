"""Curated MM repro. Pattern hash: bdf4f6e5215b."""
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
    def forward(self, arg0_1: "f32[32, 512, 32000]", arg1_1: "f32[16384, 1]", arg2_1: "f32[16384, 1]", arg3_1: "f32[16384, 1]", arg4_1: "f32[16384, 32000]", arg5_1: "f32[32, 512, 32000]", arg6_1: "f32[32000, 768]", arg7_1: "f32[768]", arg8_1: "f32[16384, 768]", arg9_1: "f32[32, 512, 1]", arg10_1: "f32[32, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
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
        mm: "f32[16384, 768]" = torch.ops.aten.mm.default(view_2, arg6_1);  view_2 = arg6_1 = None
        view_3: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_3, arg7_1);  arg7_1 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, 768)
        view_4: "f32[32, 512, 768]" = torch.ops.aten.view.default(arg8_1, _shape_param_4);  arg8_1 = _shape_param_4 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_4, 0.5)
        pow_1: "f32[32, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_4, 3.0)
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_4, mul_4);  view_4 = mul_4 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_1, 0.7978845608028654);  add_1 = None
        tanh: "f32[32, 512, 768]" = torch.ops.aten.tanh.default(mul_5);  mul_5 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, add_2);  mul_3 = add_2 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_6, arg9_1);  mul_6 = arg9_1 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, arg10_1);  sub_3 = arg10_1 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_3, mul_7);  mul_7 = None
        return (view_3, mul_1, mul_2, mul_8)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

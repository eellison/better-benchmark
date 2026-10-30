"""Curated MM repro. Pattern hash: 04f249165ef9."""
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
    def forward(self, arg0_1: "f32[1, 1, 384]", arg1_1: "f32[1, 1, 1]", arg2_1: "f32[1, 1, 1]", arg3_1: "bf16[384]", arg4_1: "bf16[384]", arg5_1: "bf16[384, 384]", arg6_1: "bf16[384]", arg7_1: "bf16[384, 384]", arg8_1: "bf16[384, 384]", arg9_1: "bf16[384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        sub: "f32[1, 1, 384]" = torch.ops.aten.sub.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        add: "f32[1, 1, 1]" = torch.ops.aten.add.Tensor(arg2_1, 1e-05);  arg2_1 = None
        rsqrt: "f32[1, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 1, 384]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[1, 1, 384]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type: "bf16[1, 1, 384]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[1, 384]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  _shape_param_0 = None
        addmm: "bf16[1, 384]" = torch.ops.aten.addmm.default(arg6_1, view, arg5_1);  arg6_1 = view = arg5_1 = None
        view_1: "bf16[1, 1, 384]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        mul_2: "bf16[1, 1, 384]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
        view_2: "bf16[1, 1, 6, 64]" = torch.ops.aten.view.default(mul_2, _shape_param_2);  mul_2 = _shape_param_2 = None
        permute_1: "bf16[1, 6, 1, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3: "bf16[1, 384]" = torch.ops.aten.view.default(convert_element_type, _shape_param_3);  _shape_param_3 = None
        mm: "bf16[1, 384]" = torch.ops.aten.mm.default(view_3, arg7_1);  view_3 = arg7_1 = None
        view_5: "bf16[1, 1, 6, 64]" = torch.ops.aten.view.default(mm, _shape_param_4);  mm = _shape_param_4 = None
        permute_3: "bf16[1, 6, 1, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6: "bf16[1, 384]" = torch.ops.aten.view.default(convert_element_type, _shape_param_5);  convert_element_type = _shape_param_5 = None
        addmm_1: "bf16[1, 384]" = torch.ops.aten.addmm.default(arg9_1, view_6, arg8_1);  arg9_1 = view_6 = arg8_1 = None
        view_8: "bf16[1, 1, 6, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_6);  addmm_1 = _shape_param_6 = None
        permute_5: "bf16[1, 6, 1, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        return (permute_1, permute_3, permute_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: 2c23e27403ce."""
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
    def forward(self, arg0_1: "f32[8, 1024, 1024]", arg1_1: "f32[8, 1024, 1]", arg2_1: "f32[8, 1024, 1]", arg3_1: "bf16[1024]", arg4_1: "bf16[1024]", arg5_1: "bf16[1024, 1024]", arg6_1: "bf16[1024]", arg7_1: "bf16[1024, 1024]", arg8_1: "bf16[1024]", arg9_1: "bf16[1024, 1024]", arg10_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        sub: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        add: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(arg2_1, 1e-05);  arg2_1 = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type: "bf16[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  _shape_param_0 = None
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg6_1, view, arg5_1);  arg6_1 = view = arg5_1 = None
        view_2: "bf16[8, 1024, 16, 64]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        permute_1: "bf16[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  _shape_param_2 = None
        addmm_1: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg8_1, view_3, arg7_1);  arg8_1 = view_3 = arg7_1 = None
        view_5: "bf16[8, 1024, 16, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_3);  addmm_1 = _shape_param_3 = None
        permute_3: "bf16[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type, _shape_param_4);  convert_element_type = _shape_param_4 = None
        addmm_2: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg10_1, view_6, arg9_1);  arg10_1 = view_6 = arg9_1 = None
        view_8: "bf16[8, 1024, 16, 64]" = torch.ops.aten.view.default(addmm_2, _shape_param_5);  addmm_2 = _shape_param_5 = None
        permute_5: "bf16[8, 16, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        return (permute_1, permute_3, permute_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

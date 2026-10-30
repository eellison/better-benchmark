"""Curated MM repro. Pattern hash: 51f8a594529c."""
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
    def forward(self, arg0_1: "bf16[16, 512, 29056]", arg1_1: "f32[8192, 1]", arg2_1: "f32[8192, 1]", arg3_1: "f32[8192, 1]", arg4_1: "f32[8192, 29056]", arg5_1: "bf16[16, 512, 29056]", arg6_1: "bf16[29056, 1024]", arg7_1: "f32[1024]", arg8_1: "bf16[8192, 1024]", arg9_1: "f32[16, 512, 1]", arg10_1: "f32[16, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        convert_element_type: "f32[16, 512, 29056]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "f32[8192, 29056]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        sub: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(view, arg1_1);  view = arg1_1 = None
        sub_1: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(sub, arg2_1);  sub = arg2_1 = None
        exp: "f32[8192, 29056]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        mul: "f32[8192, 29056]" = torch.ops.aten.mul.Tensor(exp, arg3_1);  exp = arg3_1 = None
        sub_2: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(arg4_1, mul);  arg4_1 = mul = None
        view_1: "f32[16, 512, 29056]" = torch.ops.aten.view.default(sub_2, _shape_param_1);  sub_2 = _shape_param_1 = None
        convert_element_type_1: "bf16[16, 512, 29056]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        add: "bf16[16, 512, 29056]" = torch.ops.aten.add.Tensor(arg5_1, convert_element_type_1);  arg5_1 = convert_element_type_1 = None
        view_2: "bf16[8192, 29056]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_2, arg6_1);  view_2 = arg6_1 = None
        view_3: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        convert_element_type_2: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, arg7_1);  arg7_1 = None
        mul_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1, 1024)
        view_4: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(arg8_1, _shape_param_4);  arg8_1 = _shape_param_4 = None
        convert_element_type_3: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.5)
        mul_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.7071067811865476);  convert_element_type_3 = None
        erf: "f32[16, 512, 1024]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = add_1 = None
        convert_element_type_4: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        convert_element_type_5: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        sub_3: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_5, arg9_1);  convert_element_type_5 = arg9_1 = None
        mul_6: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_3, arg10_1);  sub_3 = arg10_1 = None
        mul_7: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_6);  mul_6 = None
        return (convert_element_type_2, mul_1, mul_2, mul_7)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

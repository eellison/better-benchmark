"""Curated MM repro. Pattern hash: 6a5ae42b5863."""
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
    def forward(self, arg0_1: "bf16[8, 1024, 32128]", arg1_1: "f32[8192, 1]", arg2_1: "f32[8192, 1]", arg3_1: "f32[8192, 1]", arg4_1: "f32[8192, 32128]", arg5_1: "bf16[8, 1024, 32128]", arg6_1: "bf16[32128, 512]", arg7_1: "b8[8, 1024, 512]", arg8_1: "f32[512]", arg9_1: "f32[8, 1024, 512]", arg10_1: "f32[8, 1024, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[8192, 32128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[8192, 32128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        sub: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(convert_element_type, arg1_1);  convert_element_type = arg1_1 = None
        sub_1: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(sub, arg2_1);  sub = arg2_1 = None
        convert_element_type_1: "bf16[8192, 32128]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_2: "f32[8192, 32128]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        exp: "f32[8192, 32128]" = torch.ops.aten.exp.default(convert_element_type_2);  convert_element_type_2 = None
        mul: "f32[8192, 32128]" = torch.ops.aten.mul.Tensor(exp, arg3_1);  exp = arg3_1 = None
        sub_2: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(arg4_1, mul);  arg4_1 = mul = None
        convert_element_type_3: "bf16[8192, 32128]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_1: "bf16[8, 1024, 32128]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_1);  convert_element_type_3 = _shape_param_1 = None
        add: "bf16[8, 1024, 32128]" = torch.ops.aten.add.Tensor(arg5_1, view_1);  arg5_1 = view_1 = None
        view_2: "bf16[8192, 32128]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        mm: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_2, arg6_1);  view_2 = arg6_1 = None
        view_3: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        convert_element_type_4: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.04419417382415922);  convert_element_type_4 = None
        convert_element_type_5: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        mul_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        mul_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_3, arg8_1);  arg8_1 = None
        mul_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg9_1, arg10_1)
        mul_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_3, mul_5);  mul_3 = mul_5 = None
        mul_7: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_4, arg9_1);  arg9_1 = None
        mul_8: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_4, arg10_1);  mul_4 = arg10_1 = None
        return (mul_6, mul_7, mul_8)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

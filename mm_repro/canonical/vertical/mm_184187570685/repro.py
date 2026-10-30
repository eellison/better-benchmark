"""Curated MM repro. Pattern hash: 184187570685."""
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
    def forward(self, arg0_1: "f32[128, 56, 56, 128]", arg1_1: "f32[128, 56, 56, 128]", arg2_1: "f32[128, 56, 56, 128]", arg3_1: "f32[128, 56, 56, 128]", arg4_1: "f32[128, 56, 56, 1]", arg5_1: "f32[128, 56, 56, 128]", arg6_1: "bf16[128, 512]", arg7_1: "bf16[401408, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        sum_1: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [3], True);  arg0_1 = None
        sum_2: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(arg1_1, [3], True);  arg1_1 = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  arg2_1 = sum_2 = None
        sub: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(arg3_1, sum_1);  arg3_1 = sum_1 = None
        sub_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub, mul);  sub = mul = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        add: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(arg5_1, mul_1);  arg5_1 = mul_1 = None
        view: "f32[128, 3136, 128]" = torch.ops.aten.view.default(add, _shape_param_0);  add = _shape_param_0 = None
        convert_element_type: "bf16[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16)
        view_1: "bf16[401408, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        mm: "bf16[401408, 512]" = torch.ops.aten.mm.default(view_1, arg6_1);  arg6_1 = None
        permute: "bf16[128, 401408]" = torch.ops.aten.permute.default(view_1, [1, 0])
        view_2: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type_1: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        view_3: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(arg7_1, _shape_param_3);  arg7_1 = _shape_param_3 = None
        convert_element_type_2: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_2: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476)
        erf: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_2);  mul_2 = None
        add_1: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_3: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_4: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type_2)
        mul_5: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_4, -0.5);  mul_4 = None
        exp: "f32[128, 3136, 512]" = torch.ops.aten.exp.default(mul_5);  mul_5 = None
        mul_6: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_7: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_6);  convert_element_type_2 = mul_6 = None
        add_2: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(mul_3, mul_7);  mul_3 = mul_7 = None
        mul_8: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, add_2);  convert_element_type_1 = add_2 = None
        convert_element_type_3: "bf16[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_4: "bf16[401408, 512]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_4);  convert_element_type_3 = _shape_param_4 = None
        return (view, view_1, permute, view_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: f2023d0f43b6."""
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
    def forward(self, arg0_1: "bf16[32768, 30522]", arg1_1: "f32[32768, 1]", arg2_1: "f32[32768, 30522]", arg3_1: "bf16[256, 128, 30522]", arg4_1: "bf16[30522, 512]", arg5_1: "f32[512]", arg6_1: "bf16[32768, 512]", arg7_1: "f32[256, 128, 1]", arg8_1: "f32[256, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        convert_element_type: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        exp: "f32[32768, 30522]" = torch.ops.aten.exp.default(convert_element_type);  convert_element_type = None
        mul: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(exp, arg1_1);  exp = arg1_1 = None
        sub: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(arg2_1, mul);  arg2_1 = mul = None
        convert_element_type_1: "bf16[32768, 30522]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        view: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        add: "bf16[256, 128, 30522]" = torch.ops.aten.add.Tensor(arg3_1, view);  arg3_1 = view = None
        view_1: "bf16[32768, 30522]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        constant_pad_nd: "bf16[32768, 30528]" = torch.ops.aten.constant_pad_nd.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        constant_pad_nd_1: "bf16[30528, 512]" = torch.ops.aten.constant_pad_nd.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        mm: "bf16[32768, 512]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        view_2: "bf16[256, 128, 512]" = torch.ops.aten.view.default(mm, _shape_param_4);  mm = _shape_param_4 = None
        convert_element_type_2: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, arg5_1);  arg5_1 = None
        mul_2: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, 512)
        view_3: "bf16[256, 128, 512]" = torch.ops.aten.view.default(arg6_1, _shape_param_5);  arg6_1 = _shape_param_5 = None
        relu: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_3);  view_3 = None
        convert_element_type_3: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(relu, torch.float32);  relu = None
        sub_1: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_3, arg7_1);  convert_element_type_3 = arg7_1 = None
        mul_3: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(sub_1, arg8_1);  sub_1 = arg8_1 = None
        mul_4: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_3);  mul_3 = None
        return (convert_element_type_2, mul_1, mul_2, mul_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

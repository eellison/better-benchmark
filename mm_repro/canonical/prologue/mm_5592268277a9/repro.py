"""Curated MM repro. Pattern hash: 5592268277a9."""
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
    def forward(self, arg0_1: "bf16[1, 128]", arg1_1: "f32[1, 1]", arg2_1: "f32[1, 1]", arg3_1: "f32[1, 1]", arg4_1: "f32[1, 128]", arg5_1: "bf16[1, 128]", arg6_1: "f32[1, 128]", arg7_1: "f32[1, 1]", arg8_1: "f32[1, 1]", arg9_1: "f32[1, 128]", arg10_1: "bf16[1, 128]", arg11_1: "bf16[2, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        convert_element_type: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        sub: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, arg1_1);  convert_element_type = arg1_1 = None
        sub_1: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub, arg2_1);  sub = arg2_1 = None
        convert_element_type_1: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_2: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        exp: "f32[1, 128]" = torch.ops.aten.exp.default(convert_element_type_2);  convert_element_type_2 = None
        mul: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp, arg3_1);  exp = arg3_1 = None
        sub_2: "f32[1, 128]" = torch.ops.aten.sub.Tensor(arg4_1, mul);  arg4_1 = mul = None
        convert_element_type_3: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        add: "bf16[1, 128]" = torch.ops.aten.add.Tensor(arg5_1, convert_element_type_3);  arg5_1 = convert_element_type_3 = None
        sub_3: "f32[1, 128]" = torch.ops.aten.sub.Tensor(arg6_1, arg7_1);  arg6_1 = arg7_1 = None
        convert_element_type_4: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_3, torch.bfloat16);  sub_3 = None
        convert_element_type_5: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        exp_1: "f32[1, 128]" = torch.ops.aten.exp.default(convert_element_type_5);  convert_element_type_5 = None
        mul_1: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_1, arg8_1);  exp_1 = arg8_1 = None
        sub_4: "f32[1, 128]" = torch.ops.aten.sub.Tensor(arg9_1, mul_1);  arg9_1 = mul_1 = None
        convert_element_type_6: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_4, torch.bfloat16);  sub_4 = None
        add_1: "bf16[1, 128]" = torch.ops.aten.add.Tensor(arg10_1, convert_element_type_6);  arg10_1 = convert_element_type_6 = None
        unsqueeze: "bf16[1, 128, 1]" = torch.ops.aten.view.default(add, _shape_param_0);  add = _shape_param_0 = None
        unsqueeze_1: "bf16[1, 128, 1]" = torch.ops.aten.view.default(add_1, _shape_param_1);  add_1 = _shape_param_1 = None
        cat: "bf16[1, 128, 2]" = torch.ops.aten.cat.default([unsqueeze_1, unsqueeze], 2);  unsqueeze_1 = unsqueeze = None
        view: "bf16[128, 2]" = torch.ops.aten.view.default(cat, _shape_param_2);  cat = _shape_param_2 = None
        constant_pad_nd: "bf16[128, 8]" = torch.ops.aten.constant_pad_nd.default(view, _shape_param_3);  view = _shape_param_3 = None
        constant_pad_nd_1: "bf16[8, 4096]" = torch.ops.aten.constant_pad_nd.default(arg11_1, _shape_param_4);  arg11_1 = _shape_param_4 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        view_1: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_5);  mm = _shape_param_5 = None
        convert_element_type_7: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        return convert_element_type_7

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

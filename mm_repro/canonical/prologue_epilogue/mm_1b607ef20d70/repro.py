"""Curated MM repro. Pattern hash: 1b607ef20d70."""
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
    def forward(self, arg0_1: "f32[512, 16, 240]", arg1_1: "f32[512, 16, 240]", arg2_1: "f32[512, 16, 1]", arg3_1: "f32[512, 16, 1]", arg4_1: "f32[512, 16, 1]", arg5_1: "bf16[240, 480]", arg6_1: "bf16[8192, 480]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mul: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(arg0_1, 240);  arg0_1 = None
        mul_1: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        sub: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(mul, arg3_1);  mul = arg3_1 = None
        sub_1: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(sub, mul_1);  sub = mul_1 = None
        div: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(arg4_1, 240);  arg4_1 = None
        mul_2: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(div, sub_1);  div = sub_1 = None
        convert_element_type: "bf16[512, 16, 240]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        view: "bf16[8192, 240]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        mm: "bf16[8192, 480]" = torch.ops.aten.mm.default(view, arg5_1);  view = arg5_1 = None
        view_1: "bf16[512, 16, 480]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type_1: "f32[512, 16, 480]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        view_2: "bf16[512, 16, 480]" = torch.ops.aten.view.default(arg6_1, _shape_param_2);  arg6_1 = _shape_param_2 = None
        convert_element_type_2: "f32[512, 16, 480]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sigmoid: "f32[512, 16, 480]" = torch.ops.aten.sigmoid.default(convert_element_type_2)
        mul_3: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sigmoid);  convert_element_type_1 = None
        sub_2: "f32[512, 16, 480]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_4: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_2);  convert_element_type_2 = sub_2 = None
        add: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(mul_4, 1);  mul_4 = None
        mul_5: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(mul_3, add);  mul_3 = add = None
        convert_element_type_3: "bf16[512, 16, 480]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_3: "bf16[8192, 480]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  convert_element_type_3 = _shape_param_3 = None
        permute: "bf16[480, 8192]" = torch.ops.aten.permute.default(view_3, [1, 0])
        return (view_3, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

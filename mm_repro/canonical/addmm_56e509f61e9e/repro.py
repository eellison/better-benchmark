"""Curated MM repro. Pattern hash: 56e509f61e9e."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[960]", arg1_1: "bf16[32, 960, 7, 7]", arg2_1: "bf16[960]", arg3_1: "bf16[960]", arg4_1: "bf16[960]", arg5_1: "bf16[960, 1280]", arg6_1: "bf16[1280]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        convert_element_type: "f32[960]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        unsqueeze_1: "f32[960, 1, 1]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        sub: "f32[32, 960, 7, 7]" = torch.ops.aten.sub.Tensor(arg1_1, unsqueeze_1);  arg1_1 = unsqueeze_1 = None
        convert_element_type_1: "f32[960]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0.001);  convert_element_type_1 = None
        sqrt: "f32[960]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_3: "f32[960, 1, 1]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        mul_1: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_5: "bf16[960, 1, 1]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        mul_2: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_7: "bf16[960, 1, 1]" = torch.ops.aten.view.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        add_1: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_3: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        add_2: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_3, 3)
        clamp_min: "f32[32, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_2, 0);  add_2 = None
        clamp_max: "f32[32, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul_3: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_3, clamp_max);  convert_element_type_3 = clamp_max = None
        div: "f32[32, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_3, 6);  mul_3 = None
        convert_element_type_4: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mean: "bf16[32, 960, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_4, [2, 3], True);  convert_element_type_4 = None
        view: "bf16[32, 960]" = torch.ops.aten.view.default(mean, _shape_param_4);  mean = _shape_param_4 = None
        addmm: "bf16[32, 1280]" = torch.ops.aten.addmm.default(arg6_1, view, arg5_1);  arg6_1 = view = arg5_1 = None
        convert_element_type_5: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        add_3: "f32[32, 1280]" = torch.ops.aten.add.Tensor(convert_element_type_5, 3)
        clamp_min_1: "f32[32, 1280]" = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
        clamp_max_1: "f32[32, 1280]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
        mul_4: "f32[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_5, clamp_max_1);  convert_element_type_5 = clamp_max_1 = None
        div_1: "f32[32, 1280]" = torch.ops.aten.div.Tensor(mul_4, 6);  mul_4 = None
        convert_element_type_6: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        return convert_element_type_6

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

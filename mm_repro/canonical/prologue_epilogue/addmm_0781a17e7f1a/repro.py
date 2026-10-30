"""Curated MM repro. Pattern hash: 0781a17e7f1a."""
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
    def forward(self, arg0_1: "bf16[512]", arg1_1: "bf16[96, 512, 4, 4]", arg2_1: "bf16[512]", arg3_1: "bf16[512]", arg4_1: "bf16[512]", arg5_1: "bf16[96, 512, 4, 4]", arg6_1: "bf16[512, 65]", arg7_1: "bf16[65]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        convert_element_type: "f32[512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        unsqueeze_1: "f32[512, 1, 1]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        sub: "f32[96, 512, 4, 4]" = torch.ops.aten.sub.Tensor(arg1_1, unsqueeze_1);  arg1_1 = unsqueeze_1 = None
        convert_element_type_1: "f32[512]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
        sqrt: "f32[512]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_3: "f32[512, 1, 1]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        mul_1: "f32[96, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_5: "bf16[512, 1, 1]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        mul_2: "f32[96, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_7: "bf16[512, 1, 1]" = torch.ops.aten.view.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        add_1: "f32[96, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "bf16[96, 512, 4, 4]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        add_2: "bf16[96, 512, 4, 4]" = torch.ops.aten.add.Tensor(convert_element_type_2, arg5_1);  convert_element_type_2 = arg5_1 = None
        relu: "bf16[96, 512, 4, 4]" = torch.ops.aten.relu.default(add_2);  add_2 = None
        avg_pool2d: "bf16[96, 512, 1, 1]" = torch.ops.aten.avg_pool2d.default(relu, [4, 4]);  relu = None
        view: "bf16[96, 512]" = torch.ops.aten.view.default(avg_pool2d, _shape_param_4);  avg_pool2d = _shape_param_4 = None
        addmm: "bf16[96, 65]" = torch.ops.aten.addmm.default(arg7_1, view, arg6_1);  arg7_1 = view = arg6_1 = None
        sigmoid: "bf16[96, 65]" = torch.ops.aten.sigmoid.default(addmm);  addmm = None
        return sigmoid

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

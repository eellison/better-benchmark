"""Curated MM repro. Pattern hash: 2085869a2702."""
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
    def forward(self, arg0_1: "f32[1, 1024, 1, 1]", arg1_1: "bf16[128, 1024, 7, 7]", arg2_1: "f32[1, 1024, 1, 1]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1000]", arg6_1: "f32[1024, 1000]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        add: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-05);  arg0_1 = None
        rsqrt: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[128, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        unsqueeze_1: "f32[1024, 1, 1]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        mul_1: "f32[128, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_3: "f32[1024, 1, 1]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        add_1: "f32[128, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[128, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        relu: "bf16[128, 1024, 7, 7]" = torch.ops.aten.relu.default(convert_element_type);  convert_element_type = None
        mean: "bf16[128, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu, [2, 3], True);  relu = None
        view_default: "bf16[128, 1024]" = torch.ops.aten.view.default(mean, _shape_param_2);  mean = _shape_param_2 = None
        convert_element_type_1: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_2: "bf16[1024, 1000]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        addmm: "bf16[128, 1000]" = torch.ops.aten.addmm.default(convert_element_type_1, view_default, convert_element_type_2);  convert_element_type_1 = view_default = convert_element_type_2 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

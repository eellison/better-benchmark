"""Curated MM repro. Pattern hash: 1db55f309835."""
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
    def forward(self, arg0_1: "f32[1, 512, 1, 1]", arg1_1: "bf16[96, 512, 4, 4]", arg2_1: "f32[1, 512, 1, 1]", arg3_1: "f32[512]", arg4_1: "f32[512]", arg5_1: "bf16[96, 512, 4, 4]", arg6_1: "f32[65]", arg7_1: "f32[512, 65]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        add: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-05);  arg0_1 = None
        rsqrt: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[96, 512, 4, 4]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[96, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        unsqueeze_1: "f32[512, 1, 1]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        mul_1: "f32[96, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_3: "f32[512, 1, 1]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        add_1: "f32[96, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[96, 512, 4, 4]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        add_2: "bf16[96, 512, 4, 4]" = torch.ops.aten.add.Tensor(convert_element_type, arg5_1);  convert_element_type = arg5_1 = None
        relu: "bf16[96, 512, 4, 4]" = torch.ops.aten.relu.default(add_2);  add_2 = None
        avg_pool2d: "bf16[96, 512, 1, 1]" = torch.ops.aten.avg_pool2d.default(relu, [4, 4]);  relu = None
        view: "bf16[96, 512]" = torch.ops.aten.view.default(avg_pool2d, _shape_param_2);  avg_pool2d = _shape_param_2 = None
        convert_element_type_1: "bf16[65]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        convert_element_type_2: "bf16[512, 65]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        addmm: "bf16[96, 65]" = torch.ops.aten.addmm.default(convert_element_type_1, view, convert_element_type_2);  convert_element_type_1 = view = convert_element_type_2 = None
        sigmoid: "bf16[96, 65]" = torch.ops.aten.sigmoid.default(addmm);  addmm = None
        return sigmoid

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: b13ea5ca49fd."""
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
    def forward(self, arg0_1: "bf16[5, 64, 1, 1]", arg1_1: "f32[1, 64, 1, 1]", arg2_1: "f32[1, 64, 1, 1]", arg3_1: "bf16[64]", arg4_1: "bf16[64]", arg5_1: "bf16[64, 5]", arg6_1: "bf16[5]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        sub: "f32[5, 64, 1, 1]" = torch.ops.aten.sub.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        add: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(arg2_1, 1e-05);  arg2_1 = None
        rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[5, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        unsqueeze_1: "bf16[64, 1, 1]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        mul_1: "f32[5, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_3: "bf16[64, 1, 1]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        add_1: "f32[5, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[5, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[5, 64]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        addmm: "bf16[5, 5]" = torch.ops.aten.addmm.default(arg6_1, view, arg5_1);  arg6_1 = view = arg5_1 = None
        convert_element_type_1: "f32[5, 5]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        return (addmm, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

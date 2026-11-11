"""Curated MM repro. Pattern hash: ed8fe2f19b8c."""
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
    def forward(self, arg0_1: "f32[32, 128, 1]", arg1_1: "f32[32, 128, 2560]", arg2_1: "f32[32, 128, 1]", arg3_1: "f32[2560]", arg4_1: "f32[2560]", arg5_1: "f32[2560, 2560]", arg6_1: "bf16[2560]", arg7_1: "f32[2560, 2560]", arg8_1: "bf16[2560]", arg9_1: "f32[2560, 2560]", arg10_1: "bf16[2560]", _shape_param_0):
        # No stacktrace found for following nodes
        add: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-05);  arg0_1 = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_1: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[4096, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        addmm: "bf16[4096, 2560]" = torch.ops.aten.addmm.default(arg6_1, view, convert_element_type);  arg6_1 = convert_element_type = None
        convert_element_type_2: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        addmm_1: "bf16[4096, 2560]" = torch.ops.aten.addmm.default(arg8_1, view, convert_element_type_2);  arg8_1 = convert_element_type_2 = None
        convert_element_type_3: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        addmm_2: "bf16[4096, 2560]" = torch.ops.aten.addmm.default(arg10_1, view, convert_element_type_3);  arg10_1 = view = convert_element_type_3 = None
        return (addmm, addmm_1, addmm_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""Curated MM repro. Pattern hash: d218a826fb0b."""
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
    def forward(self, arg0_1: "f32[1, 128, 1]", arg1_1: "f32[1, 128, 4096]", arg2_1: "f32[1, 128, 1]", arg3_1: "f32[4096]", arg4_1: "f32[4096]", arg5_1: "f32[16384]", arg6_1: "f32[4096, 16384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-05);  arg0_1 = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        convert_element_type_1: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_2: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        addmm: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_1, view, convert_element_type_2);  convert_element_type_1 = view = convert_element_type_2 = None
        view_1: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm, _shape_param_1);  _shape_param_1 = None
        mul_2: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        convert_element_type_3: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        pow_1: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 3.0);  convert_element_type_3 = None
        mul_3: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_2: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_1, mul_3);  view_1 = mul_3 = None
        mul_4: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_2, 0.7978845608028654);  add_2 = None
        tanh: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_4);  mul_4 = None
        add_3: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_5: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_2, add_3);  mul_2 = add_3 = None
        convert_element_type_4: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_2: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_2);  convert_element_type_4 = _shape_param_2 = None
        return (addmm, view_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

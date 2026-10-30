"""Curated MM repro. Pattern hash: 2d4a59c07371."""
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
    def forward(self, arg0_1: "f32[1, 1000, 1]", arg1_1: "f32[1, 1000, 1024]", arg2_1: "bf16[1024]", arg3_1: "bf16[1024, 2048]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-06);  arg0_1 = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1000, 1024]" = torch.ops.aten.mul.Tensor(arg1_1, rsqrt);  arg1_1 = rsqrt = None
        convert_element_type: "bf16[1, 1000, 1024]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[1, 1000, 1024]" = torch.ops.aten.mul.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        view: "bf16[1000, 1024]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        mm: "bf16[1000, 2048]" = torch.ops.aten.mm.default(view, arg3_1);  view = arg3_1 = None
        view_2: "bf16[1, 1000, 16, 128]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type_1: "f32[1, 1000, 16, 128]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        pow_1: "f32[1, 1000, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 2)
        return (convert_element_type_1, pow_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

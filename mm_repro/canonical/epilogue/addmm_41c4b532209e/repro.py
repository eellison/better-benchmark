"""Curated MM repro. Pattern hash: 41c4b532209e."""
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
    def forward(self, arg0_1: "f32[128]", arg1_1: "f32[4096, 128]", arg2_1: "bf16[4096, 4096]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[4096, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[4096, 128]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        view: "bf16[8, 512, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  _shape_param_0 = None
        mul: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(view, 0.5)
        convert_element_type_2: "f32[8, 512, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        pow_1: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 3.0);  convert_element_type_2 = None
        mul_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(view, mul_1);  view = mul_1 = None
        mul_2: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[8, 512, 128]" = torch.ops.aten.tanh.default(mul_2);  mul_2 = None
        add_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_3: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        return (addmm, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

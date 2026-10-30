"""Curated MM repro. Pattern hash: 9188867c2500."""
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
    def forward(self, arg0_1: "bf16[2304, 9216]", arg1_1: "bf16[1000, 2304]", arg2_1: "bf16[1, 1000, 9216]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[1000, 9216]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view: "bf16[1, 1000, 9216]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[1, 1000, 9216]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type)
        mul_2: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(mul_1, convert_element_type);  mul_1 = None
        mul_3: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(mul_2, 0.044715);  mul_2 = None
        add: "f32[1, 1000, 9216]" = torch.ops.aten.add.Tensor(convert_element_type, mul_3);  convert_element_type = mul_3 = None
        mul_4: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[1, 1000, 9216]" = torch.ops.aten.tanh.default(mul_4);  mul_4 = None
        add_1: "f32[1, 1000, 9216]" = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        mul_5: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        convert_element_type_1: "bf16[1, 1000, 9216]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        mul_6: "bf16[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type_1, arg2_1);  convert_element_type_1 = arg2_1 = None
        view_1: "bf16[1000, 9216]" = torch.ops.aten.view.default(mul_6, _shape_param_1);  mul_6 = _shape_param_1 = None
        return view_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

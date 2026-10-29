"""Curated MM repro. Pattern hash: bfda18a45b33."""
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
    def forward(self, arg0_1: "f32[512, 1024]", arg1_1: "bf16[4096, 512]", arg2_1: "f32[512, 1024]", arg3_1: "b8[32, 128, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mm: "bf16[4096, 1024]" = torch.ops.aten.mm.default(arg1_1, convert_element_type);  convert_element_type = None
        view: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        mul: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view, 0.5)
        convert_element_type_1: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        pow_1: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 3.0);  convert_element_type_1 = None
        mul_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view, mul_1);  view = mul_1 = None
        mul_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_2);  mul_2 = None
        add_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        convert_element_type_2: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mm_1: "bf16[4096, 1024]" = torch.ops.aten.mm.default(arg1_1, convert_element_type_2);  arg1_1 = convert_element_type_2 = None
        view_1: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_1, _shape_param_1);  mm_1 = _shape_param_1 = None
        mul_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_3, view_1);  mul_3 = view_1 = None
        mul_5: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(arg3_1, mul_4);  arg3_1 = mul_4 = None
        mul_6: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        convert_element_type_3: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        view_2: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_2);  convert_element_type_3 = _shape_param_2 = None
        return view_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: 4a9e1a392d86."""
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
    def forward(self, arg0_1: "bf16[16384, 768]", arg1_1: "bf16[768, 3072]", arg2_1: "bf16[32, 512, 3072]", arg3_1: "f32[32, 512, 3072]", arg4_1: "f32[32, 512, 3072]", arg5_1: "f32[32, 512, 3072]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[16384, 3072]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[16384, 3072]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view: "f32[32, 512, 3072]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        mul: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view, arg2_1);  arg2_1 = None
        mul_1: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view, arg3_1);  view = arg3_1 = None
        convert_element_type_1: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        mul_2: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        mul_3: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_2, 0.7978845608028654);  mul_2 = None
        convert_element_type_2: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16)
        mul_4: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_3, 0.044715);  mul_3 = None
        mul_5: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, arg5_1);  mul_4 = arg5_1 = None
        convert_element_type_3: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        add: "bf16[32, 512, 3072]" = torch.ops.aten.add.Tensor(convert_element_type_2, convert_element_type_3);  convert_element_type_2 = convert_element_type_3 = None
        mul_6: "bf16[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.5);  convert_element_type_1 = None
        add_1: "bf16[32, 512, 3072]" = torch.ops.aten.add.Tensor(add, mul_6);  add = mul_6 = None
        view_1: "bf16[16384, 3072]" = torch.ops.aten.view.default(add_1, _shape_param_1);  add_1 = _shape_param_1 = None
        return view_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: 7aece3a27dd2."""
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096, 16384]", arg2_1: "bf16[4096, 4096]", arg3_1: "bf16[4096, 16384]", arg4_1: "bf16[4096, 16384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 16384]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mm_1: "bf16[4096, 16384]" = torch.ops.aten.mm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        view: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        view_1: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        mul: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        mul_1: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  mul = None
        convert_element_type_2: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        pow_1: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 3.0)
        mul_2: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_1, mul_2);  view_1 = mul_2 = None
        mul_3: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_3);  mul_3 = None
        add_1: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_4: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type, add_1);  convert_element_type = add_1 = None
        convert_element_type_3: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_5: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_5);  mul_5 = None
        mul_6: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_1, sub);  mul_1 = sub = None
        mul_7: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_6, 0.7978845608028654);  mul_6 = None
        convert_element_type_4: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16)
        mul_8: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_7, 0.044715);  mul_7 = None
        pow_2: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2.0);  convert_element_type_2 = None
        mul_9: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_10: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        convert_element_type_5: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        add_2: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_4, convert_element_type_5);  convert_element_type_4 = convert_element_type_5 = None
        mul_11: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.5);  convert_element_type_3 = None
        add_3: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_2, mul_11);  add_2 = mul_11 = None
        view_2: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_3, _shape_param_2);  add_3 = _shape_param_2 = None
        permute: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (convert_element_type_1, view_2, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

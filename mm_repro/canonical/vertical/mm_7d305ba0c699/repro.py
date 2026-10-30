"""Curated MM repro. Pattern hash: 7d305ba0c699."""
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
    def forward(self, arg0_1: "f32[8, 512, 128]", arg1_1: "f32[8, 512, 128]", arg2_1: "f32[8, 512, 128]", arg3_1: "f32[8, 512, 128]", arg4_1: "f32[8, 512, 1]", arg5_1: "bf16[8, 512, 128]", arg6_1: "f32[8, 512, 128]", arg7_1: "f32[8, 512, 128]", arg8_1: "f32[8, 512, 128]", arg9_1: "bf16[128, 4096]", arg10_1: "f32[4096]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        sum_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [2], True);  arg0_1 = None
        sum_2: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(arg1_1, [2], True);  arg1_1 = None
        mul: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  arg2_1 = sum_2 = None
        sub: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(arg3_1, sum_1);  arg3_1 = sum_1 = None
        sub_1: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub, mul);  sub = mul = None
        div: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(arg4_1, 128);  arg4_1 = None
        mul_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(div, sub_1);  div = sub_1 = None
        mul_2: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, arg5_1);  arg5_1 = None
        mul_3: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        convert_element_type: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_4: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(arg7_1, arg7_1);  arg7_1 = None
        sub_2: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(1, mul_4);  mul_4 = None
        mul_5: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_2, sub_2);  mul_2 = sub_2 = None
        mul_6: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_5, 0.7978845608028654);  mul_5 = None
        convert_element_type_1: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16)
        mul_7: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_6, 0.044715);  mul_6 = None
        pow_1: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(arg8_1, 2.0);  arg8_1 = None
        mul_8: "f32[8, 512, 128]" = torch.ops.aten.mul.Scalar(pow_1, 3.0);  pow_1 = None
        mul_9: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None
        convert_element_type_2: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        add: "bf16[8, 512, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_2);  convert_element_type_1 = convert_element_type_2 = None
        mul_10: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5);  convert_element_type = None
        add_1: "bf16[8, 512, 128]" = torch.ops.aten.add.Tensor(add, mul_10);  add = mul_10 = None
        view: "bf16[4096, 128]" = torch.ops.aten.view.default(add_1, _shape_param_0);  add_1 = _shape_param_0 = None
        mm: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view, arg9_1);  arg9_1 = None
        permute: "bf16[128, 4096]" = torch.ops.aten.permute.default(view, [1, 0])
        view_1: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type_3: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_11: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg10_1);  arg10_1 = None
        return (view, permute, convert_element_type_3, mul_11)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

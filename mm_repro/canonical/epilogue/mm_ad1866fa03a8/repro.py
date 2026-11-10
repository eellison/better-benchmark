"""Curated MM repro. Pattern hash: ad1866fa03a8."""
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
    def forward(self, arg0_1: "bf16[8192, 32128]", arg1_1: "bf16[32128, 512]", arg2_1: "b8[8, 1024, 512]", arg3_1: "f32[512]", arg4_1: "f32[8, 1024, 512]", arg5_1: "f32[8, 1024, 1]", arg6_1: "b8[8, 1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.04419417382415922);  convert_element_type = None
        convert_element_type_1: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul, mul_1);  mul = mul_1 = None
        mul_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_2, arg3_1);  arg3_1 = None
        mul_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg4_1, arg5_1)
        mul_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_2, mul_4);  mul_2 = mul_4 = None
        sum_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1], True, dtype = torch.float32);  mul_5 = None
        view_1: "f32[512]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        mul_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_3, arg4_1)
        mul_7: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_3, arg5_1);  mul_3 = None
        sum_2: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_6, [2], True, dtype = torch.float32);  mul_6 = None
        pow_1: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg5_1, 3);  arg5_1 = None
        mul_8: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_9: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_8, pow_1);  mul_8 = pow_1 = None
        expand: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_9, _shape_param_2);  mul_9 = _shape_param_2 = None
        div: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand, 512);  expand = None
        pow_2: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg4_1, 1.0);  arg4_1 = None
        mul_10: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        mul_11: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div, mul_10);  div = mul_10 = None
        add: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_7, mul_11);  mul_7 = mul_11 = None
        convert_element_type_2: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16)
        convert_element_type_3: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        mul_12: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_13: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_12);  convert_element_type_2 = mul_12 = None
        view_2: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_13, _shape_param_3);  mul_13 = _shape_param_3 = None
        permute: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_1, add, view_2, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

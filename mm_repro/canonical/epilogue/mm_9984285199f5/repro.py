"""Curated MM repro. Pattern hash: 9984285199f5."""
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
    def forward(self, arg0_1: "bf16[4096, 30000]", arg1_1: "bf16[30000, 128]", arg2_1: "f32[128]", arg3_1: "bf16[4096, 128]", arg4_1: "f32[8, 512, 1]", arg5_1: "f32[8, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 512, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8, 512, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, 128)
        sum_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        view_1: "bf16[8, 512, 128]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        mul_2: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        convert_element_type_1: "f32[8, 512, 128]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        pow_1: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 3.0)
        mul_3: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(view_1, mul_3);  view_1 = mul_3 = None
        mul_4: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[8, 512, 128]" = torch.ops.aten.tanh.default(mul_4);  mul_4 = None
        add_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_5: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_2, add_1)
        sub: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_5, arg4_1);  mul_5 = arg4_1 = None
        mul_6: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = None
        mul_7: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, mul_6);  mul = None
        sum_2: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [2], True);  mul_7 = None
        mul_8: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_6, sum_2);  sum_2 = None
        sub_1: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub_1, mul_8);  sub_1 = mul_8 = None
        div: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(arg5_1, 128);  arg5_1 = None
        mul_9: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_10: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_6);  mul_6 = None
        sum_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_10, [0, 1]);  mul_10 = None
        sum_4: "f32[128]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1]);  convert_element_type = None
        mul_11: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_9, mul_2);  mul_2 = None
        mul_12: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_9, add_1);  mul_9 = add_1 = None
        convert_element_type_2: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_13: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_3: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(1, mul_13);  mul_13 = None
        mul_14: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_11, sub_3);  mul_11 = sub_3 = None
        mul_15: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_14, 0.7978845608028654);  mul_14 = None
        convert_element_type_3: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16)
        mul_16: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_15, 0.044715);  mul_15 = None
        pow_2: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 2.0);  convert_element_type_1 = None
        mul_17: "f32[8, 512, 128]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_18: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        convert_element_type_4: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        add_2: "bf16[8, 512, 128]" = torch.ops.aten.add.Tensor(convert_element_type_3, convert_element_type_4);  convert_element_type_3 = convert_element_type_4 = None
        mul_19: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5);  convert_element_type_2 = None
        add_3: "bf16[8, 512, 128]" = torch.ops.aten.add.Tensor(add_2, mul_19);  add_2 = mul_19 = None
        view_2: "bf16[4096, 128]" = torch.ops.aten.view.default(add_3, _shape_param_2);  add_3 = _shape_param_2 = None
        permute: "bf16[128, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_5: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[128]" = torch.ops.aten.view.default(sum_5, _shape_param_3);  sum_5 = _shape_param_3 = None
        convert_element_type_5: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_6: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        return (sum_3, sum_4, view_2, permute, convert_element_type_6)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

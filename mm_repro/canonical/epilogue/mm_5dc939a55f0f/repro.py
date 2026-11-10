"""Curated MM repro. Pattern hash: 5dc939a55f0f."""
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "bf16[1024, 1024]", arg2_1: "f32[512, 16, 1024]", arg3_1: "bf16[8192, 1024]", arg4_1: "bf16[8192, 1024]", arg5_1: "f32[1024]", arg6_1: "f32[512, 16, 1024]", arg7_1: "f32[512, 16, 1]", arg8_1: "b8[512, 16, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        unsqueeze: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        squeeze: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view, 4);  view = None
        squeeze_1: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze, 3);  squeeze = None
        convert_element_type: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        add: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        unsqueeze_1: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(arg3_1, 0);  arg3_1 = None
        view_1: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_1);  unsqueeze_1 = _shape_param_1 = None
        squeeze_2: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view_1, 4);  view_1 = None
        squeeze_3: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_2, 3);  squeeze_2 = None
        convert_element_type_1: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_3, torch.float32);  squeeze_3 = None
        add_1: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add, convert_element_type_1);  add = convert_element_type_1 = None
        unsqueeze_2: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(arg4_1, 0);  arg4_1 = None
        view_2: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_2, _shape_param_2);  unsqueeze_2 = _shape_param_2 = None
        squeeze_4: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view_2, 4);  view_2 = None
        squeeze_5: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_4, 3);  squeeze_4 = None
        convert_element_type_2: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_5, torch.float32);  squeeze_5 = None
        add_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_2);  add_1 = convert_element_type_2 = None
        mul: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_2, arg5_1);  arg5_1 = None
        mul_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sum_1: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = None
        sum_2: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg6_1, sum_2);  sum_2 = None
        sub: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg7_1, sub_1);  arg7_1 = sub_1 = None
        mul_5: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_2, arg6_1);  arg6_1 = None
        sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_2, [0, 1]);  add_2 = None
        convert_element_type_3: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        convert_element_type_4: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        mul_6: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_7: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_3, mul_6);  convert_element_type_3 = mul_6 = None
        view_3: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_7, _shape_param_3);  mul_7 = _shape_param_3 = None
        permute: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_3, [1, 0])
        sum_5: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32)
        view_4: "f32[1024]" = torch.ops.aten.view.default(sum_5, _shape_param_4);  sum_5 = _shape_param_4 = None
        convert_element_type_5: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_6: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        return (mul_4, sum_3, sum_4, view_3, permute, convert_element_type_6)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

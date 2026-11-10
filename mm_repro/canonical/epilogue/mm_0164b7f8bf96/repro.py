"""Curated MM repro. Pattern hash: 0164b7f8bf96."""
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
    def forward(self, arg0_1: "bf16[25088, 1536]", arg1_1: "bf16[1536, 512]", arg2_1: "f32[512]", arg3_1: "bf16[128, 14, 14, 512]", arg4_1: "f32[128, 14, 14, 1]", arg5_1: "f32[128, 14, 14, 1]", arg6_1: "bf16[128, 14, 14, 512]", arg7_1: "b8[128, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        mm: "bf16[25088, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[512, 49, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        view_2: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        mul: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_3, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul, 512)
        sum_1: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul, [3], True)
        convert_element_type_1: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul_2: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = None
        mul_3: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [3], True);  mul_3 = None
        mul_4: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(arg5_1, 512);  arg5_1 = None
        mul_5: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_3, mul_2);  mul_2 = None
        sum_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1, 2]);  mul_6 = None
        sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_3, [0, 1, 2]);  view_3 = None
        convert_element_type_2: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        add: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(arg6_1, convert_element_type_2);  arg6_1 = convert_element_type_2 = None
        view_4: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        convert_element_type_3: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        div_1: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_3, 0.9521739110350609);  convert_element_type_3 = None
        mul_7: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_4, div_1);  div_1 = None
        view_5: "bf16[25088, 512]" = torch.ops.aten.view.default(mul_7, _shape_param_5);  mul_7 = _shape_param_5 = None
        permute_1: "bf16[512, 25088]" = torch.ops.aten.permute.default(view_5, [1, 0])
        sum_5: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32)
        view_6: "f32[512]" = torch.ops.aten.view.default(sum_5, _shape_param_6);  sum_5 = _shape_param_6 = None
        convert_element_type_4: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        convert_element_type_5: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        return (sum_3, sum_4, view_4, view_5, permute_1, convert_element_type_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

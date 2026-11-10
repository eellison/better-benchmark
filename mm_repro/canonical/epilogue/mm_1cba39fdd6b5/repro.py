"""Curated MM repro. Pattern hash: 1cba39fdd6b5."""
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
    def forward(self, arg0_1: "bf16[100352, 768]", arg1_1: "bf16[768, 256]", arg2_1: "i64[28]", arg3_1: "f32[256]", arg4_1: "bf16[128, 28, 28, 256]", arg5_1: "f32[128, 28, 28, 1]", arg6_1: "f32[128, 28, 28, 1]", arg7_1: "bf16[128, 28, 28, 256]", arg8_1: "b8[128, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        mm: "bf16[100352, 256]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[2048, 49, 256]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "f32[2048, 7, 7, 256]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        view_2: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "f32[128, 28, 28, 256]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        index: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(view_3, [None, None, arg2_1]);  view_3 = None
        index_1: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index, [None, arg2_1]);  index = arg2_1 = None
        mul: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(index_1, arg3_1);  arg3_1 = None
        mul_1: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul, 256)
        sum_1: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul, [3], True)
        convert_element_type_1: "f32[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        sub: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg5_1);  convert_element_type_1 = arg5_1 = None
        mul_2: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub, arg6_1);  sub = None
        mul_3: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [3], True);  mul_3 = None
        mul_4: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[128, 28, 28, 1]" = torch.ops.aten.div.Tensor(arg6_1, 256);  arg6_1 = None
        mul_5: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(index_1, mul_2);  mul_2 = None
        sum_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1, 2]);  mul_6 = None
        sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(index_1, [0, 1, 2]);  index_1 = None
        convert_element_type_2: "bf16[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        add: "bf16[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(arg7_1, convert_element_type_2);  arg7_1 = convert_element_type_2 = None
        view_4: "bf16[128, 784, 256]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        convert_element_type_3: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        div_1: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_3, 0.9913043472915888);  convert_element_type_3 = None
        mul_7: "bf16[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_4, div_1);  div_1 = None
        view_5: "bf16[100352, 256]" = torch.ops.aten.view.default(mul_7, _shape_param_5);  mul_7 = _shape_param_5 = None
        permute_1: "bf16[256, 100352]" = torch.ops.aten.permute.default(view_5, [1, 0])
        sum_5: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32)
        view_6: "f32[256]" = torch.ops.aten.view.default(sum_5, _shape_param_6);  sum_5 = _shape_param_6 = None
        convert_element_type_4: "bf16[256]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        convert_element_type_5: "f32[256]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        return (sum_3, sum_4, view_4, view_5, permute_1, convert_element_type_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""Curated MM repro. Pattern hash: e14e9103ee60."""
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
    def forward(self, arg0_1: "bf16[100352, 256]", arg1_1: "bf16[256, 512]", arg2_1: "f32[512]", arg3_1: "f32[128, 28, 28, 512]", arg4_1: "f32[128, 28, 28, 1]", arg5_1: "b8[128, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mm: "bf16[100352, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 28, 28, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[128, 28, 28, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul, 512)
        sum_1: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul, [3], True)
        mul_2: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = None
        sum_2: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [3], True);  mul_2 = None
        mul_3: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(arg3_1, sum_2);  sum_2 = None
        sub: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        mul_5: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, arg3_1);  arg3_1 = None
        sum_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1, 2]);  mul_5 = None
        sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1, 2]);  convert_element_type = None
        view_1: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.view.default(mul_4, _shape_param_1);  mul_4 = _shape_param_1 = None
        permute: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.permute.default(view_1, [0, 1, 4, 2, 3, 5]);  view_1 = None
        clone: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        view_3: "f32[128, 3136, 128]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        convert_element_type_1: "bf16[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16)
        convert_element_type_2: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        div: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_2, 0.9956521736457944);  convert_element_type_2 = None
        mul_6: "bf16[128, 3136, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, div);  convert_element_type_1 = div = None
        view_4: "bf16[401408, 128]" = torch.ops.aten.view.default(mul_6, _shape_param_4);  mul_6 = _shape_param_4 = None
        permute_1: "bf16[128, 401408]" = torch.ops.aten.permute.default(view_4, [1, 0])
        sum_5: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32)
        view_5: "f32[128]" = torch.ops.aten.view.default(sum_5, _shape_param_5);  sum_5 = _shape_param_5 = None
        convert_element_type_3: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        convert_element_type_4: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (sum_3, sum_4, view_3, view_4, permute_1, convert_element_type_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

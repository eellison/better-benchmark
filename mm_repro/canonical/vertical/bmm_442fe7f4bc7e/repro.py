"""Curated MM repro. Pattern hash: 442fe7f4bc7e."""
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
    def forward(self, arg0_1: "bf16[128, 768, 7, 7]", arg1_1: "bf16[768, 128, 49]", arg2_1: "bf16[768, 56, 56]", arg3_1: "f32[128, 6, 49, 1]", arg4_1: "f32[128, 6, 49, 1]", arg5_1: "b8[128, 6, 49, 1]", arg6_1: "f32[128, 6, 49, 1]", arg7_1: "bf16[768, 49, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[128, 6, 128, 49]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[128, 6, 49, 128]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2]);  view = None
        clone: "bf16[128, 6, 49, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[768, 49, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        bmm: "bf16[768, 49, 49]" = torch.ops.aten.bmm.default(view_1, arg1_1);  view_1 = arg1_1 = None
        view_2: "bf16[128, 6, 49, 49]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        convert_element_type: "f32[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        slice_1: "bf16[768, 49, 56]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, -7);  arg2_1 = None
        slice_2: "bf16[768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view_3: "bf16[128, 6, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_3);  slice_2 = _shape_param_3 = None
        mul: "bf16[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_3, 0.08838834764831845)
        convert_element_type_1: "f32[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_2: "f32[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_1: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        sub: "f32[128, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        mul_2: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub, 0.08838834764831845);  sub = None
        sub_1: "f32[128, 6, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        where: "f32[128, 6, 49, 49]" = torch.ops.aten.where.self(arg5_1, mul_2, sub_1);  arg5_1 = mul_2 = sub_1 = None
        exp: "f32[128, 6, 49, 49]" = torch.ops.aten.exp.default(where);  where = None
        div: "f32[128, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
        mul_3: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = None
        sum_1: "f32[128, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [3], True)
        neg: "f32[128, 6, 49, 49]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[128, 6, 49, 49]" = torch.ops.prims.fma.default(neg, sum_1, mul_3);  neg = sum_1 = mul_3 = None
        convert_element_type_3: "bf16[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        mul_4: "bf16[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.08838834764831845);  convert_element_type_3 = None
        view_4: "bf16[768, 49, 49]" = torch.ops.aten.view.default(mul_4, _shape_param_4);  mul_4 = _shape_param_4 = None
        bmm_1: "bf16[768, 49, 128]" = torch.ops.aten.bmm.default(view_4, arg7_1);  arg7_1 = None
        view_5: "bf16[128, 6, 49, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_5);  bmm_1 = _shape_param_5 = None
        return (view_4, view_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)

"""Curated MM repro. Pattern hash: 3676efcf8789."""
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
    def forward(self, arg0_1: "bf16[768, 56, 128]", arg1_1: "bf16[768, 128, 56]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[768, 56, 56]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "bf16[768, 49, 56]" = torch.ops.aten.slice.Tensor(bmm, 1, 0, -7)
        slice_2: "bf16[768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view: "bf16[128, 6, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_0);  slice_2 = _shape_param_0 = None
        mul: "bf16[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view, 0.08838834764831845)
        convert_element_type: "f32[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_1: "f32[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul_1: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        amax: "f32[128, 6, 49, 1]" = torch.ops.aten.amax.default(mul_1, [-1], True)
        sub: "f32[128, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_1, amax);  mul_1 = None
        mul_2: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub, 0.08838834764831845);  sub = None
        amax_1: "f32[128, 6, 49, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub_1: "f32[128, 6, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type, amax_1)
        abs_1: "f32[128, 6, 49, 49]" = torch.ops.aten.abs.default(convert_element_type)
        ne: "b8[128, 6, 49, 49]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        eq: "b8[128, 6, 49, 49]" = torch.ops.aten.eq.Tensor(convert_element_type, convert_element_type);  convert_element_type = None
        mul_3: "b8[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(eq, ne);  eq = ne = None
        logical_not: "b8[128, 6, 49, 49]" = torch.ops.aten.logical_not.default(mul_3);  mul_3 = None
        any_1: "b8[128, 6, 49, 1]" = torch.ops.aten.any.dims(logical_not, [-1], True);  logical_not = None
        logical_not_1: "b8[128, 6, 49, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        where: "f32[128, 6, 49, 49]" = torch.ops.aten.where.self(logical_not_1, mul_2, sub_1);  mul_2 = sub_1 = None
        exp: "f32[128, 6, 49, 49]" = torch.ops.aten.exp.default(where);  where = None
        sum_1: "f32[128, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[128, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_2: "bf16[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand: "bf16[128, 6, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        view_1: "bf16[768, 49, 49]" = torch.ops.aten.view.default(expand, _shape_param_2);  expand = _shape_param_2 = None
        permute: "bf16[768, 49, 49]" = torch.ops.aten.permute.default(view_1, [0, 2, 1])
        return (bmm, amax, amax_1, logical_not_1, sum_1, view_1, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

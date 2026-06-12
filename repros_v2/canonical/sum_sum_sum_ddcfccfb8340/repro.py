"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: ddcfccfb8340
Shape hash: 8185fd2d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 320, 56, 56]", arg1_1: "bf16[128, 320, 56, 56]", arg2_1: "f32[128, 320, 1, 1]", arg3_1: "f32[128, 1, 1, 1]", arg4_1: "f32[320]", arg5_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 320, 56, 56]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "bf16[128, 320, 56, 56]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.bfloat16)
        convert_element_type_2: "f32[128, 320, 56, 56]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_1: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476)
        erf: "f32[128, 320, 56, 56]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[128, 320, 56, 56]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul, add);  mul = None
        convert_element_type_3: "bf16[128, 320, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        div: "f32[128, 320, 1, 1]" = torch.ops.aten.div.Tensor(arg2_1, arg3_1)
        mul_3: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_3, div)
        mul_4: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Scalar(mul_3, 1);  mul_3 = None
        mul_5: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_4);  mul_4 = None
        view: "f32[1, 320, 1, 1]" = torch.ops.aten.view.default(arg4_1, [1, -1, 1, 1]);  arg4_1 = None
        mul_6: "f32[1, 320, 1, 1]" = torch.ops.aten.mul.Scalar(view, 1);  view = None
        mul_7: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_6);  mul_6 = None
        sum_1: "f32[1, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 2, 3], True, dtype = torch.float32);  convert_element_type = None
        sum_2: "f32[1, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 2, 3], True, dtype = torch.float32);  mul_5 = None
        mul_8: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, convert_element_type_3)
        mul_9: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, div);  mul_7 = None
        convert_element_type_4: "bf16[128, 320, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        sum_3: "f32[128, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2, 3], True, dtype = torch.float32);  mul_8 = None
        add_1: "bf16[128, 320, 56, 56]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_4);  convert_element_type_1 = convert_element_type_4 = None
        view_1: "f32[320]" = torch.ops.aten.view.default(sum_2, _shape_param_0);  sum_2 = _shape_param_0 = None
        view_2: "f32[320]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        div_1: "f32[128, 320, 1, 1]" = torch.ops.aten.div.Tensor(div, arg3_1);  div = None
        neg: "f32[128, 320, 1, 1]" = torch.ops.aten.neg.default(sum_3)
        mul_10: "f32[128, 320, 1, 1]" = torch.ops.aten.mul.Tensor(neg, div_1);  neg = div_1 = None
        div_2: "f32[128, 320, 1, 1]" = torch.ops.aten.div.Tensor(sum_3, arg3_1);  sum_3 = arg3_1 = None
        sum_4: "f32[128, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_10, [1], True, dtype = torch.float32);  mul_10 = None
        expand: "f32[128, 320, 1, 1]" = torch.ops.aten.expand.default(sum_4, _shape_param_2);  sum_4 = _shape_param_2 = None
        div_3: "f32[128, 320, 1, 1]" = torch.ops.aten.div.Scalar(expand, 320);  expand = None
        add_2: "f32[128, 320, 1, 1]" = torch.ops.aten.add.Tensor(div_2, div_3);  div_2 = div_3 = None
        div_4: "f32[128, 320, 56, 56]" = torch.ops.aten.div.Tensor(convert_element_type_3, arg2_1);  convert_element_type_3 = None
        eq: "b8[128, 320, 1, 1]" = torch.ops.aten.eq.Scalar(arg2_1, 0);  arg2_1 = None
        where: "f32[128, 320, 56, 56]" = torch.ops.aten.where.self(eq, arg5_1, div_4);  eq = arg5_1 = div_4 = None
        clone: "f32[128, 320, 56, 56]" = torch.ops.aten.clone.default(where, memory_format = torch.contiguous_format);  where = None
        mul_11: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(add_2, clone);  add_2 = clone = None
        convert_element_type_5: "bf16[128, 320, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        add_3: "bf16[128, 320, 56, 56]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_5);  add_1 = convert_element_type_5 = None
        convert_element_type_6: "f32[128, 320, 56, 56]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        mul_12: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_13: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type_2)
        mul_14: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, -0.5);  mul_13 = None
        exp: "f32[128, 320, 56, 56]" = torch.ops.aten.exp.default(mul_14);  mul_14 = None
        mul_15: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_16: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_15);  convert_element_type_2 = mul_15 = None
        add_4: "f32[128, 320, 56, 56]" = torch.ops.aten.add.Tensor(mul_12, mul_16);  mul_12 = mul_16 = None
        mul_17: "f32[128, 320, 56, 56]" = torch.ops.aten.mul.Tensor(convert_element_type_6, add_4);  convert_element_type_6 = add_4 = None
        convert_element_type_7: "bf16[128, 320, 56, 56]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        sum_5: "bf16[320]" = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0, 2, 3])
        convert_element_type_8: "f32[320]" = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        return (view_1, view_2, convert_element_type_7, convert_element_type_8)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

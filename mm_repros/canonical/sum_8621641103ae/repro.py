"""
Standalone repro captured via capture_hook.
Label: visformer_small
Pattern hash: 8621641103ae
Shape hash: d2a73ce0
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
    def forward(self, arg0_1: "bf16[128, 768, 7, 7]", arg1_1: "bf16[768, 49, 49]", arg2_1: "bf16[768, 128, 49]", arg3_1: "bf16[768, 56, 56]", arg4_1: "f32[128, 6, 49, 1]", arg5_1: "f32[128, 6, 49, 1]", arg6_1: "b8[128, 6, 49, 1]", arg7_1: "f32[128, 6, 49, 1]", arg8_1: "bf16[768, 128, 49]", arg9_1: "bf16[768, 49, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        view: "bf16[128, 6, 128, 49]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[128, 6, 49, 128]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2]);  view = None
        clone: "bf16[128, 6, 49, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[768, 49, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        bmm: "bf16[768, 49, 128]" = torch.ops.aten.bmm.default(arg1_1, view_1);  arg1_1 = None
        bmm_1: "bf16[768, 49, 49]" = torch.ops.aten.bmm.default(view_1, arg2_1);  view_1 = arg2_1 = None
        view_2: "bf16[128, 6, 49, 128]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        view_3: "bf16[128, 6, 49, 49]" = torch.ops.aten.view.default(bmm_1, _shape_param_3);  bmm_1 = _shape_param_3 = None
        convert_element_type: "f32[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        slice_1: "bf16[768, 49, 56]" = torch.ops.aten.slice.Tensor(arg3_1, 1, 0, -7);  arg3_1 = None
        slice_2: "bf16[768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view_4: "bf16[128, 6, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_4);  slice_2 = _shape_param_4 = None
        mul: "bf16[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_4, 0.08838834764831845)
        convert_element_type_1: "f32[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_2: "f32[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_1: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        sub: "f32[128, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        mul_2: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub, 0.08838834764831845);  sub = None
        sub_1: "f32[128, 6, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg5_1);  convert_element_type_1 = arg5_1 = None
        where: "f32[128, 6, 49, 49]" = torch.ops.aten.where.self(arg6_1, mul_2, sub_1);  arg6_1 = mul_2 = sub_1 = None
        exp: "f32[128, 6, 49, 49]" = torch.ops.aten.exp.default(where);  where = None
        div: "f32[128, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp, arg7_1);  exp = arg7_1 = None
        mul_3: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = None
        sum_1: "f32[128, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [-1], True)
        neg: "f32[128, 6, 49, 49]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[128, 6, 49, 49]" = torch.ops.prims.fma.default(neg, sum_1, mul_3);  neg = sum_1 = mul_3 = None
        convert_element_type_3: "bf16[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        mul_4: "bf16[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.08838834764831845);  convert_element_type_3 = None
        view_5: "bf16[768, 49, 49]" = torch.ops.aten.view.default(mul_4, _shape_param_5);  mul_4 = _shape_param_5 = None
        bmm_2: "bf16[768, 128, 49]" = torch.ops.aten.bmm.default(arg8_1, view_5);  arg8_1 = None
        bmm_3: "bf16[768, 49, 128]" = torch.ops.aten.bmm.default(view_5, arg9_1);  view_5 = arg9_1 = None
        view_6: "bf16[128, 6, 128, 49]" = torch.ops.aten.view.default(bmm_2, _shape_param_6);  bmm_2 = _shape_param_6 = None
        view_7: "bf16[128, 6, 49, 128]" = torch.ops.aten.view.default(bmm_3, _shape_param_7);  bmm_3 = _shape_param_7 = None
        permute_1: "bf16[128, 6, 49, 128]" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 2]);  view_6 = None
        cat: "bf16[384, 6, 49, 128]" = torch.ops.aten.cat.default([view_7, permute_1, view_2]);  view_7 = permute_1 = view_2 = None
        view_8: "bf16[3, 128, 6, 49, 128]" = torch.ops.aten.view.default(cat, _shape_param_8);  cat = _shape_param_8 = None
        permute_2: "bf16[128, 3, 6, 128, 49]" = torch.ops.aten.permute.default(view_8, [1, 0, 2, 4, 3]);  view_8 = None
        clone_1: "bf16[128, 3, 6, 128, 49]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_9: "bf16[128, 2304, 7, 7]" = torch.ops.aten.view.default(clone_1, _shape_param_9);  clone_1 = _shape_param_9 = None
        return view_9



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

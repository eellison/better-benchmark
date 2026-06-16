"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: f8e425e72787
Shape hash: df5bbfd4
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
    def forward(self, arg0_1: "bf16[768, 200, 200]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        slice_1: "bf16[768, 196, 200]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, -4);  arg0_1 = None
        slice_2: "bf16[768, 196, 196]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -4);  slice_1 = None
        view: "bf16[128, 6, 196, 196]" = torch.ops.aten.view.default(slice_2, _shape_param_0);  slice_2 = _shape_param_0 = None
        mul: "bf16[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view, 0.125)
        convert_element_type: "f32[128, 6, 196, 196]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_1: "f32[128, 6, 196, 196]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul_1: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        amax: "f32[128, 6, 196, 1]" = torch.ops.aten.amax.default(mul_1, [-1], True)
        sub: "f32[128, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_1, amax);  mul_1 = None
        mul_2: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub, 0.125);  sub = None
        amax_1: "f32[128, 6, 196, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub_1: "f32[128, 6, 196, 196]" = torch.ops.aten.sub.Tensor(convert_element_type, amax_1)
        abs_1: "f32[128, 6, 196, 196]" = torch.ops.aten.abs.default(convert_element_type)
        ne: "b8[128, 6, 196, 196]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        eq: "b8[128, 6, 196, 196]" = torch.ops.aten.eq.Tensor(convert_element_type, convert_element_type);  convert_element_type = None
        mul_3: "b8[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(eq, ne);  eq = ne = None
        logical_not: "b8[128, 6, 196, 196]" = torch.ops.aten.logical_not.default(mul_3);  mul_3 = None
        any_1: "b8[128, 6, 196, 1]" = torch.ops.aten.any.dims(logical_not, [-1], True);  logical_not = None
        logical_not_1: "b8[128, 6, 196, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        where: "f32[128, 6, 196, 196]" = torch.ops.aten.where.self(logical_not_1, mul_2, sub_1);  mul_2 = sub_1 = None
        exp: "f32[128, 6, 196, 196]" = torch.ops.aten.exp.default(where);  where = None
        sum_1: "f32[128, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[128, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_2: "bf16[128, 6, 196, 196]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand: "bf16[128, 6, 196, 196]" = torch.ops.aten.expand.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        view_1: "bf16[768, 196, 196]" = torch.ops.aten.view.default(expand, _shape_param_2);  expand = _shape_param_2 = None
        constant_pad_nd: "bf16[768, 200, 200]" = torch.ops.aten.constant_pad_nd.default(view_1, _shape_param_3);  _shape_param_3 = None
        permute: "bf16[768, 196, 196]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return (amax, amax_1, logical_not_1, sum_1, constant_pad_nd, permute)



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

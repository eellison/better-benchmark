"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: cd8a1b4cf13b
Shape hash: 56ba83ca
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
    def forward(self, arg0_1: "b8[16, 1, 128, 128]", arg1_1: "bf16[]", arg2_1: "bf16[]", arg3_1: "bf16[512, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        where: "bf16[16, 1, 128, 128]" = torch.ops.aten.where.self(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        add: "bf16[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view, where);  view = where = None
        convert_element_type: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        amax: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add, -inf);  add = None
        logical_not: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[16, 32, 128, 128]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        where_1: "bf16[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_1);  logical_not_1 = full = convert_element_type_1 = None
        expand: "bf16[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_1, _shape_param_2);  _shape_param_2 = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_3);  expand = _shape_param_3 = None
        return (where_1, view_1)



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

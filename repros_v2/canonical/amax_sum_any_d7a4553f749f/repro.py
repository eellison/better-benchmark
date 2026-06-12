"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train
Pattern hash: d7a4553f749f
Shape hash: b31e9601
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
    def forward(self, arg0_1: "bf16[512, 512, 512]", arg1_1: "bf16[8, 64, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        amax: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view, -inf);  view = None
        logical_not: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        where: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_1, arg1_1, convert_element_type_1);  logical_not_1 = arg1_1 = convert_element_type_1 = None
        expand: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where, _shape_param_1);  _shape_param_1 = None
        view_1: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand, _shape_param_2);  expand = _shape_param_2 = None
        return (where, view_1)



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

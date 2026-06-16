"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer
Pattern hash: b2700cd0ebd1
Shape hash: ec934f37
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
    def forward(self, arg0_1: "bf16[128, 2304, 9, 9]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 2304, 9, 9]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        neg: "f32[128, 2304, 9, 9]" = torch.ops.aten.neg.default(convert_element_type)
        exp: "f32[128, 2304, 9, 9]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[128, 2304, 9, 9]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 2304, 9, 9]" = torch.ops.aten.div.Tensor(convert_element_type, add);  convert_element_type = add = None
        convert_element_type_1: "bf16[128, 2304, 9, 9]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mean: "bf16[128, 2304, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_1, [-1, -2], True);  convert_element_type_1 = None
        as_strided: "bf16[128, 2304, 1, 1]" = torch.ops.aten.as_strided.default(mean, _shape_param_0, _shape_param_1);  mean = _shape_param_0 = _shape_param_1 = None
        view: "bf16[128, 2304]" = torch.ops.aten.view.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        return view



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

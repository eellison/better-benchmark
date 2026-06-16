"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer
Pattern hash: 15e408415993
Shape hash: fd0f60cd
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
    def forward(self, arg0_1: "bf16[8192, 480]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[512, 16, 480]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[512, 16, 480]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        neg: "f32[512, 16, 480]" = torch.ops.aten.neg.default(convert_element_type)
        exp: "f32[512, 16, 480]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[512, 16, 480]" = torch.ops.aten.div.Tensor(convert_element_type, add);  convert_element_type = add = None
        convert_element_type_1: "bf16[512, 16, 480]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view_1: "bf16[8192, 480]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        return view_1



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

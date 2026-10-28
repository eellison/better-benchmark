"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/LayoutLMForMaskedLM/full_graph_000.py
Pattern hash: 079dfeb381b2
Shape hash: 41c7b013
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
    def forward(self, arg0_1: "bf16[384, 512, 64]", arg1_1: "bf16[384, 64, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        bmm: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        mul: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view, 0.125);  view = None
        convert_element_type: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        abs_1: "f32[32, 12, 512, 512]" = torch.ops.aten.abs.default(convert_element_type)
        ne: "b8[32, 12, 512, 512]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        eq: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type, convert_element_type)
        mul_1: "b8[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(eq, ne);  eq = ne = None
        logical_not: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(mul_1);  mul_1 = None
        return (convert_element_type, logical_not)



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

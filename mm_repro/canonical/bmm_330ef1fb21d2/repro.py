"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/AlbertForMaskedLM/full_graph_000.py
Pattern hash: 330ef1fb21d2
Shape hash: 5bac3ca8
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
    def forward(self, arg0_1: "bf16[512, 512, 64]", arg1_1: "bf16[512, 64, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        eq: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view, -inf);  view = None
        logical_not: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        return logical_not



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

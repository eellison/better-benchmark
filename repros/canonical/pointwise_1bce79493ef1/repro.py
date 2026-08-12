"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_infer
Pattern hash: 1bce79493ef1
Shape hash: 0fc085ac
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
    def forward(self, arg0_1: "bf16[30522, 128]", arg1_1: "bf16[384, 30522]", _shape_param_0):
        # No stacktrace found for following nodes
        permute: "bf16[128, 30522]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        cat: "bf16[512, 30522]" = torch.ops.aten.cat.default([permute, arg1_1]);  permute = arg1_1 = None
        constant_pad_nd: "bf16[512, 30528]" = torch.ops.aten.constant_pad_nd.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        return constant_pad_nd



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

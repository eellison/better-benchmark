"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train
Pattern hash: 7551a58e5d34
Shape hash: 846b3407
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
    def forward(self, arg0_1: "bf16[32, 512, 50257]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[16384, 50257]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[50257, 16384]" = torch.ops.aten.permute.default(view, [1, 0])
        constant_pad_nd: "bf16[50264, 16384]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        constant_pad_nd_1: "bf16[16384, 50264]" = torch.ops.aten.constant_pad_nd.default(view, _shape_param_2);  view = _shape_param_2 = None
        return (constant_pad_nd, constant_pad_nd_1)



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

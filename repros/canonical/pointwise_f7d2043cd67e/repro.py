"""
Standalone repro captured via capture_hook.
Label: hf_google/gemma-2-2b_infer
Pattern hash: f7d2043cd67e
Shape hash: 736b279f
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
    def forward(self, arg0_1: "bf16[1000, 256000]", _shape_param_0):
        # No stacktrace found for following nodes
        view: "bf16[1, 1000, 256000]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        div: "bf16[1, 1000, 256000]" = torch.ops.aten.div.Tensor(view, 30.0);  view = None
        tanh: "bf16[1, 1000, 256000]" = torch.ops.aten.tanh.default(div);  div = None
        mul: "bf16[1, 1000, 256000]" = torch.ops.aten.mul.Tensor(tanh, 30.0);  tanh = None
        return mul



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

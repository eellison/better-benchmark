"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_train
Pattern hash: a888b5ed0623
Shape hash: 56e670a4
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
    def forward(self, arg0_1: "i64[1, 128]", arg1_1: "f32[2050, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        add: "i64[1, 128]" = torch.ops.aten.add.Tensor(arg0_1, 2);  arg0_1 = None
        view: "i64[128]" = torch.ops.aten.view.default(add, [-1]);  add = None
        index: "f32[128, 1024]" = torch.ops.aten.index.Tensor(arg1_1, [view]);  arg1_1 = view = None
        view_1: "f32[1, 128, 1024]" = torch.ops.aten.view.default(index, _shape_param_0);  index = _shape_param_0 = None
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

"""
Standalone repro captured via capture_hook.
Label: torchbench_LearningToPaint_infer
Pattern hash: 246850b7c198
Shape hash: d69ae404
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1024, 65], f32))"

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[1024, 65]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:136 in forward, code: x = torch.sigmoid(x)
        sigmoid_default: "f32[1024, 65]" = torch.ops.aten.sigmoid.default(addmm);  addmm = None
        return sigmoid_default



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

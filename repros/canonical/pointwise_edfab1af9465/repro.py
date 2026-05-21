"""
Standalone repro captured via capture_hook.
Label: torchbench_lennard_jones_infer
Pattern hash: edfab1af9465
Shape hash: 0a019d11
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
_shapes_config = "(T([1000, 16], f32))"

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f32[1000, 16]"):
        # No stacktrace found for following nodes
        tanh_default: "f32[1000, 16]" = torch.ops.aten.tanh.default(addmm_2);  addmm_2 = None
        return tanh_default



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

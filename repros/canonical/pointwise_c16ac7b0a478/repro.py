"""
Standalone repro captured via capture_hook.
Label: torchbench_lennard_jones_infer
Pattern hash: c16ac7b0a478
Shape hash: 3bfc4322
"""
import sys
from pathlib import Path

import sys
from pathlib import Path
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f32[1000, 16]", arg9_1: "f32[1, 16]"):
        # No stacktrace found for following nodes
        tanh_default: "f32[1000, 16]" = torch.ops.aten.tanh.default(addmm_2);  addmm_2 = None
        permute_default: "f32[16, 1]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        return (tanh_default, permute_default)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

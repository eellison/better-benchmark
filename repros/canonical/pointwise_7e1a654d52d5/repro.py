"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_infer
Pattern hash: 7e1a654d52d5
Shape hash: 9d56daaf
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
_shapes_config = "(T([8, 4096, 92], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_11: "f32[8, 4096, 92]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:215 in forward, code: x = encode(x)
        glu_default: "f32[8, 2048, 92]" = torch.ops.aten.glu.default(convolution_11, 1);  convolution_11 = None
        return glu_default



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

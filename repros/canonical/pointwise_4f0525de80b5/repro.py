"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train_003
Pattern hash: 4f0525de80b5
Shape hash: 16387f75
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
_shapes_config = "(T([64, 128, 4096], f32))"

class Repro(torch.nn.Module):
    def forward(self, relu: "f32[64, 128, 4096]"):
        # No stacktrace found for following nodes
        le_scalar: "b8[64, 128, 4096]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return le_scalar



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

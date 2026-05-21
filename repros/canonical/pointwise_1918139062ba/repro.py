"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_000
Pattern hash: 1918139062ba
Shape hash: 8f3ea597
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
_shapes_config = "(T([8, 512, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, rsqrt_24: "f32[8, 512, 1]"):
        # No stacktrace found for following nodes
        div_tensor: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 4096);  rsqrt_24 = None
        return div_tensor



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

"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train_000
Pattern hash: 0eade0de46c0
Shape hash: d7517139
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
_shapes_config = "()"

class Repro(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        full_default: "f32[2, 1, 1, 1024]" = torch.ops.aten.full.default([2, 1, 1, 1024], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_int: "f32[2, 1, 1024]" = torch.ops.aten.select.int(full_default, 1, 0);  full_default = None
        select_int_1: "f32[2, 1024]" = torch.ops.aten.select.int(select_int, 1, 0);  select_int = None
        return select_int_1



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

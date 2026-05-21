"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer
Pattern hash: d69532cc2657
Shape hash: befc7a09
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
_shapes_config = "(T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg7_1: "f32[64, 1024, 8, 64]", slice_scatter_1: "f32[64, 1024, 8, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        copy__default: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg7_1, slice_scatter_1);  arg7_1 = slice_scatter_1 = None
        return copy__default



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

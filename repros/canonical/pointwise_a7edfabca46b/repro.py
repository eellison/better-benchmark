"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer
Pattern hash: a7edfabca46b
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
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_default: "i32[1, 1]" = torch.ops.aten.full.default([1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        return full_default



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

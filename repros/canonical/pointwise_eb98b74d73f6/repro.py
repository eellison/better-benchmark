"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train
Pattern hash: eb98b74d73f6
Shape hash: 7534eecf
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
_shapes_config = "(T([64, 512, 1452], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_21: "f32[64, 512, 1452]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        full_default: "f32[64, 512, 1493]" = torch.ops.aten.full.default([64, 512, 1493], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[64, 512, 1493]" = torch.ops.aten.slice_scatter.default(full_default, getitem_21, 2, 20, -21);  full_default = getitem_21 = None
        return slice_scatter_default



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

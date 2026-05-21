"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: b67c65f00e4d
Shape hash: 3c1b3005
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
_shapes_config = "(T([1024, 384, 13, 13], f32, stride=(64896, 1, 4992, 384)), T([], f32), T([1024, 384, 13, 13], f32, stride=(64896, 1, 4992, 384)))"

class Repro(torch.nn.Module):
    def forward(self, relu_2: "f32[1024, 384, 13, 13]", full_default: "f32[]", getitem_9: "f32[1024, 384, 13, 13]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        le_scalar: "b8[1024, 384, 13, 13]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_self: "f32[1024, 384, 13, 13]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_9);  le_scalar = full_default = getitem_9 = None
        return where_self



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

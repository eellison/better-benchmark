"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train
Pattern hash: 0828bb1e27fe
Shape hash: 0db2286d
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
_shapes_config = "(T([64, 32, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_5: "f32[64, 32, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        squeeze_dims: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_5, [2, 3]);  getitem_5 = None
        return squeeze_dims



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

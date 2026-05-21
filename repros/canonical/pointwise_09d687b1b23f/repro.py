"""
Standalone repro captured via capture_hook.
Label: torchbench_dcgan_train_001
Pattern hash: 09d687b1b23f
Shape hash: d2b3adb9
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
_shapes_config = "(T([1024, 64, 32, 32], f32), T([1024, 64, 32, 32], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg9_1: "f32[1024, 64, 32, 32]", getitem_9: "f32[1024, 64, 32, 32]"):
        # No stacktrace found for following nodes
        gt_scalar: "b8[1024, 64, 32, 32]" = torch.ops.aten.gt.Scalar(arg9_1, 0);  arg9_1 = None
        mul_tensor: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_9, 0.2)
        where_self: "f32[1024, 64, 32, 32]" = torch.ops.aten.where.self(gt_scalar, getitem_9, mul_tensor);  gt_scalar = getitem_9 = mul_tensor = None
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

"""
Standalone repro captured via capture_hook.
Label: torchbench_lennard_jones_train
Pattern hash: 2609cfb0f942
Shape hash: 45cd6de0
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
_shapes_config = "(T([1000, 16], f32), T([1000, 16], f32))"

class Repro(torch.nn.Module):
    def forward(self, tanh_1: "f32[1000, 16]", mm_3: "f32[1000, 16]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_tensor: "f32[1000, 16]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_1: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(mm_3, sub_tensor);  mm_3 = sub_tensor = None
        return mul_tensor_1



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

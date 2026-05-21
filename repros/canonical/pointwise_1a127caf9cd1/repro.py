"""
Standalone repro captured via capture_hook.
Label: torchbench_dcgan_train
Pattern hash: 1a127caf9cd1
Shape hash: 3cc64220
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
_shapes_config = "(T([512], f32), T([512], f32), T([256], f32), T([256], f32), T([128], f32), T([128], f32))"

class Repro(torch.nn.Module):
    def forward(self, sum_2: "f32[512]", squeeze_7: "f32[512]", sum_4: "f32[256]", squeeze_4: "f32[256]", sum_6: "f32[128]", squeeze_1: "f32[128]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:128 in forward, code: return self.main(input)
        mul_tensor: "f32[512]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_7);  sum_2 = squeeze_7 = None
        mul_tensor_1: "f32[256]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_4);  sum_4 = squeeze_4 = None
        mul_tensor_2: "f32[128]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_1);  sum_6 = squeeze_1 = None
        return (mul_tensor, mul_tensor_1, mul_tensor_2)



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

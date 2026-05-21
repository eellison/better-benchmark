"""
Standalone repro captured via capture_hook.
Label: torchbench_dcgan_infer
Pattern hash: 4827c8cbb45d
Shape hash: 38702c6e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution: "f32[1024, 64, 32, 32]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:128 in forward, code: return self.main(input)
        gt_scalar: "b8[1024, 64, 32, 32]" = torch.ops.aten.gt.Scalar(convolution, 0)
        mul_tensor: "f32[1024, 64, 32, 32]" = torch.ops.aten.mul.Tensor(convolution, 0.2)
        where_self: "f32[1024, 64, 32, 32]" = torch.ops.aten.where.self(gt_scalar, convolution, mul_tensor);  gt_scalar = convolution = mul_tensor = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn([1024, 64, 32, 32], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""
Standalone repro captured via capture_hook.
Label: torchbench_dcgan_train
Pattern hash: ef7782270e46
Shape hash: 05e93948
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
    def forward(self, sigmoid: "f32[1024, 1, 1, 1]", tangents_1: "f32[1024, 1, 1, 1]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:128 in forward, code: return self.main(input)
        sub_tensor: "f32[1024, 1, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_tensor: "f32[1024, 1, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid, sub_tensor);  sigmoid = sub_tensor = None
        mul_tensor_1: "f32[1024, 1, 1, 1]" = torch.ops.aten.mul.Tensor(tangents_1, mul_tensor);  tangents_1 = mul_tensor = None
        return mul_tensor_1


def _default_make_inputs():
    return [
    torch.randn([1024, 1, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1, 1, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

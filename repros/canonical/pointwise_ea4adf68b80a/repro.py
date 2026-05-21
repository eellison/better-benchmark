"""
Standalone repro captured via capture_hook.
Label: torchbench_opacus_cifar10_infer
Pattern hash: ea4adf68b80a
Shape hash: c4ec4045
"""
import sys
from pathlib import Path

import sys
from pathlib import Path
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[64, 64, 8, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        relu_default: "f32[64, 64, 8, 8]" = torch.ops.aten.relu.default(arg2_1)
        copy__default: "f32[64, 64, 8, 8]" = torch.ops.aten.copy_.default(arg2_1, relu_default);  arg2_1 = relu_default = None
        return copy__default


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

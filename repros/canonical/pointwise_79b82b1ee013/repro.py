"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_infer
Pattern hash: 79b82b1ee013
Shape hash: 0b6b69f4
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
    def forward(self, convolution_9: "f16[128, 512, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        relu_default: "f16[128, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem: "f16[128, 512, 14, 14]" = _low_memory_max_pool_with_offsets_default[0];  _low_memory_max_pool_with_offsets_default = None
        return getitem


def _default_make_inputs():
    return [
    torch.randn([128, 512, 28, 28], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

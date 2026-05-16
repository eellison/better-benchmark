"""
Standalone repro captured via capture_hook.
Label: vgg16_inference
Pattern hash: 1128aaa6f6c7
Shape hash: a43b1211
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_12: "f32[1, 512, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        relu_default: "f32[1, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem: "f32[1, 512, 7, 7]" = _low_memory_max_pool_with_offsets_default[0];  _low_memory_max_pool_with_offsets_default = None
        return getitem


def _default_make_inputs():
    return [
    torch.randn([1, 512, 14, 14], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

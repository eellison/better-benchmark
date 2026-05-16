"""
Standalone repro captured via capture_hook.
Label: vgg16_inference
Pattern hash: 1452242c7662
Shape hash: aa505336
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_1: "f32[1, 4096]", arg31_1: "f32[1000, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        relu_default: "f32[1, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_default: "f32[4096, 1000]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        return (relu_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 4096], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""
Standalone repro captured via capture_hook.
Label: vgg16_inference
Pattern hash: caf00befef88
Shape hash: 0b087e78
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_8: "f32[1, 512, 7, 7]", _shape_param_0, arg27_1: "f32[4096, 25088]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:67 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d_default: "f32[1, 512, 7, 7]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_8, [7, 7]);  getitem_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:68 in forward, code: x = torch.flatten(x, 1)
        reshape_default: "f32[1, 25088]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d_default, _shape_param_0);  _adaptive_avg_pool2d_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        permute_default: "f32[25088, 4096]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1, 512, 7, 7], dtype=torch.float32, device='cuda'),
    [1, 25088],  # _shape_param_0
    torch.randn([4096, 25088], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

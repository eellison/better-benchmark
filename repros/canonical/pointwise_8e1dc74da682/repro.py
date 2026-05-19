"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_infer
Pattern hash: 8e1dc74da682
Shape hash: d27d17e3
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
    def forward(self, convolution_42: "f32[128, 640, 7, 7]", convolution_39: "f32[128, 640, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[128, 640, 7, 7]" = torch.ops.aten.add.Tensor(convolution_42, convolution_39);  convolution_42 = convolution_39 = None
        return add_tensor


def _default_make_inputs():
    return [
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([128, 640, 7, 7], [31360, 1, 4480, 640]),  # convolution_42
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([128, 640, 7, 7], [31360, 1, 4480, 640]),  # convolution_39
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

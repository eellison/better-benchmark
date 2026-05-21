"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_infer
Pattern hash: 0a306c604828
Shape hash: 59e61e38
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
    def forward(self, convolution_23: "f16[512, 256, 13, 13]", convolution_24: "f16[512, 256, 13, 13]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        relu_default: "f16[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        relu_default_1: "f16[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_default: "f16[512, 512, 13, 13]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1);  relu_default = relu_default_1 = None
        return cat_default


def _default_make_inputs():
    return [
    torch.randn([512, 256, 13, 13], dtype=torch.float16, device='cuda'),
    torch.randn([512, 256, 13, 13], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

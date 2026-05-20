"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_infer
Pattern hash: c4dd226e9b97
Shape hash: 04016629
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 1280, 1, 1], f32), T([1000, 1280], f32), S([512, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_94: "f32[512, 1280, 1, 1]", arg431_1: "f32[1000, 1280]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:833 in forward_head, code: x = self.act2(x)
        relu_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.relu.default(convolution_94);  convolution_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:834 in forward_head, code: x = self.flatten(x)
        reshape_default: "f32[512, 1280]" = torch.ops.aten.reshape.default(relu_default, _shape_param_0);  relu_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_default: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        return (reshape_default, permute_default)


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

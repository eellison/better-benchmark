"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_infer
Pattern hash: 268b1185b97f
Shape hash: 692983dc
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 1000, 13, 13], f16), S([512, 1000]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_25: "f16[512, 1000, 13, 13]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        relu_default: "f16[512, 1000, 13, 13]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
        mean_dim: "f16[512, 1000, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:97 in forward, code: return torch.flatten(x, 1)
        reshape_default: "f16[512, 1000]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        return reshape_default



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

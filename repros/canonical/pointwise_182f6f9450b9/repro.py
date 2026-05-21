"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_regnet_infer
Pattern hash: 182f6f9450b9
Shape hash: de982d11
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
_shapes_config = "(T([32, 2240, 1, 1], f16), T([32, 2240, 7, 7], f16))"

class Repro(torch.nn.Module):
    def forward(self, convolution_97: "f16[32, 2240, 1, 1]", relu_74: "f16[32, 2240, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f16[32, 2240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_97);  convolution_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f16[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(relu_74, sigmoid_default);  relu_74 = sigmoid_default = None
        return mul_tensor



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

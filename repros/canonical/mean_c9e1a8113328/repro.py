"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer
Pattern hash: c9e1a8113328
Shape hash: 5e38a2dd
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 1536, 7, 7], f32, stride=(75264, 1, 10752, 1536)))"

class Repro(torch.nn.Module):
    def forward(self, convolution_77: "f32[128, 1536, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_77, [2, 3], True);  convolution_77 = None
        return mean_dim


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

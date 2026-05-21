"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_infer
Pattern hash: ac1a00c68785
Shape hash: 17204068
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
_shapes_config = "(T([128, 1536, 12, 12], f32, stride=(221184, 1, 18432, 1536)))"

class Repro(torch.nn.Module):
    def forward(self, mul_334: "f32[128, 1536, 12, 12]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_default: "f32[128, 1536, 6, 6]" = torch.ops.aten.avg_pool2d.default(mul_334, [2, 2], [2, 2], [0, 0], True, False);  mul_334 = None
        return avg_pool2d_default



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

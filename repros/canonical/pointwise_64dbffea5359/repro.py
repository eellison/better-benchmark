"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 64dbffea5359
Shape hash: 1ae687b6
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
_shapes_config = "(T([128, 2048, 8, 8], f32, stride=(131072, 1, 16384, 2048)))"

class Repro(torch.nn.Module):
    def forward(self, cat_11: "f32[128, 2048, 8, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:236 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_default: "f32[128, 2048, 8, 8]" = torch.ops.aten.avg_pool2d.default(cat_11, [3, 3], [1, 1], [1, 1]);  cat_11 = None
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

"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: b67c65f00e4d
Shape hash: f1dac5f3
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
_shapes_config = "(T([128, 128, 1, 1], f32), T([], f32), T([128, 128, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, relu: "f32[128, 128, 1, 1]", full_default: "f32[]", getitem_324: "f32[128, 128, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_scalar: "b8[128, 128, 1, 1]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_self: "f32[128, 128, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_324);  le_scalar = full_default = getitem_324 = None
        return where_self



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

"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: 02528bc5e96e
Shape hash: 7b090fc4
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 384, 1, 1], f32), T([128, 384, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, relu_11: "f32[128, 384, 1, 1]", getitem_117: "f32[128, 384, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_scalar: "b8[128, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 384, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_117);  le_scalar = full_default = getitem_117 = None
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

"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 801957297c4b
Shape hash: 7d40d29c
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
_shapes_config = "(T([1, 16, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_1: "f32[1, 16, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        squeeze_dims: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2]);  getitem_1 = None
        unsqueeze_default: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        return unsqueeze_default_1



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

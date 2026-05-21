"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 4832707ab8a5
Shape hash: 63ad8c2a
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
_shapes_config = "(T([192], f32), T([192], f32))"

class Repro(torch.nn.Module):
    def forward(self, sum_3: "f32[192]", squeeze_280: "f32[192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor: "f32[192]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_280);  sum_3 = squeeze_280 = None
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

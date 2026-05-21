"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: c7a56ee42903
Shape hash: 08301c01
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
_shapes_config = "(T([128, 768, 14, 14], f32, stride=(151296, 1, 10752, 768)))"

class Repro(torch.nn.Module):
    def forward(self, view_362: "f32[128, 768, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_362, [0, 2, 3]);  view_362 = None
        return sum_dim_int_list



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

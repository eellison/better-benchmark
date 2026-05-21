"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: 1ea18851a876
Shape hash: 35d8f3da
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
_shapes_config = "(T([128, 197, 768], f32))"

class Repro(torch.nn.Module):
    def forward(self, add_133: "f32[128, 197, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:787 in forward_features, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        slice_tensor: "f32[128, 1, 768]" = torch.ops.aten.slice.Tensor(add_133, 1, 0, 1);  add_133 = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0], True);  slice_tensor = None
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

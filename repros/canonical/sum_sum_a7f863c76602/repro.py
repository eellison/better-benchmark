"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_train
Pattern hash: a7f863c76602
Shape hash: 6f5549c8
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
_shapes_config = "(T([128, 1000], f32), S([1000]), S([1000]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 1000]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:124 in forward_head, code: return (x + x_dist) / 2
        div_tensor: "f32[128, 1000]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:118 in forward_head, code: x_dist = self.head_dist(x_dist)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(div_tensor, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div_tensor, [0], True)
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:117 in forward_head, code: x = self.head(x)
        sum_dim_int_list_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div_tensor, [0], True);  div_tensor = None
        reshape_default_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        return (permute_default, reshape_default, reshape_default_1)



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

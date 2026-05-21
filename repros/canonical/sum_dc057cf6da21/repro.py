"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_train
Pattern hash: dc057cf6da21
Shape hash: bec85e89
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
_shapes_config = "(T([128, 768], f32), S([128, 1, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_10: "f32[128, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default: "f32[128, 1, 768]" = torch.ops.aten.reshape.default(mm_10, _shape_param_0);  mm_10 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:101 in forward, code: q_latent = self.latent.expand(B, -1, -1)
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0], True);  reshape_default = None
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

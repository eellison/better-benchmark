"""
Standalone repro captured via capture_hook.
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([2048, 8], i64, max=100))"

class Repro(torch.nn.Module):
    def forward(self, getitem_55: "i64[2048, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:386 in grouped_mm_experts_forward, code: expert_ids = top_k_index.reshape(-1)  # (S,)
        reshape_default: "i64[16384]" = torch.ops.aten.reshape.default(getitem_55, [-1]);  getitem_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:389 in grouped_mm_experts_forward, code: expert_ids_g, perm = torch.sort(expert_ids)
        sort_default = torch.ops.aten.sort.default(reshape_default);  reshape_default = None
        getitem: "i64[16384]" = sort_default[0]
        getitem_56: "i64[16384]" = sort_default[1];  sort_default = None
        return (getitem, getitem_56)


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

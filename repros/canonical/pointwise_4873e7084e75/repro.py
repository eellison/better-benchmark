"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_inference
Pattern hash: 4873e7084e75
Shape hash: d7517139
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:167 in _get_least_common_mult_chunk_len, code: return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
        full_default: "i64[]" = torch.ops.aten.full.default([], 64, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 64, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        return (full_default, full_default_1)


def _default_make_inputs():
    return [

    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

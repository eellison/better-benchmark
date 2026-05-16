"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, primals_7: "f32[1536, 1536]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:235 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_default_1: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return permute_default_1


def _default_make_inputs():
    return [
    torch.randn([1536, 1536], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

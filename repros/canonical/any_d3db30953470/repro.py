"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=any, ranges=[], reduction_ranges=[]
#   origins: ['aten.any.default']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1240 in forward, code: is_index_global_attn = attention_mask > 0
        gt_scalar: "b8[8, 1024]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1243 in forward, code: is_global_attn = is_index_global_attn.flatten().any().item()
        reshape_default: "b8[8192]" = torch.ops.aten.reshape.default(gt_scalar, [8192]);  gt_scalar = None
        any_default: "b8[]" = torch.ops.aten.any.default(reshape_default);  reshape_default = None
        return any_default


def _default_make_inputs():
    return [
    torch.randn([8, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

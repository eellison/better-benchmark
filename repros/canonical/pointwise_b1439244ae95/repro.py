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
    def forward(self, mm_default_1: "f32[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:416 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_1, 0);  mm_default_1 = None
        reshape_default: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.reshape.default(unsqueeze_default, [1024, 16, 64, 1, 1]);  unsqueeze_default = None
        squeeze_dim: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(reshape_default, 4);  reshape_default = None
        squeeze_dim_1: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_dim, 3);  squeeze_dim = None
        return squeeze_dim_1


def _default_make_inputs():
    return [
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

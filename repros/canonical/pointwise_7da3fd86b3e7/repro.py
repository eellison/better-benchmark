"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, primals_3: "f32[1024, 16, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:416 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, 3);  primals_3 = None
        unsqueeze_default_1: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        reshape_default: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_1, [1, 1024, 1024]);  unsqueeze_default_1 = None
        permute_default: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        squeeze_dim: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default, 0);  permute_default = None
        return squeeze_dim


def _default_make_inputs():
    return [
    torch.randn([1024, 16, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

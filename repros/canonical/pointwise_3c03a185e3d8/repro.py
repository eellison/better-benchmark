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
    def forward(self, bmm_543: "f32[256, 64, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:263 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        reshape_default: "f32[16, 16, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_543, [16, 16, 64, 512, 1]);  bmm_543 = None
        permute_default: "f32[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 4, 3, 2]);  reshape_default = None
        permute_default_1: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default, [3, 0, 1, 4, 2]);  permute_default = None
        squeeze_dim: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_1, 4);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:417 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        reshape_default_1: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim, [512, 16, 16, 64, 1]);  squeeze_dim = None
        permute_default_2: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 4, 2, 3]);  reshape_default_1 = None
        clone_default: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_2: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(clone_default, [1, 8192, 1024]);  clone_default = None
        squeeze_dim_1: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_2, 0);  reshape_default_2 = None
        return squeeze_dim_1


def _default_make_inputs():
    return [
    torch.randn([256, 64, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

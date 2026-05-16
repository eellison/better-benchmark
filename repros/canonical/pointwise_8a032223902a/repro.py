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
    def forward(self, bmm_190: "f32[256, 512, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:294 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        reshape_default: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.reshape.default(bmm_190, [16, 16, 512, 1, 64]);  bmm_190 = None
        permute_default: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [2, 0, 1, 4, 3]);  reshape_default = None
        reshape_default_1: "f32[512, 16, 16, 64]" = torch.ops.aten.reshape.default(permute_default, [512, 16, 16, 64]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, 4);  reshape_default_1 = None
        permute_default_1: "f32[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_default, [0, 1, 4, 3, 2]);  unsqueeze_default = None
        permute_default_2: "f32[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 4, 2]);  permute_default_1 = None
        clone_default: "f32[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_2: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(clone_default, [1, 8192, 1024]);  clone_default = None
        squeeze_dim: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_2, 0);  reshape_default_2 = None
        return squeeze_dim


def _default_make_inputs():
    return [
    torch.randn([256, 512, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

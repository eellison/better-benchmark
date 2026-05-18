"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '16', '64'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '16', '64'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_542: "f32[256, 512, 64]", bmm_544: "f32[256, 512, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:266 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default: "f32[16, 16, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_542, [16, 16, 512, 64, 1]);  bmm_542 = None
        permute_default: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 2, 4, 3]);  reshape_default = None
        permute_default_1: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default, [2, 0, 1, 4, 3]);  permute_default = None
        squeeze_dim: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_1, 4);  permute_default_1 = None
        sum_dim_int_list: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_dim, [0, 1], True)
        reshape_default_1: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list, [16, 64]);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:263 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        reshape_default_2: "f32[16, 16, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_544, [16, 16, 512, 64, 1]);  bmm_544 = None
        permute_default_2: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 2, 4, 3]);  reshape_default_2 = None
        permute_default_3: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_2, [2, 0, 1, 4, 3]);  permute_default_2 = None
        squeeze_dim_1: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_3, 4);  permute_default_3 = None
        sum_dim_int_list_1: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_dim_1, [0, 1], True)
        reshape_default_3: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [16, 64]);  sum_dim_int_list_1 = None
        add_tensor: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_dim, squeeze_dim_1);  squeeze_dim = squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:416 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        reshape_default_4: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(add_tensor, [512, 16, 16, 64, 1]);  add_tensor = None
        permute_default_4: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 1, 4, 2, 3]);  reshape_default_4 = None
        clone_default: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_5: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(clone_default, [1, 8192, 1024]);  clone_default = None
        squeeze_dim_2: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_5, 0);  reshape_default_5 = None
        return (reshape_default_1, reshape_default_3, squeeze_dim_2)


def _default_make_inputs():
    return [
    torch.randn([256, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([256, 512, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

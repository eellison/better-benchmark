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
    def forward(self, mm_default_4: "f32[8192, 1024]", arg351_1: "f32[16, 64]", arg352_1: "f32[16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:416 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_4, 0);  mm_default_4 = None
        reshape_default: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 4, 2]);  reshape_default = None
        reshape_default_1: "f32[512, 16, 16, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:263 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_tensor: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(reshape_default_1, arg351_1);  arg351_1 = None
        unsqueeze_default_1: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, 4);  add_tensor = None
        permute_default_1: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_1, [1, 2, 0, 4, 3]);  unsqueeze_default_1 = None
        permute_default_2: "f32[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 2, 4, 3]);  permute_default_1 = None
        reshape_default_2: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_2);  permute_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:266 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_tensor_1: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(reshape_default_1, arg352_1);  reshape_default_1 = arg352_1 = None
        unsqueeze_default_2: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 4);  add_tensor_1 = None
        permute_default_3: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_2, [1, 2, 0, 4, 3]);  unsqueeze_default_2 = None
        permute_default_4: "f32[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_default_3, [0, 1, 2, 4, 3]);  permute_default_3 = None
        reshape_default_3: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_3);  permute_default_4 = _shape_param_3 = None
        return (reshape_default_2, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 64], dtype=torch.float32, device='cuda'),
    torch.randn([16, 64], dtype=torch.float32, device='cuda'),
    [512, 16, 1, 16, 64],  # _shape_param_0
    [512, 16, 16, 64],  # _shape_param_1
    [256, 512, 64],  # _shape_param_2
    [256, 512, 64],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 70d3decb62c3
Shape hash: b2b5c60a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([256, 512, 64], f32), T([1024, 16, 64], f32), S([16, 16, 512, 1, 64]), S([512, 16, 16, 64]), S([1, 8192, 1024]), S([1, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_190: "f32[256, 512, 64]", primals_354: "f32[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        reshape_default: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.reshape.default(bmm_190, _shape_param_0);  bmm_190 = _shape_param_0 = None
        permute_default: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [2, 0, 1, 4, 3]);  reshape_default = None
        reshape_default_1: "f32[512, 16, 16, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, 4);  reshape_default_1 = None
        permute_default_1: "f32[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_default, [0, 1, 4, 3, 2]);  unsqueeze_default = None
        unsqueeze_default_1: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_354, 3);  primals_354 = None
        unsqueeze_default_2: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 4);  unsqueeze_default_1 = None
        permute_default_2: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_default_2, [3, 4, 0, 2, 1]);  unsqueeze_default_2 = None
        permute_default_3: "f32[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 4, 2]);  permute_default_1 = None
        clone_default: "f32[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_2: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        permute_default_4: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_default_2, [3, 4, 2, 0, 1]);  permute_default_2 = None
        clone_default_1: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_3: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        squeeze_dim: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_2, 0);  reshape_default_2 = None
        squeeze_dim_1: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_3, 0);  reshape_default_3 = None
        return (squeeze_dim, squeeze_dim_1)


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

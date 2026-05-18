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
    def forward(self, arg353_1: "f32[1024, 16, 64]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg353_1, 3);  arg353_1 = None
        unsqueeze_default_1: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        permute_default: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_default_1, [3, 4, 0, 2, 1]);  unsqueeze_default_1 = None
        permute_default_1: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_default, [3, 4, 2, 0, 1]);  permute_default = None
        clone_default: "f32[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        squeeze_dim: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default, 0);  reshape_default = None
        return squeeze_dim


def _default_make_inputs():
    return [
    torch.randn([1024, 16, 64], dtype=torch.float32, device='cuda'),
    [1, 1024, 1024],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

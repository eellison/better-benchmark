"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_inference
Pattern hash: c5de6395bc06
Shape hash: 9931de09
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_50: "f32[8, 16, 256, 64]", _shape_param_0, _shape_param_1, arg187_1: "f32[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_default: "f32[8, 256, 16, 64]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        clone_default: "f32[8, 256, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default: "f32[8, 256, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_1: "f32[2048, 1024]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        return (reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn(2097152, dtype=torch.float32, device='cuda').as_strided([8, 16, 256, 64], [262144, 64, 1024, 1]),  # getitem_50
    [8, 256, 1024],  # _shape_param_0
    [2048, 1024],  # _shape_param_1
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

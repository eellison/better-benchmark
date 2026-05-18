"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_training
Pattern hash: e5c544a31374
Shape hash: be64d758
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
    def forward(self, mm_4: "f32[2048, 1024]", _shape_param_0, _shape_param_1, primals_8: "f32[8, 1, 256, 256]", _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[8, 256, 1024]" = torch.ops.aten.reshape.default(mm_4, _shape_param_0);  mm_4 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        reshape_default_1: "f32[8, 256, 16, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_default: "f32[8, 16, 256, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        clone_default: "f32[8, 16, 256, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # No stacktrace found for following nodes
        expand_default: "f32[8, 16, 256, 256]" = torch.ops.aten.expand.default(primals_8, _shape_param_2);  primals_8 = _shape_param_2 = None
        return (clone_default, expand_default)


def _default_make_inputs():
    return [
    torch.randn([2048, 1024], dtype=torch.float32, device='cuda'),
    [8, 256, 1024],  # _shape_param_0
    [8, 256, 16, 64],  # _shape_param_1
    torch.randn([8, 1, 256, 256], dtype=torch.float32, device='cuda'),
    [8, 16, 256, 256],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

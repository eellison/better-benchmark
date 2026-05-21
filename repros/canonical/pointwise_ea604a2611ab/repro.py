"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train
Pattern hash: ea604a2611ab
Shape hash: 6f712f06
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([96, 256, 64], f32), S([24, 4, 256, 1, 64]), S([24, 4, 256, 64]), S([2, 12, 1024, 64]), S([1024, 2, 768]), S([2048, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_23: "f32[96, 256, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        reshape_default: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_23, _shape_param_0);  bmm_23 = _shape_param_0 = None
        permute_default: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 2, 4, 3]);  reshape_default = None
        reshape_default_1: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        reshape_default_2: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_default_2: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_default_1, [1, 0, 2, 3]);  permute_default_1 = None
        clone_default: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_3: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_default_3: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0, 2]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_default_1: "f32[2, 1024, 768]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_4: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        return reshape_default_4



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

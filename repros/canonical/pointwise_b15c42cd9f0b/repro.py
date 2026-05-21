"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train
Pattern hash: b15c42cd9f0b
Shape hash: d621b103
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
_shapes_config = "(T([72, 64, 512], f32), T([1572864], f32), T([2359296], i64, gen=Index(1572864)), S([24, 3, 64, 512, 1]), S([2359296]), S([24, 1024, 64]), S([2, 12, 1024, 64]), S([1024, 2, 768]), S([2048, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_70: "f32[72, 64, 512]", full_default_121: "f32[1572864]", view_1433: "i64[2359296]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        reshape_default: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_70, _shape_param_0);  bmm_70 = _shape_param_0 = None
        permute_default: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 4, 3, 2]);  reshape_default = None
        permute_default_1: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_default, [0, 1, 3, 4, 2]);  permute_default = None
        squeeze_dim: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_default_1, 4);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_default: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        reshape_default_1: "f32[2359296]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        index_put_default: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], reshape_default_1, True);  full_default_121 = view_1433 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_default: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_default, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_default = None
        reshape_default_2: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_2);  as_strided_default = _shape_param_2 = None
        reshape_default_3: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        permute_default_3: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_default_2, [1, 0, 2, 3]);  permute_default_2 = None
        reshape_default_4: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_4);  permute_default_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        reshape_default_5: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        return reshape_default_5



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

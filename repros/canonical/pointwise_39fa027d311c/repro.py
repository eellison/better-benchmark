"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_training
Pattern hash: 39fa027d311c
Shape hash: 0b2ed45d
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
    def forward(self, bmm_68: "f32[384, 768, 64]", _shape_param_0, full_default_98: "f32[9437184]", view_1398: "i64[18874368]", _shape_param_1, _shape_param_2, bmm_70: "f32[288, 64, 512]", _shape_param_3, bmm_71: "f32[288, 512, 64]", _shape_param_4, _shape_param_5, full_default_121: "f32[6291456]", view_1433: "i64[9437184]", _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, primals_6: "f32[768, 768]", _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, primals_4: "f32[768, 768]", _shape_param_14, primals_2: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        reshape_default: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_68, _shape_param_0);  bmm_68 = _shape_param_0 = None
        squeeze_dim: "f32[96, 4, 768, 64]" = torch.ops.aten.squeeze.dim(reshape_default, 4);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        reshape_default_1: "f32[18874368]" = torch.ops.aten.reshape.default(squeeze_dim, [-1]);  squeeze_dim = None
        index_put_default: "f32[9437184]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], reshape_default_1, True);  full_default_98 = view_1398 = reshape_default_1 = None
        as_strided_default: "f32[96, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_default, [96, 1536, 64], [98304, 64, 1], 0);  index_put_default = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[96, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_default, [0, 0, -256, -256]);  as_strided_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        reshape_default_2: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_default, _shape_param_1);  constant_pad_nd_default = _shape_param_1 = None
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_default_1: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_default, [1, 0, 2, 3]);  permute_default = None
        clone_default: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        reshape_default_4: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_70, _shape_param_3);  bmm_70 = _shape_param_3 = None
        permute_default_2: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 1, 4, 3, 2]);  reshape_default_4 = None
        reshape_default_5: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_71, _shape_param_4);  bmm_71 = _shape_param_4 = None
        permute_default_3: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_default_2, [0, 1, 3, 4, 2]);  permute_default_2 = None
        squeeze_dim_1: "f32[96, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_default_3, 4);  permute_default_3 = None
        squeeze_dim_2: "f32[96, 3, 512, 64]" = torch.ops.aten.squeeze.dim(reshape_default_5, 4);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_default_1: "f32[96, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_dim_1, memory_format = torch.contiguous_format);  squeeze_dim_1 = None
        reshape_default_6: "f32[9437184]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        index_put_default_1: "f32[6291456]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], reshape_default_6, True);  reshape_default_6 = None
        reshape_default_7: "f32[9437184]" = torch.ops.aten.reshape.default(squeeze_dim_2, [-1]);  squeeze_dim_2 = None
        index_put_default_2: "f32[6291456]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], reshape_default_7, True);  full_default_121 = view_1433 = reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_default_1: "f32[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_default_2, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_put_default_2 = None
        reshape_default_8: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_default_1, _shape_param_6);  as_strided_default_1 = _shape_param_6 = None
        reshape_default_9: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_7);  reshape_default_8 = _shape_param_7 = None
        permute_default_4: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_9, [0, 2, 1, 3]);  reshape_default_9 = None
        permute_default_5: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_default_4, [1, 0, 2, 3]);  permute_default_4 = None
        reshape_default_10: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_8);  permute_default_5 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_tensor: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(reshape_default_10, 8.0);  reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default_11: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_9);  reshape_default_3 = _shape_param_9 = None
        permute_default_6: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_7: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_6, [1, 0]);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_default_2: "f32[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_default_1, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_put_default_1 = None
        reshape_default_12: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_default_2, _shape_param_10);  as_strided_default_2 = _shape_param_10 = None
        reshape_default_13: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(reshape_default_12, _shape_param_11);  reshape_default_12 = _shape_param_11 = None
        permute_default_8: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_13, [0, 2, 1, 3]);  reshape_default_13 = None
        permute_default_9: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_default_8, [1, 0, 2, 3]);  permute_default_8 = None
        reshape_default_14: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(permute_default_9, _shape_param_12);  permute_default_9 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        reshape_default_15: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_14, _shape_param_13);  reshape_default_14 = _shape_param_13 = None
        permute_default_10: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_11: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_10, [1, 0]);  permute_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        reshape_default_16: "f32[8192, 768]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_14);  div_tensor = _shape_param_14 = None
        permute_default_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_default_13: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_12, [1, 0]);  permute_default_12 = None
        return (reshape_default_11, permute_default_7, reshape_default_15, permute_default_11, reshape_default_16, permute_default_13)


def _default_make_inputs():
    return [
    torch.randn([384, 768, 64], dtype=torch.float32, device='cuda'),
    [96, 4, 768, 64, 1],  # _shape_param_0
    torch.randn([9437184], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [18874368], dtype=torch.int64, device='cuda'),
    [8, 12, 1024, 64],  # _shape_param_1
    [1024, 8, 768],  # _shape_param_2
    torch.randn([288, 64, 512], dtype=torch.float32, device='cuda'),
    [96, 3, 64, 512, 1],  # _shape_param_3
    torch.randn([288, 512, 64], dtype=torch.float32, device='cuda'),
    [96, 3, 512, 64, 1],  # _shape_param_4
    [9437184],  # _shape_param_5
    torch.randn([6291456], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [9437184], dtype=torch.int64, device='cuda'),
    [96, 1024, 64],  # _shape_param_6
    [8, 12, 1024, 64],  # _shape_param_7
    [1024, 8, 768],  # _shape_param_8
    [8192, 768],  # _shape_param_9
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [96, 1024, 64],  # _shape_param_10
    [8, 12, 1024, 64],  # _shape_param_11
    [1024, 8, 768],  # _shape_param_12
    [8192, 768],  # _shape_param_13
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [8192, 768],  # _shape_param_14
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

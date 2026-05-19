"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 46ac3170d265
Shape hash: cca37664
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_24: "f32[384, 768, 64]", bmm_26: "f32[288, 64, 512]", bmm_27: "f32[288, 512, 64]", primals_184: "f32[768, 768]", primals_182: "f32[768, 768]", primals_180: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        reshape_default: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_24, _shape_param_0);  bmm_24 = _shape_param_0 = None
        squeeze_dim: "f32[96, 4, 768, 64]" = torch.ops.aten.squeeze.dim(reshape_default, 4);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        full_default: "f32[9437184]" = torch.ops.aten.full.default([9437184], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default: "i64[9437184]" = torch.ops.prims.iota.default(9437184, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        as_strided_default: "i64[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(iota_default, [96, 4, 768, 64], [98304, 16384, 64, 1], 0);  iota_default = None
        reshape_default_1: "f32[18874368]" = torch.ops.aten.reshape.default(squeeze_dim, [-1]);  squeeze_dim = None
        clone_default: "i64[96, 4, 768, 64]" = torch.ops.aten.clone.default(as_strided_default, memory_format = torch.contiguous_format);  as_strided_default = None
        reshape_default_2: "i64[18874368]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        index_put_default: "f32[9437184]" = torch.ops.aten.index_put.default(full_default, [reshape_default_2], reshape_default_1, True);  full_default = reshape_default_2 = reshape_default_1 = None
        as_strided_default_1: "f32[96, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_default, [96, 1536, 64], [98304, 64, 1], 0);  index_put_default = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[96, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_default_1, [0, 0, -256, -256]);  as_strided_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        reshape_default_3: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_default, _shape_param_2);  constant_pad_nd_default = _shape_param_2 = None
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_default_1: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_default, [1, 0, 2, 3]);  permute_default = None
        clone_default_1: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        reshape_default_5: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_26, _shape_param_4);  bmm_26 = _shape_param_4 = None
        permute_default_2: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 1, 4, 3, 2]);  reshape_default_5 = None
        reshape_default_6: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_27, _shape_param_5);  bmm_27 = _shape_param_5 = None
        permute_default_3: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_default_2, [0, 1, 3, 4, 2]);  permute_default_2 = None
        squeeze_dim_1: "f32[96, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_default_3, 4);  permute_default_3 = None
        squeeze_dim_2: "f32[96, 3, 512, 64]" = torch.ops.aten.squeeze.dim(reshape_default_6, 4);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        full_default_1: "f32[6291456]" = torch.ops.aten.full.default([6291456], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default_1: "i64[6291456]" = torch.ops.prims.iota.default(6291456, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        as_strided_default_2: "i64[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(iota_default_1, [96, 3, 512, 64], [64, 1572864, 6144, 1], 0);  iota_default_1 = None
        clone_default_2: "f32[96, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_dim_1, memory_format = torch.contiguous_format);  squeeze_dim_1 = None
        reshape_default_7: "f32[9437184]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_6);  clone_default_2 = _shape_param_6 = None
        clone_default_3: "i64[96, 3, 512, 64]" = torch.ops.aten.clone.default(as_strided_default_2, memory_format = torch.contiguous_format);  as_strided_default_2 = None
        reshape_default_8: "i64[9437184]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_7);  clone_default_3 = _shape_param_7 = None
        index_put_default_1: "f32[6291456]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_8], reshape_default_7, True);  reshape_default_7 = None
        reshape_default_9: "f32[9437184]" = torch.ops.aten.reshape.default(squeeze_dim_2, [-1]);  squeeze_dim_2 = None
        index_put_default_2: "f32[6291456]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_8], reshape_default_9, True);  full_default_1 = reshape_default_8 = reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_default_3: "f32[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_default_2, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_put_default_2 = None
        reshape_default_10: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_default_3, _shape_param_8);  as_strided_default_3 = _shape_param_8 = None
        reshape_default_11: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(reshape_default_10, _shape_param_9);  reshape_default_10 = _shape_param_9 = None
        permute_default_4: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_11, [0, 2, 1, 3]);  reshape_default_11 = None
        permute_default_5: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_default_4, [1, 0, 2, 3]);  permute_default_4 = None
        reshape_default_12: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_10);  permute_default_5 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_tensor: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(reshape_default_12, 8.0);  reshape_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default_13: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_11);  reshape_default_4 = _shape_param_11 = None
        permute_default_6: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_default_7: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_6, [1, 0]);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_default_4: "f32[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_default_1, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_put_default_1 = None
        reshape_default_14: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_default_4, _shape_param_12);  as_strided_default_4 = _shape_param_12 = None
        reshape_default_15: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(reshape_default_14, _shape_param_13);  reshape_default_14 = _shape_param_13 = None
        permute_default_8: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_15, [0, 2, 1, 3]);  reshape_default_15 = None
        permute_default_9: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_default_8, [1, 0, 2, 3]);  permute_default_8 = None
        reshape_default_16: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(permute_default_9, _shape_param_14);  permute_default_9 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        reshape_default_17: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_16, _shape_param_15);  reshape_default_16 = _shape_param_15 = None
        permute_default_10: "f32[768, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        permute_default_11: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_10, [1, 0]);  permute_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        reshape_default_18: "f32[8192, 768]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_16);  div_tensor = _shape_param_16 = None
        permute_default_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
        permute_default_13: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_12, [1, 0]);  permute_default_12 = None
        return (reshape_default_13, permute_default_7, reshape_default_17, permute_default_11, reshape_default_18, permute_default_13)


def _default_make_inputs():
    return [
    torch.randn([384, 768, 64], dtype=torch.float32, device='cuda'),
    torch.randn([288, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([288, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [96, 4, 768, 64, 1],  # _shape_param_0
    [18874368],  # _shape_param_1
    [8, 12, 1024, 64],  # _shape_param_2
    [1024, 8, 768],  # _shape_param_3
    [96, 3, 64, 512, 1],  # _shape_param_4
    [96, 3, 512, 64, 1],  # _shape_param_5
    [9437184],  # _shape_param_6
    [9437184],  # _shape_param_7
    [96, 1024, 64],  # _shape_param_8
    [8, 12, 1024, 64],  # _shape_param_9
    [1024, 8, 768],  # _shape_param_10
    [8192, 768],  # _shape_param_11
    [96, 1024, 64],  # _shape_param_12
    [8, 12, 1024, 64],  # _shape_param_13
    [1024, 8, 768],  # _shape_param_14
    [8192, 768],  # _shape_param_15
    [8192, 768],  # _shape_param_16
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

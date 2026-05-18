"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_training
Pattern hash: 8f57e3556aea
Shape hash: e2da0436
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[8192, 768]", _shape_param_0, primals_7: "f32[768]", bmm: "f32[288, 512, 512]", _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, primals_8: "f32[8, 1024]", _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, primals_9: "b8[8, 1024]", _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None
        add_tensor: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(reshape_default, primals_7);  reshape_default = primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        reshape_default_1: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        permute_default: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 2, 4, 3]);  reshape_default_1 = None
        reshape_default_2: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(reshape_default_2, [0, 0, 0, 1], 0.0);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        reshape_default_3: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_default, _shape_param_3);  constant_pad_nd_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_default: "f32[96, 4, 256, 513]" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_tensor: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(reshape_default_3, 2, 0, 256)
        slice_tensor_1: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 257);  slice_tensor = None
        slice_tensor_2: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_default, 1, 0, -1)
        slice_tensor_3: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 256, 9223372036854775807)
        copy_default: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_3, slice_tensor_1);  slice_tensor_3 = slice_tensor_1 = None
        slice_scatter_default: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_2, copy_default, 3, 256, 9223372036854775807);  slice_tensor_2 = copy_default = None
        slice_scatter_default_1: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_default, 1, 0, -1);  full_default = slice_scatter_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_int: "f32[96, 512, 513]" = torch.ops.aten.select.int(reshape_default_3, 1, -1)
        slice_tensor_4: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_int, 1, 256, 9223372036854775807);  select_int = None
        slice_tensor_5: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 2, 0, 257);  slice_tensor_4 = None
        select_int_1: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        slice_tensor_6: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_1, 2, 256, 9223372036854775807)
        copy_default_1: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_6, slice_tensor_5);  slice_tensor_6 = slice_tensor_5 = None
        slice_scatter_default_2: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_1, copy_default_1, 2, 256, 9223372036854775807);  select_int_1 = copy_default_1 = None
        select_scatter_default: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_1, slice_scatter_default_2, 1, -1);  slice_scatter_default_1 = slice_scatter_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_tensor_7: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(reshape_default_3, 2, -257, -1)
        slice_tensor_8: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_7, 3, 257, 9223372036854775807);  slice_tensor_7 = None
        slice_tensor_9: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_tensor_10: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_9, 3, 0, 256)
        copy_default_2: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_10, slice_tensor_8);  slice_tensor_10 = slice_tensor_8 = None
        slice_scatter_default_3: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_9, copy_default_2, 3, 0, 256);  slice_tensor_9 = copy_default_2 = None
        slice_scatter_default_4: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default, slice_scatter_default_3, 1, 1, 9223372036854775807);  select_scatter_default = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_int_2: "f32[96, 512, 513]" = torch.ops.aten.select.int(reshape_default_3, 1, 0);  reshape_default_3 = None
        slice_tensor_11: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_2, 1, 0, 255);  select_int_2 = None
        slice_tensor_12: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_11, 2, -255, 9223372036854775807);  slice_tensor_11 = None
        select_int_3: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_4, 1, 0)
        slice_tensor_13: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_3, 1, 1, 256)
        slice_tensor_14: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_13, 2, 1, 256)
        copy_default_3: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_14, slice_tensor_12);  slice_tensor_14 = slice_tensor_12 = None
        slice_scatter_default_5: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_13, copy_default_3, 2, 1, 256);  slice_tensor_13 = copy_default_3 = None
        slice_scatter_default_6: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_3, slice_scatter_default_5, 1, 1, 256);  select_int_3 = slice_scatter_default_5 = None
        select_scatter_default_1: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_4, slice_scatter_default_6, 1, 0);  slice_scatter_default_4 = slice_scatter_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        full_default_1: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_default, -2);  iota_default = None
        iota_default_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        sub_tensor: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_default, unsqueeze_default_1);  unsqueeze_default = unsqueeze_default_1 = None
        le_scalar: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_tensor, 0);  sub_tensor = None
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[256, 257]" = torch.ops.aten.where.self(le_scalar, full_default_1, full_default_2);  le_scalar = full_default_1 = None
        rev_default: "f32[256, 257]" = torch.ops.prims.rev.default(where_self, [0]);  where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_default_2: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_default, 0);  rev_default = None
        unsqueeze_default_3: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_default_1: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_default_3, [1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_default: "f32[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_default_3, _shape_param_4);  _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        reshape_default_4: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_1, _shape_param_5);  select_scatter_default_1 = _shape_param_5 = None
        permute_default_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        slice_tensor_15: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_1, 1, 0, 256)
        slice_tensor_16: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_15, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_3: "f32[8, 12, 256, 257]" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_default_2: "f32[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_default_3, [0, 2, 1, 3]);  full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_default: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_default, torch.bool);  expand_default = None
        where_self_1: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_default, permute_default_2, slice_tensor_16);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_default_4: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_16, where_self_1);  slice_tensor_16 = where_self_1 = None
        slice_scatter_default_7: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_15, copy_default_4, 3, 0, 257);  slice_tensor_15 = copy_default_4 = None
        slice_scatter_default_8: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_1, slice_scatter_default_7, 1, 0, 256);  permute_default_1 = slice_scatter_default_7 = None
        permute_default_3: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_8, [0, 2, 1, 3]);  slice_scatter_default_8 = None
        reshape_default_5: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_6);  permute_default_3 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_default_1: "f32[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_default_1, _shape_param_7);  _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        reshape_default_6: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(reshape_default_5, _shape_param_8);  reshape_default_5 = _shape_param_8 = None
        permute_default_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None
        slice_tensor_17: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_4, 1, -256, 9223372036854775807)
        slice_tensor_18: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_17, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_default_1: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_default_1, torch.bool);  expand_default_1 = None
        where_self_2: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_default_1, permute_default_2, slice_tensor_18);  convert_element_type_default_1 = permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_default_5: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_18, where_self_2);  slice_tensor_18 = where_self_2 = None
        slice_scatter_default_9: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_17, copy_default_5, 3, -257, 9223372036854775807);  slice_tensor_17 = copy_default_5 = None
        slice_scatter_default_10: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_4, slice_scatter_default_9, 1, -256, 9223372036854775807);  permute_default_4 = slice_scatter_default_9 = None
        permute_default_5: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_10, [0, 2, 1, 3]);  slice_scatter_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_scalar: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(primals_8, 0);  primals_8 = None
        unsqueeze_default_4: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_scalar, 2);  ne_scalar = None
        unsqueeze_default_5: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_default_2: "f32[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_5, torch.float32)
        full_default_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_3: "f32[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_default_5, full_default_4, convert_element_type_default_2);  unsqueeze_default_5 = full_default_4 = convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_default_5: "f32[8, 1024, 1, 1]" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_6: "f32[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_default_5, [0, 2, 1, 3]);  full_default_5 = None
        reshape_default_7: "f32[8, 1024, 1]" = torch.ops.aten.reshape.default(permute_default_6, _shape_param_9);  permute_default_6 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_7: "f32[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_self_3, [0, 2, 1, 3]);  where_self_3 = None
        reshape_default_8: "f32[8, 1024, 1]" = torch.ops.aten.reshape.default(permute_default_7, _shape_param_10);  permute_default_7 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        reshape_default_9: "f32[8, 2, 512, 1]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_11);  reshape_default_7 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_default: "f32[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(reshape_default_9, [8, 3, 512, 1], [1024, 256, 1, 1]);  reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        reshape_default_10: "f32[8, 2, 512, 1]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_12);  reshape_default_8 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_default_1: "f32[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(reshape_default_10, [8, 3, 512, 1], [1024, 256, 1, 1]);  reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_default_6: "f32[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        permute_default_8: "f32[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_default_6, [0, 1, 2, 4, 3]);  unsqueeze_default_6 = None
        unsqueeze_default_7: "f32[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default_1, 4);  as_strided_default_1 = None
        permute_default_9: "f32[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_default_7, [0, 1, 4, 2, 3]);  unsqueeze_default_7 = None
        mul_tensor: "f32[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_default_8, permute_default_9);  permute_default_8 = permute_default_9 = None
        reshape_default_11: "f32[8, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_13);  mul_tensor = _shape_param_13 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_1: "f32[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(reshape_default_11, [0, 0, 0, 1], 0.0);  reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        reshape_default_12: "f32[8, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_default_1, _shape_param_14);  constant_pad_nd_default_1 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_default_6: "f32[8, 4, 256, 513]" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_tensor_19: "f32[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(reshape_default_12, 2, 0, 256)
        slice_tensor_20: "f32[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_19, 3, 0, 257);  slice_tensor_19 = None
        slice_tensor_21: "f32[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_default_6, 1, 0, -1)
        slice_tensor_22: "f32[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_21, 3, 256, 9223372036854775807)
        copy_default_6: "f32[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_22, slice_tensor_20);  slice_tensor_22 = slice_tensor_20 = None
        slice_scatter_default_11: "f32[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_21, copy_default_6, 3, 256, 9223372036854775807);  slice_tensor_21 = copy_default_6 = None
        slice_scatter_default_12: "f32[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_6, slice_scatter_default_11, 1, 0, -1);  full_default_6 = slice_scatter_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_int_4: "f32[8, 512, 513]" = torch.ops.aten.select.int(reshape_default_12, 1, -1)
        slice_tensor_23: "f32[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_int_4, 1, 256, 9223372036854775807);  select_int_4 = None
        slice_tensor_24: "f32[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_23, 2, 0, 257);  slice_tensor_23 = None
        select_int_5: "f32[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_12, 1, -1)
        slice_tensor_25: "f32[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_5, 2, 256, 9223372036854775807)
        copy_default_7: "f32[8, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_25, slice_tensor_24);  slice_tensor_25 = slice_tensor_24 = None
        slice_scatter_default_13: "f32[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_5, copy_default_7, 2, 256, 9223372036854775807);  select_int_5 = copy_default_7 = None
        select_scatter_default_2: "f32[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_12, slice_scatter_default_13, 1, -1);  slice_scatter_default_12 = slice_scatter_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_tensor_26: "f32[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(reshape_default_12, 2, -257, -1)
        slice_tensor_27: "f32[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_26, 3, 257, 9223372036854775807);  slice_tensor_26 = None
        slice_tensor_28: "f32[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 1, 9223372036854775807)
        slice_tensor_29: "f32[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_28, 3, 0, 256)
        copy_default_8: "f32[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_29, slice_tensor_27);  slice_tensor_29 = slice_tensor_27 = None
        slice_scatter_default_14: "f32[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_28, copy_default_8, 3, 0, 256);  slice_tensor_28 = copy_default_8 = None
        slice_scatter_default_15: "f32[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_2, slice_scatter_default_14, 1, 1, 9223372036854775807);  select_scatter_default_2 = slice_scatter_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_int_6: "f32[8, 512, 513]" = torch.ops.aten.select.int(reshape_default_12, 1, 0);  reshape_default_12 = None
        slice_tensor_30: "f32[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_6, 1, 0, 255);  select_int_6 = None
        slice_tensor_31: "f32[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_30, 2, -255, 9223372036854775807);  slice_tensor_30 = None
        select_int_7: "f32[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, 0)
        slice_tensor_32: "f32[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_7, 1, 1, 256)
        slice_tensor_33: "f32[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_32, 2, 1, 256)
        copy_default_9: "f32[8, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_33, slice_tensor_31);  slice_tensor_33 = slice_tensor_31 = None
        slice_scatter_default_16: "f32[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_32, copy_default_9, 2, 1, 256);  slice_tensor_32 = copy_default_9 = None
        slice_scatter_default_17: "f32[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_7, slice_scatter_default_16, 1, 1, 256);  select_int_7 = slice_scatter_default_16 = None
        select_scatter_default_3: "f32[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_15, slice_scatter_default_17, 1, 0);  slice_scatter_default_15 = slice_scatter_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_default_2: "f32[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_default_3, _shape_param_15);  unsqueeze_default_3 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        reshape_default_13: "f32[8, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_3, _shape_param_16);  select_scatter_default_3 = _shape_param_16 = None
        permute_default_10: "f32[8, 1024, 1, 513]" = torch.ops.aten.permute.default(reshape_default_13, [0, 2, 1, 3]);  reshape_default_13 = None
        slice_tensor_34: "f32[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_10, 1, 0, 256)
        slice_tensor_35: "f32[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_34, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_7: "f32[8, 256, 1, 257]" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_default_3: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_default_2, torch.bool);  expand_default_2 = None
        where_self_4: "f32[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_default_3, full_default_7, slice_tensor_35);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_default_10: "f32[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_tensor_35, where_self_4);  slice_tensor_35 = where_self_4 = None
        slice_scatter_default_18: "f32[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_34, copy_default_10, 3, 0, 257);  slice_tensor_34 = copy_default_10 = None
        slice_scatter_default_19: "f32[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_default_10, slice_scatter_default_18, 1, 0, 256);  permute_default_10 = slice_scatter_default_18 = None
        permute_default_11: "f32[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_19, [0, 2, 1, 3]);  slice_scatter_default_19 = None
        reshape_default_14: "f32[8, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_default_11, _shape_param_17);  permute_default_11 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_default_3: "f32[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_default_1, _shape_param_18);  rev_default_1 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        reshape_default_15: "f32[8, 1, 1024, 513]" = torch.ops.aten.reshape.default(reshape_default_14, _shape_param_19);  reshape_default_14 = _shape_param_19 = None
        permute_default_12: "f32[8, 1024, 1, 513]" = torch.ops.aten.permute.default(reshape_default_15, [0, 2, 1, 3]);  reshape_default_15 = None
        slice_tensor_36: "f32[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_12, 1, -256, 9223372036854775807)
        slice_tensor_37: "f32[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_36, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_default_4: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_default_3, torch.bool);  expand_default_3 = None
        where_self_5: "f32[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_default_4, full_default_7, slice_tensor_37);  convert_element_type_default_4 = full_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_default_11: "f32[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_tensor_37, where_self_5);  slice_tensor_37 = where_self_5 = None
        slice_scatter_default_20: "f32[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_36, copy_default_11, 3, -257, 9223372036854775807);  slice_tensor_36 = copy_default_11 = None
        slice_scatter_default_21: "f32[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_default_12, slice_scatter_default_20, 1, -256, 9223372036854775807);  permute_default_12 = slice_scatter_default_20 = None
        permute_default_13: "f32[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_21, [0, 2, 1, 3]);  slice_scatter_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_default_14: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_default_5, [0, 2, 1, 3]);  permute_default_5 = None
        permute_default_15: "f32[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_default_13, [0, 2, 1, 3]);  permute_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_tensor_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_default_14, permute_default_15);  permute_default_14 = permute_default_15 = None
        permute_default_16: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1, 3]);  add_tensor_1 = None
        permute_default_17: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_default_16, [0, 2, 1, 3]);  permute_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_default: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_default_17, memory_format = torch.contiguous_format);  permute_default_17 = None
        amax_default: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_default, [-1], True)
        sub_tensor_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_default, amax_default);  clone_default = amax_default = None
        exp_default: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_default_8: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(primals_9, 2);  primals_9 = None
        unsqueeze_default_9: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        where_self_6: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_default_9, full_default_2, div_tensor);  unsqueeze_default_9 = full_default_2 = div_tensor = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[36]" = torch.ops.prims.inductor_seeds.default(36, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_scalar, where_self_6);  gt_scalar = where_self_6 = None
        mul_tensor_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        reshape_default_16: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_20);  add_tensor = _shape_param_20 = None
        permute_default_18: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_16, [1, 0, 2, 3]);  reshape_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_default_19: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_tensor_2, [0, 2, 1, 3]);  mul_tensor_2 = None
        clone_default_1: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_19, memory_format = torch.contiguous_format);  permute_default_19 = None
        reshape_default_17: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_21);  clone_default_1 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_20: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default_18, [0, 2, 1, 3]);  permute_default_18 = None
        reshape_default_18: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_default_20, _shape_param_22);  permute_default_20 = _shape_param_22 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_2: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(reshape_default_18, [0, 0, 256, 256], -1.0);  reshape_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_default_2: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_default_2, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_default_2 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_3: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(reshape_default_17, [0, 257], 0.0);  reshape_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        reshape_default_19: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_default_3, _shape_param_23);  constant_pad_nd_default_3 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_tensor_38: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(reshape_default_19, 2, 0, -256);  reshape_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        reshape_default_20: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_tensor_38, _shape_param_24);  slice_tensor_38 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_tensor_39: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(reshape_default_20, 3, 0, -1);  reshape_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_default_10: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_39, 4);  slice_tensor_39 = None
        unsqueeze_default_11: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default_2, 4);  as_strided_default_2 = None
        reshape_default_21: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_default_10, _shape_param_25);  unsqueeze_default_10 = _shape_param_25 = None
        clone_default_2: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default_11, memory_format = torch.contiguous_format);  unsqueeze_default_11 = None
        reshape_default_22: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_26);  clone_default_2 = _shape_param_26 = None
        return (reshape_default_21, reshape_default_22)


def _default_make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [1024, 8, 768],  # _shape_param_0
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([288, 512, 512], dtype=torch.float32, device='cuda'),
    [96, 3, 512, 1, 512],  # _shape_param_1
    [96, 3, 512, 512],  # _shape_param_2
    [96, 3, 512, 513],  # _shape_param_3
    [8, 256, 12, 257],  # _shape_param_4
    [8, 12, 1024, 513],  # _shape_param_5
    [96, 4, 256, 513],  # _shape_param_6
    [8, 256, 12, 257],  # _shape_param_7
    [8, 12, 1024, 513],  # _shape_param_8
    torch.randn([8, 1024], dtype=torch.float32, device='cuda'),
    [8, 1024, 1],  # _shape_param_9
    [8, 1024, 1],  # _shape_param_10
    [8, 2, 512, 1],  # _shape_param_11
    [8, 2, 512, 1],  # _shape_param_12
    [8, 3, 512, 512],  # _shape_param_13
    [8, 3, 512, 513],  # _shape_param_14
    [8, 256, 1, 257],  # _shape_param_15
    [8, 1, 1024, 513],  # _shape_param_16
    [8, 4, 256, 513],  # _shape_param_17
    [8, 256, 1, 257],  # _shape_param_18
    [8, 1, 1024, 513],  # _shape_param_19
    torch.randint(0, 2, [8, 1024], dtype=torch.bool, device='cuda'),
    [1024, 8, 12, 64],  # _shape_param_20
    [96, 4, 256, 513],  # _shape_param_21
    [96, 1024, 64],  # _shape_param_22
    [96, 4, -1],  # _shape_param_23
    [96, 4, 256, 769],  # _shape_param_24
    [384, 256, 768],  # _shape_param_25
    [384, 768, 64],  # _shape_param_26
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer
Pattern hash: a5cc2805dc48
Shape hash: f974ed9a
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
_shapes_config = "(T([1, 4096], b8), T([180, 512, 512], f16), T([1, 4096], f16), T([4096, 768], f16), S([12, 15, 512, 1, 512]), S([12, 15, 512, 512]), S([12, 15, 512, 513]), S([1, 12, 4096, 513]), S([1, 12, 4096, 513]), S([1, 256, 12, 257]), S([1, 12, 4096, 513]), S([12, 16, 256, 513]), S([1, 12, 4096, 513]), S([1, 12, 4096, 513]), S([1, 256, 12, 257]), S([1, 12, 4096, 513]), S([1, 4096, 1]), S([1, 8, 512, 1]), S([1, 4096, 1]), S([1, 8, 512, 1]), S([1, 15, 512, 512]), S([1, 15, 512, 513]), S([1, 1, 4096, 513]), S([1, 1, 4096, 513]), S([1, 256, 1, 257]), S([1, 1, 4096, 513]), S([1, 16, 256, 513]), S([1, 1, 4096, 513]), S([1, 1, 4096, 513]), S([1, 256, 1, 257]), S([1, 1, 4096, 513]), S([12, 16, 256, 513]), S([12, 16, -1]), S([12, 16, 256, 769]), S([192, 256, 768]), S([4096, 1, 768]), S([4096, 1, 12, 64]), S([12, 4096, 64]), S([192, 768, 64]))"

class Repro(torch.nn.Module):
    def forward(self, arg8_1: "b8[1, 4096]", bmm_22: "f16[180, 512, 512]", arg7_1: "f16[1, 4096]", addmm_68: "f16[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_default: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2);  arg8_1 = None
        unsqueeze_default_1: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 3);  unsqueeze_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_default_1: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_tensor: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_default_1, 1, 0, -1)
        slice_tensor_1: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_default_1, 1, 0, -1)
        slice_tensor_2: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_1, 3, 256, 9223372036854775807);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        reshape_default: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_22, _shape_param_0);  bmm_22 = _shape_param_0 = None
        permute_default: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 2, 4, 3]);  reshape_default = None
        reshape_default_1: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(reshape_default_1, [0, 0, 0, 1], 0.0);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        reshape_default_2: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_default, _shape_param_2);  constant_pad_nd_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_tensor_3: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(reshape_default_2, 2, 0, 256)
        slice_tensor_4: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_3, 3, 0, 257);  slice_tensor_3 = None
        copy_default: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_2, slice_tensor_4);  slice_tensor_2 = slice_tensor_4 = None
        slice_scatter_default: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor, copy_default, 3, 256, 9223372036854775807);  slice_tensor = copy_default = None
        slice_scatter_default_1: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_scatter_default, 1, 0, -1);  full_default_1 = slice_scatter_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_int: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        select_int_1: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        slice_tensor_5: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_1, 2, 256, 9223372036854775807);  select_int_1 = None
        select_int_2: "f16[12, 512, 513]" = torch.ops.aten.select.int(reshape_default_2, 1, -1)
        slice_tensor_6: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_int_2, 1, 256, 9223372036854775807);  select_int_2 = None
        slice_tensor_7: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_6, 2, 0, 257);  slice_tensor_6 = None
        copy_default_1: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_5, slice_tensor_7);  slice_tensor_5 = slice_tensor_7 = None
        slice_scatter_default_2: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int, copy_default_1, 2, 256, 9223372036854775807);  select_int = copy_default_1 = None
        select_scatter_default: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_1, slice_scatter_default_2, 1, -1);  slice_scatter_default_1 = slice_scatter_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_tensor_8: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_tensor_9: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_tensor_10: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_9, 3, 0, 256);  slice_tensor_9 = None
        slice_tensor_11: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(reshape_default_2, 2, -257, -1)
        slice_tensor_12: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_11, 3, 257, 9223372036854775807);  slice_tensor_11 = None
        copy_default_2: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_10, slice_tensor_12);  slice_tensor_10 = slice_tensor_12 = None
        slice_scatter_default_3: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_8, copy_default_2, 3, 0, 256);  slice_tensor_8 = copy_default_2 = None
        slice_scatter_default_4: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default, slice_scatter_default_3, 1, 1, 9223372036854775807);  select_scatter_default = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_int_3: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_4, 1, 0)
        slice_tensor_13: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_3, 1, 1, 256)
        select_int_4: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_4, 1, 0)
        slice_tensor_14: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_4, 1, 1, 256);  select_int_4 = None
        slice_tensor_15: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_14, 2, 1, 256);  slice_tensor_14 = None
        select_int_5: "f16[12, 512, 513]" = torch.ops.aten.select.int(reshape_default_2, 1, 0);  reshape_default_2 = None
        slice_tensor_16: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_5, 1, 0, 255);  select_int_5 = None
        slice_tensor_17: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_16, 2, -255, 9223372036854775807);  slice_tensor_16 = None
        copy_default_3: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_15, slice_tensor_17);  slice_tensor_15 = slice_tensor_17 = None
        slice_scatter_default_5: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_13, copy_default_3, 2, 1, 256);  slice_tensor_13 = copy_default_3 = None
        slice_scatter_default_6: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_3, slice_scatter_default_5, 1, 1, 256);  select_int_3 = slice_scatter_default_5 = None
        select_scatter_default_1: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_4, slice_scatter_default_6, 1, 0);  slice_scatter_default_4 = slice_scatter_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        reshape_default_3: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_default_1, _shape_param_3);  _shape_param_3 = None
        permute_default_1: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        slice_tensor_18: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_1, 1, 0, 256)
        reshape_default_4: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_default_1, _shape_param_4);  _shape_param_4 = None
        permute_default_2: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        slice_tensor_19: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_2, 1, 0, 256);  permute_default_2 = None
        slice_tensor_20: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_19, 3, 0, 257);  slice_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_default: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_2: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_default, -2);  iota_default = None
        iota_default_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_3: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        sub_tensor: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_default_2, unsqueeze_default_3);  unsqueeze_default_2 = unsqueeze_default_3 = None
        le_scalar: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_tensor, 0);  sub_tensor = None
        full_default_2: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[256, 257]" = torch.ops.aten.where.self(le_scalar, full_default_2, full_default_3);  le_scalar = full_default_2 = full_default_3 = None
        rev_default: "f16[256, 257]" = torch.ops.prims.rev.default(where_self, [0]);  where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_default_4: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_default, 0);  rev_default = None
        unsqueeze_default_5: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_default: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_default_5, _shape_param_5);  _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_default: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_default, torch.bool);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_4: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_default_3: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_default_4, [0, 2, 1, 3]);  full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        reshape_default_5: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_default_1, _shape_param_6);  select_scatter_default_1 = _shape_param_6 = None
        permute_default_4: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None
        slice_tensor_21: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_4, 1, 0, 256);  permute_default_4 = None
        slice_tensor_22: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_21, 3, 0, 257);  slice_tensor_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_self_1: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_default, permute_default_3, slice_tensor_22);  convert_element_type_default = permute_default_3 = slice_tensor_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_default_4: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_20, where_self_1);  slice_tensor_20 = where_self_1 = None
        slice_scatter_default_7: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_18, copy_default_4, 3, 0, 257);  slice_tensor_18 = copy_default_4 = None
        slice_scatter_default_8: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_1, slice_scatter_default_7, 1, 0, 256);  permute_default_1 = slice_scatter_default_7 = None
        permute_default_5: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_default_8, [0, 2, 1, 3]);  slice_scatter_default_8 = None
        reshape_default_6: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_7);  permute_default_5 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        reshape_default_7: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_8);  _shape_param_8 = None
        permute_default_6: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(reshape_default_7, [0, 2, 1, 3]);  reshape_default_7 = None
        slice_tensor_23: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_6, 1, -256, 9223372036854775807)
        reshape_default_8: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_9);  _shape_param_9 = None
        permute_default_7: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(reshape_default_8, [0, 2, 1, 3]);  reshape_default_8 = None
        slice_tensor_24: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_7, 1, -256, 9223372036854775807);  permute_default_7 = None
        slice_tensor_25: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_24, 3, -257, 9223372036854775807);  slice_tensor_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_default_1: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_default_5, [1, 3]);  unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_default_1: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_default_1, _shape_param_10);  rev_default_1 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_default_1: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_default_1, torch.bool);  expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_5: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_default_8: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_default_5, [0, 2, 1, 3]);  full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        reshape_default_9: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_11);  reshape_default_6 = _shape_param_11 = None
        permute_default_9: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(reshape_default_9, [0, 2, 1, 3]);  reshape_default_9 = None
        slice_tensor_26: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_9, 1, -256, 9223372036854775807);  permute_default_9 = None
        slice_tensor_27: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_26, 3, -257, 9223372036854775807);  slice_tensor_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_self_2: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_default_1, permute_default_8, slice_tensor_27);  convert_element_type_default_1 = permute_default_8 = slice_tensor_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_default_5: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_25, where_self_2);  slice_tensor_25 = where_self_2 = None
        slice_scatter_default_9: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_23, copy_default_5, 3, -257, 9223372036854775807);  slice_tensor_23 = copy_default_5 = None
        slice_scatter_default_10: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_6, slice_scatter_default_9, 1, -256, 9223372036854775807);  permute_default_6 = slice_scatter_default_9 = None
        permute_default_10: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_default_10, [0, 2, 1, 3]);  slice_scatter_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_default_11: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_default_10, [0, 2, 1, 3]);  permute_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_default_6: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_tensor_28: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_default_6, 1, 0, -1)
        slice_tensor_29: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_default_6, 1, 0, -1)
        slice_tensor_30: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_29, 3, 256, 9223372036854775807);  slice_tensor_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_default_7: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_12: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_default_7, [0, 2, 1, 3]);  full_default_7 = None
        reshape_default_10: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_default_12, _shape_param_12);  permute_default_12 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        reshape_default_11: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(reshape_default_10, _shape_param_13);  reshape_default_10 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_default: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(reshape_default_11, [1, 15, 512, 1], [4096, 256, 1, 1]);  reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_default_6: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        permute_default_13: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_default_6, [0, 1, 2, 4, 3]);  unsqueeze_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_8: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_scalar: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0);  arg7_1 = None
        unsqueeze_default_7: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_scalar, 2);  ne_scalar = None
        unsqueeze_default_8: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_default_2: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_8, torch.float16)
        where_self_3: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_default_8, full_default_8, convert_element_type_default_2);  unsqueeze_default_8 = full_default_8 = convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_14: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_self_3, [0, 2, 1, 3]);  where_self_3 = None
        reshape_default_12: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_default_14, _shape_param_14);  permute_default_14 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        reshape_default_13: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(reshape_default_12, _shape_param_15);  reshape_default_12 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_default_1: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(reshape_default_13, [1, 15, 512, 1], [4096, 256, 1, 1]);  reshape_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_default_9: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default_1, 4);  as_strided_default_1 = None
        permute_default_15: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_default_9, [0, 1, 4, 2, 3]);  unsqueeze_default_9 = None
        mul_tensor: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_default_13, permute_default_15);  permute_default_13 = permute_default_15 = None
        reshape_default_14: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_16);  mul_tensor = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_1: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(reshape_default_14, [0, 0, 0, 1], 0.0);  reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        reshape_default_15: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_default_1, _shape_param_17);  constant_pad_nd_default_1 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_tensor_31: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(reshape_default_15, 2, 0, 256)
        slice_tensor_32: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_31, 3, 0, 257);  slice_tensor_31 = None
        copy_default_6: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_30, slice_tensor_32);  slice_tensor_30 = slice_tensor_32 = None
        slice_scatter_default_11: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_28, copy_default_6, 3, 256, 9223372036854775807);  slice_tensor_28 = copy_default_6 = None
        slice_scatter_default_12: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_6, slice_scatter_default_11, 1, 0, -1);  full_default_6 = slice_scatter_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_int_6: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_12, 1, -1)
        select_int_7: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_12, 1, -1)
        slice_tensor_33: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_7, 2, 256, 9223372036854775807);  select_int_7 = None
        select_int_8: "f16[1, 512, 513]" = torch.ops.aten.select.int(reshape_default_15, 1, -1)
        slice_tensor_34: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_int_8, 1, 256, 9223372036854775807);  select_int_8 = None
        slice_tensor_35: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_34, 2, 0, 257);  slice_tensor_34 = None
        copy_default_7: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_33, slice_tensor_35);  slice_tensor_33 = slice_tensor_35 = None
        slice_scatter_default_13: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_6, copy_default_7, 2, 256, 9223372036854775807);  select_int_6 = copy_default_7 = None
        select_scatter_default_2: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_12, slice_scatter_default_13, 1, -1);  slice_scatter_default_12 = slice_scatter_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_tensor_36: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 1, 9223372036854775807)
        slice_tensor_37: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 1, 9223372036854775807)
        slice_tensor_38: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_37, 3, 0, 256);  slice_tensor_37 = None
        slice_tensor_39: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(reshape_default_15, 2, -257, -1)
        slice_tensor_40: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_39, 3, 257, 9223372036854775807);  slice_tensor_39 = None
        copy_default_8: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_38, slice_tensor_40);  slice_tensor_38 = slice_tensor_40 = None
        slice_scatter_default_14: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_36, copy_default_8, 3, 0, 256);  slice_tensor_36 = copy_default_8 = None
        slice_scatter_default_15: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_2, slice_scatter_default_14, 1, 1, 9223372036854775807);  select_scatter_default_2 = slice_scatter_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_int_9: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, 0)
        slice_tensor_41: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_9, 1, 1, 256)
        select_int_10: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, 0)
        slice_tensor_42: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_10, 1, 1, 256);  select_int_10 = None
        slice_tensor_43: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_42, 2, 1, 256);  slice_tensor_42 = None
        select_int_11: "f16[1, 512, 513]" = torch.ops.aten.select.int(reshape_default_15, 1, 0);  reshape_default_15 = None
        slice_tensor_44: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_11, 1, 0, 255);  select_int_11 = None
        slice_tensor_45: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_44, 2, -255, 9223372036854775807);  slice_tensor_44 = None
        copy_default_9: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_43, slice_tensor_45);  slice_tensor_43 = slice_tensor_45 = None
        slice_scatter_default_16: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_41, copy_default_9, 2, 1, 256);  slice_tensor_41 = copy_default_9 = None
        slice_scatter_default_17: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_9, slice_scatter_default_16, 1, 1, 256);  select_int_9 = slice_scatter_default_16 = None
        select_scatter_default_3: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_15, slice_scatter_default_17, 1, 0);  slice_scatter_default_15 = slice_scatter_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        reshape_default_16: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_default_3, _shape_param_18);  _shape_param_18 = None
        permute_default_16: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(reshape_default_16, [0, 2, 1, 3]);  reshape_default_16 = None
        slice_tensor_46: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_16, 1, 0, 256)
        reshape_default_17: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_default_3, _shape_param_19);  _shape_param_19 = None
        permute_default_17: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(reshape_default_17, [0, 2, 1, 3]);  reshape_default_17 = None
        slice_tensor_47: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_17, 1, 0, 256);  permute_default_17 = None
        slice_tensor_48: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_47, 3, 0, 257);  slice_tensor_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_default_2: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_10: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_default_2, -2);  iota_default_2 = None
        iota_default_3: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_11: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_default_3, -1);  iota_default_3 = None
        sub_tensor_1: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_default_10, unsqueeze_default_11);  unsqueeze_default_10 = unsqueeze_default_11 = None
        le_scalar_1: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_tensor_1, 0);  sub_tensor_1 = None
        full_default_9: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_4: "f16[256, 257]" = torch.ops.aten.where.self(le_scalar_1, full_default_9, full_default_10);  le_scalar_1 = full_default_9 = full_default_10 = None
        rev_default_2: "f16[256, 257]" = torch.ops.prims.rev.default(where_self_4, [0]);  where_self_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_default_12: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_default_2, 0);  rev_default_2 = None
        unsqueeze_default_13: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_default_2: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_default_13, _shape_param_20);  _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_default_3: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_default_2, torch.bool);  expand_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_11: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        reshape_default_18: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_default_3, _shape_param_21);  select_scatter_default_3 = _shape_param_21 = None
        permute_default_18: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(reshape_default_18, [0, 2, 1, 3]);  reshape_default_18 = None
        slice_tensor_49: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_18, 1, 0, 256);  permute_default_18 = None
        slice_tensor_50: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_49, 3, 0, 257);  slice_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_self_5: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_default_3, full_default_11, slice_tensor_50);  convert_element_type_default_3 = full_default_11 = slice_tensor_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_default_10: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_tensor_48, where_self_5);  slice_tensor_48 = where_self_5 = None
        slice_scatter_default_18: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_46, copy_default_10, 3, 0, 257);  slice_tensor_46 = copy_default_10 = None
        slice_scatter_default_19: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_default_16, slice_scatter_default_18, 1, 0, 256);  permute_default_16 = slice_scatter_default_18 = None
        permute_default_19: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_default_19, [0, 2, 1, 3]);  slice_scatter_default_19 = None
        reshape_default_19: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_default_19, _shape_param_22);  permute_default_19 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        reshape_default_20: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(reshape_default_19, _shape_param_23);  _shape_param_23 = None
        permute_default_20: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(reshape_default_20, [0, 2, 1, 3]);  reshape_default_20 = None
        slice_tensor_51: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_20, 1, -256, 9223372036854775807)
        reshape_default_21: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(reshape_default_19, _shape_param_24);  _shape_param_24 = None
        permute_default_21: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(reshape_default_21, [0, 2, 1, 3]);  reshape_default_21 = None
        slice_tensor_52: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_21, 1, -256, 9223372036854775807);  permute_default_21 = None
        slice_tensor_53: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_52, 3, -257, 9223372036854775807);  slice_tensor_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_default_3: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_default_13, [1, 3]);  unsqueeze_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_default_3: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_default_3, _shape_param_25);  rev_default_3 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_default_4: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_default_3, torch.bool);  expand_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_12: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        reshape_default_22: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(reshape_default_19, _shape_param_26);  reshape_default_19 = _shape_param_26 = None
        permute_default_22: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(reshape_default_22, [0, 2, 1, 3]);  reshape_default_22 = None
        slice_tensor_54: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_22, 1, -256, 9223372036854775807);  permute_default_22 = None
        slice_tensor_55: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_54, 3, -257, 9223372036854775807);  slice_tensor_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_self_6: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_default_4, full_default_12, slice_tensor_55);  convert_element_type_default_4 = full_default_12 = slice_tensor_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_default_11: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_tensor_53, where_self_6);  slice_tensor_53 = where_self_6 = None
        slice_scatter_default_20: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_51, copy_default_11, 3, -257, 9223372036854775807);  slice_tensor_51 = copy_default_11 = None
        slice_scatter_default_21: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_default_20, slice_scatter_default_20, 1, -256, 9223372036854775807);  permute_default_20 = slice_scatter_default_20 = None
        permute_default_23: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_default_21, [0, 2, 1, 3]);  slice_scatter_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_default_24: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_default_23, [0, 2, 1, 3]);  permute_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_tensor: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_default_11, permute_default_24);  permute_default_11 = permute_default_24 = None
        permute_default_25: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1, 3]);  add_tensor = None
        permute_default_26: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_default_25, [0, 2, 1, 3]);  permute_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_default: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_default_26, memory_format = torch.contiguous_format);  permute_default_26 = None
        convert_element_type_default_5: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_default, torch.float32);  clone_default = None
        amax_default: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_default_5, [-1], True)
        sub_tensor_2: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_default_5, amax_default);  convert_element_type_default_5 = amax_default = None
        exp_default: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
        sum_dim_int_list: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_self_7: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, div_tensor);  unsqueeze_default_1 = full_default = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_default_6: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_self_7, torch.float16);  where_self_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_default_27: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_default_6, [0, 2, 1, 3]);  convert_element_type_default_6 = None
        reshape_default_23: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_default_27, _shape_param_27);  permute_default_27 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_2: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(reshape_default_23, [0, 257], 0.0);  reshape_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        reshape_default_24: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_default_2, _shape_param_28);  constant_pad_nd_default_2 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_tensor_56: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(reshape_default_24, 2, 0, -256);  reshape_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        reshape_default_25: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_tensor_56, _shape_param_29);  slice_tensor_56 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_tensor_57: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(reshape_default_25, 3, 0, -1);  reshape_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_default_14: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_57, 4);  slice_tensor_57 = None
        reshape_default_26: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_default_14, _shape_param_30);  unsqueeze_default_14 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default_27: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_68, _shape_param_31);  addmm_68 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        reshape_default_28: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_27, _shape_param_32);  reshape_default_27 = _shape_param_32 = None
        permute_default_28: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(reshape_default_28, [1, 0, 2, 3]);  reshape_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_29: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_default_28, [0, 2, 1, 3]);  permute_default_28 = None
        reshape_default_29: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_default_29, _shape_param_33);  permute_default_29 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_3: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(reshape_default_29, [0, 0, 256, 256], -1.0);  reshape_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_default_2: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_default_3, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_default_15: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default_2, 4);  as_strided_default_2 = None
        clone_default_1: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default_15, memory_format = torch.contiguous_format);  unsqueeze_default_15 = None
        reshape_default_30: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_34);  clone_default_1 = _shape_param_34 = None
        return (reshape_default_26, reshape_default_30)



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

"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 835da2618e7d
Shape hash: 096e2b52
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
    def forward(self, bmm_25: "f32[384, 256, 768]", gt_33: "b8[8, 1024, 12, 513]", primals_9: "b8[8, 1024]", div_117: "f32[8, 1024, 12, 513]", rev_1: "f32[1, 256, 1, 257]", unsqueeze_8: "f32[1, 256, 1, 257]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        reshape_default: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_25, _shape_param_0);  bmm_25 = _shape_param_0 = None
        squeeze_dim: "f32[96, 4, 256, 768]" = torch.ops.aten.squeeze.dim(reshape_default, 4);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        full_default: "f32[96, 4, 256, 769]" = torch.ops.aten.full.default([96, 4, 256, 769], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[96, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default, squeeze_dim, 3, 0, -1);  full_default = squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        reshape_default_1: "f32[96, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_default, _shape_param_1);  slice_scatter_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        full_default_1: "f32[96, 4, 197120]" = torch.ops.aten.full.default([96, 4, 197120], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_1: "f32[96, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_1, reshape_default_1, 2, 0, -256);  full_default_1 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        reshape_default_2: "f32[96, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_default_1, _shape_param_2);  slice_scatter_default_1 = _shape_param_2 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[96, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(reshape_default_2, [0, -257]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        reshape_default_3: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_default, _shape_param_3);  constant_pad_nd_default = _shape_param_3 = None
        permute_default: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_tensor: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_default, mul_tensor);  permute_default = mul_tensor = None
        clone_default: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_tensor_1, memory_format = torch.contiguous_format);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_default: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(primals_9, 2);  primals_9 = None
        unsqueeze_default_1: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 3);  unsqueeze_default = None
        where_self: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_2, clone_default);  unsqueeze_default_1 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_tensor_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_self, div_117);  where_self = None
        sum_dim_int_list: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[8, 1024, 12, 513]" = torch.ops.aten.neg.default(div_117);  div_117 = None
        fma_default: "f32[8, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_default_1: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_default, [0, 2, 1, 3]);  fma_default = None
        clone_default_1: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        reshape_default_5: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None
        clone_default_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format)
        copy_default: "f32[8, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_default_2, clone_default_2);  permute_default_2 = clone_default_2 = None
        permute_default_3: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_default, [0, 2, 1, 3]);  copy_default = None
        reshape_default_6: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_6);  permute_default_3 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        reshape_default_7: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_7);  reshape_default_6 = _shape_param_7 = None
        permute_default_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(reshape_default_7, [0, 2, 1, 3]);  reshape_default_7 = None
        slice_tensor: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_4, 1, -256, 9223372036854775807)
        slice_tensor_1: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, -257, 9223372036854775807)
        clone_default_3: "f32[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_tensor_1, memory_format = torch.contiguous_format)
        full_default_3: "f32[8, 256, 12, 257]" = torch.ops.aten.full.default([8, 256, 12, 257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_default_1: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_1, full_default_3);  slice_tensor_1 = None
        slice_scatter_default_2: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor, copy_default_1, 3, -257, 9223372036854775807);  slice_tensor = copy_default_1 = None
        slice_scatter_default_3: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_4, slice_scatter_default_2, 1, -256, 9223372036854775807);  permute_default_4 = slice_scatter_default_2 = None
        permute_default_5: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_3, [0, 2, 1, 3]);  slice_scatter_default_3 = None
        reshape_default_8: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_8);  permute_default_5 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_default: "f32[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, _shape_param_9);  rev_1 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_default_1: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_default, torch.bool);  expand_default = None
        where_self_1: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_default_1, full_default_2, clone_default_3);  convert_element_type_default_1 = clone_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        full_default_4: "f32[8, 256, 12, 513]" = torch.ops.aten.full.default([8, 256, 12, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_4: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_4, where_self_1, 3, -257, 9223372036854775807);  where_self_1 = None
        full_default_5: "f32[8, 1024, 12, 513]" = torch.ops.aten.full.default([8, 1024, 12, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_5: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_default_4, 1, -256, 9223372036854775807);  slice_scatter_default_4 = None
        permute_default_6: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_5, [0, 2, 1, 3]);  slice_scatter_default_5 = None
        clone_default_4: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_6, memory_format = torch.contiguous_format);  permute_default_6 = None
        reshape_default_9: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_default_4, _shape_param_10);  clone_default_4 = _shape_param_10 = None
        add_tensor: "f32[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(reshape_default_8, reshape_default_9);  reshape_default_8 = reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        reshape_default_10: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_11);  add_tensor = _shape_param_11 = None
        permute_default_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(reshape_default_10, [0, 2, 1, 3]);  reshape_default_10 = None
        slice_tensor_2: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_7, 1, 0, 256)
        slice_tensor_3: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 0, 257)
        clone_default_5: "f32[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_tensor_3, memory_format = torch.contiguous_format)
        copy_default_2: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_3, full_default_3);  slice_tensor_3 = full_default_3 = None
        slice_scatter_default_6: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_2, copy_default_2, 3, 0, 257);  slice_tensor_2 = copy_default_2 = None
        slice_scatter_default_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_7, slice_scatter_default_6, 1, 0, 256);  permute_default_7 = slice_scatter_default_6 = None
        permute_default_8: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_7, [0, 2, 1, 3]);  slice_scatter_default_7 = None
        reshape_default_11: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_default_8, _shape_param_12);  permute_default_8 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_default_1: "f32[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_8, _shape_param_13);  unsqueeze_8 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_default_2: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_default_1, torch.bool);  expand_default_1 = None
        where_self_2: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_default_2, full_default_2, clone_default_5);  convert_element_type_default_2 = full_default_2 = clone_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_default_8: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_4, where_self_2, 3, 0, 257);  full_default_4 = where_self_2 = None
        slice_scatter_default_9: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_default_8, 1, 0, 256);  full_default_5 = slice_scatter_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_default_9: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_9, [0, 2, 1, 3]);  slice_scatter_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_default_6: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_9, memory_format = torch.contiguous_format);  permute_default_9 = None
        reshape_default_12: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_default_6, _shape_param_14);  clone_default_6 = _shape_param_14 = None
        add_tensor_1: "f32[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(reshape_default_11, reshape_default_12);  reshape_default_11 = reshape_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_int: "f32[96, 256, 513]" = torch.ops.aten.select.int(add_tensor_1, 1, 0)
        slice_tensor_4: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_int, 1, 1, 256)
        slice_tensor_5: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 2, 1, 256)
        clone_default_7: "f32[96, 255, 255]" = torch.ops.aten.clone.default(slice_tensor_5, memory_format = torch.contiguous_format)
        full_default_6: "f32[96, 255, 255]" = torch.ops.aten.full.default([96, 255, 255], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_default_3: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_5, full_default_6);  slice_tensor_5 = full_default_6 = None
        slice_scatter_default_10: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_4, copy_default_3, 2, 1, 256);  slice_tensor_4 = copy_default_3 = None
        slice_scatter_default_11: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int, slice_scatter_default_10, 1, 1, 256);  select_int = slice_scatter_default_10 = None
        select_scatter_default: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_tensor_1, slice_scatter_default_11, 1, 0);  add_tensor_1 = slice_scatter_default_11 = None
        full_default_7: "f32[96, 255, 513]" = torch.ops.aten.full.default([96, 255, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_12: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_7, clone_default_7, 2, -255, 9223372036854775807);  full_default_7 = clone_default_7 = None
        full_default_8: "f32[96, 512, 513]" = torch.ops.aten.full.default([96, 512, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_13: "f32[96, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_8, slice_scatter_default_12, 1, 0, 255);  slice_scatter_default_12 = None
        full_default_9: "f32[96, 3, 512, 513]" = torch.ops.aten.full.default([96, 3, 512, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default_1: "f32[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_9, slice_scatter_default_13, 1, 0);  slice_scatter_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_tensor_6: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_tensor_7: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_6, 3, 0, 256)
        clone_default_8: "f32[96, 3, 256, 256]" = torch.ops.aten.clone.default(slice_tensor_7, memory_format = torch.contiguous_format)
        full_default_10: "f32[96, 3, 256, 256]" = torch.ops.aten.full.default([96, 3, 256, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_default_4: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_7, full_default_10);  slice_tensor_7 = full_default_10 = None
        slice_scatter_default_14: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_6, copy_default_4, 3, 0, 256);  slice_tensor_6 = copy_default_4 = None
        slice_scatter_default_15: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default, slice_scatter_default_14, 1, 1, 9223372036854775807);  select_scatter_default = slice_scatter_default_14 = None
        full_default_11: "f32[96, 3, 256, 513]" = torch.ops.aten.full.default([96, 3, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_16: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_11, clone_default_8, 3, 257, 9223372036854775807);  clone_default_8 = None
        slice_scatter_default_17: "f32[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_9, slice_scatter_default_16, 2, -257, -1);  slice_scatter_default_16 = None
        add_tensor_2: "f32[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_default_1, slice_scatter_default_17);  select_scatter_default_1 = slice_scatter_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_int_1: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, -1)
        slice_tensor_8: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_1, 2, 256, 9223372036854775807)
        clone_default_9: "f32[96, 256, 257]" = torch.ops.aten.clone.default(slice_tensor_8, memory_format = torch.contiguous_format)
        full_default_12: "f32[96, 256, 257]" = torch.ops.aten.full.default([96, 256, 257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_default_5: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_8, full_default_12);  slice_tensor_8 = full_default_12 = None
        slice_scatter_default_18: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_1, copy_default_5, 2, 256, 9223372036854775807);  select_int_1 = copy_default_5 = None
        select_scatter_default_2: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_15, slice_scatter_default_18, 1, -1);  slice_scatter_default_15 = slice_scatter_default_18 = None
        full_default_13: "f32[96, 256, 513]" = torch.ops.aten.full.default([96, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_19: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_13, clone_default_9, 2, 0, 257);  full_default_13 = clone_default_9 = None
        slice_scatter_default_20: "f32[96, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_8, slice_scatter_default_19, 1, 256, 9223372036854775807);  full_default_8 = slice_scatter_default_19 = None
        select_scatter_default_3: "f32[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_9, slice_scatter_default_20, 1, -1);  slice_scatter_default_20 = None
        add_tensor_3: "f32[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_tensor_2, select_scatter_default_3);  add_tensor_2 = select_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_tensor_9: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 0, -1);  select_scatter_default_2 = None
        slice_tensor_10: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_9, 3, 256, 9223372036854775807);  slice_tensor_9 = None
        clone_default_10: "f32[96, 3, 256, 257]" = torch.ops.aten.clone.default(slice_tensor_10, memory_format = torch.contiguous_format);  slice_tensor_10 = None
        slice_scatter_default_21: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_11, clone_default_10, 3, 0, 257);  full_default_11 = clone_default_10 = None
        slice_scatter_default_22: "f32[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_9, slice_scatter_default_21, 2, 0, 256);  full_default_9 = slice_scatter_default_21 = None
        add_tensor_4: "f32[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_scatter_default_22);  add_tensor_3 = slice_scatter_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        reshape_default_13: "f32[96, 3, 513, 512]" = torch.ops.aten.reshape.default(add_tensor_4, _shape_param_15);  add_tensor_4 = _shape_param_15 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_1: "f32[96, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(reshape_default_13, [0, 0, 0, -1]);  reshape_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        reshape_default_14: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_default_1, _shape_param_16);  constant_pad_nd_default_1 = _shape_param_16 = None
        permute_default_10: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.permute.default(reshape_default_14, [0, 1, 2, 4, 3]);  reshape_default_14 = None
        reshape_default_15: "f32[288, 512, 512]" = torch.ops.aten.reshape.default(permute_default_10, _shape_param_17);  permute_default_10 = _shape_param_17 = None
        return reshape_default_15


def _default_make_inputs():
    return [
    torch.randn([384, 256, 768], dtype=torch.float32, device='cuda'),
    torch.randn(51380108, dtype=torch.bool, device='cuda').as_strided([8, 1024, 12, 513], [6422528, 6272, 513, 1]),  # gt_33
    torch.randint(0, 2, [8, 1024], dtype=torch.bool, device='cuda'),
    torch.randn(50593772, dtype=torch.float32, device='cuda').as_strided([8, 1024, 12, 513], [6324224, 6176, 513, 1]),  # div_117
    torch.randn([1, 256, 1, 257], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 257], dtype=torch.float32, device='cuda'),
    [96, 4, 256, 768, 1],  # _shape_param_0
    [96, 4, 196864],  # _shape_param_1
    [96, 4, 256, 770],  # _shape_param_2
    [8, 12, 1024, 513],  # _shape_param_3
    [96, 4, 256, 513],  # _shape_param_4
    [8, 12, 1024, 513],  # _shape_param_5
    [96, 4, 256, 513],  # _shape_param_6
    [8, 12, 1024, 513],  # _shape_param_7
    [96, 4, 256, 513],  # _shape_param_8
    [8, 256, 12, 257],  # _shape_param_9
    [96, 4, 256, 513],  # _shape_param_10
    [8, 12, 1024, 513],  # _shape_param_11
    [96, 4, 256, 513],  # _shape_param_12
    [8, 256, 12, 257],  # _shape_param_13
    [96, 4, 256, 513],  # _shape_param_14
    [96, 3, 513, 512],  # _shape_param_15
    [96, 3, 512, 512, 1],  # _shape_param_16
    [288, 512, 512],  # _shape_param_17
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

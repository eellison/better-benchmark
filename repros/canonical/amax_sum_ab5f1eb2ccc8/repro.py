"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train
Pattern hash: ab5f1eb2ccc8
Shape hash: b2c6c352
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
_shapes_config = "(T([72, 512, 512], f32), T([24, 3, 256, 257], f32, stride=(525312, 131328, 513, 1)), T([24, 3, 256, 513], f32, stride=(525312, 131328, 513, 1)), T([24, 4, 256, 513], f32), T([2, 256, 12, 257], b8), T([2, 256, 12, 257], f32, stride=(789504, 257, 65792, 1)), T([2, 256, 12, 257], b8), T([2, 1024, 1, 513], f32), T([2, 1024, 1, 1], b8), T([], f32), T([36], i64), S([24, 3, 512, 1, 512]), S([24, 3, 512, 512]), S([24, 3, 512, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([24, 4, -1]), S([24, 4, 256, 769]), S([96, 256, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_22: "f32[72, 512, 512]", slice_4: "f32[24, 3, 256, 257]", slice_3: "f32[24, 3, 256, 513]", full: "f32[24, 4, 256, 513]", convert_element_type: "b8[2, 256, 12, 257]", permute_35: "f32[2, 256, 12, 257]", convert_element_type_1: "b8[2, 256, 12, 257]", permute_82: "f32[2, 1024, 1, 513]", unsqueeze_18: "b8[2, 1024, 1, 1]", full_default_1: "f32[]", inductor_seeds_default: "i64[36]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        reshape_default: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_22, _shape_param_0);  bmm_22 = _shape_param_0 = None
        permute_default: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 2, 4, 3]);  reshape_default = None
        reshape_default_1: "f32[24, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[24, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(reshape_default_1, [0, 0, 0, 1], 0.0);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        reshape_default_2: "f32[24, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_default, _shape_param_2);  constant_pad_nd_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_tensor: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(reshape_default_2, 2, 0, 256)
        slice_tensor_1: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 257);  slice_tensor = None
        copy_default: "f32[24, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_tensor_1);  slice_4 = slice_tensor_1 = None
        slice_scatter_default: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_default, 3, 256, 9223372036854775807);  slice_3 = copy_default = None
        slice_scatter_default_1: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_default, 1, 0, -1);  full = slice_scatter_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_int: "f32[24, 512, 513]" = torch.ops.aten.select.int(reshape_default_2, 1, -1)
        slice_tensor_2: "f32[24, 256, 513]" = torch.ops.aten.slice.Tensor(select_int, 1, 256, 9223372036854775807);  select_int = None
        slice_tensor_3: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 2, 0, 257);  slice_tensor_2 = None
        select_int_1: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        slice_tensor_4: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_1, 2, 256, 9223372036854775807)
        copy_default_1: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_4, slice_tensor_3);  slice_tensor_4 = slice_tensor_3 = None
        slice_scatter_default_2: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_1, copy_default_1, 2, 256, 9223372036854775807);  select_int_1 = copy_default_1 = None
        select_scatter_default: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_1, slice_scatter_default_2, 1, -1);  slice_scatter_default_1 = slice_scatter_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_tensor_5: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(reshape_default_2, 2, -257, -1)
        slice_tensor_6: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_5, 3, 257, 9223372036854775807);  slice_tensor_5 = None
        slice_tensor_7: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_tensor_8: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_7, 3, 0, 256)
        copy_default_2: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_8, slice_tensor_6);  slice_tensor_8 = slice_tensor_6 = None
        slice_scatter_default_3: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_7, copy_default_2, 3, 0, 256);  slice_tensor_7 = copy_default_2 = None
        slice_scatter_default_4: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default, slice_scatter_default_3, 1, 1, 9223372036854775807);  select_scatter_default = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_int_2: "f32[24, 512, 513]" = torch.ops.aten.select.int(reshape_default_2, 1, 0);  reshape_default_2 = None
        slice_tensor_9: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_2, 1, 0, 255);  select_int_2 = None
        slice_tensor_10: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_9, 2, -255, 9223372036854775807);  slice_tensor_9 = None
        select_int_3: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_4, 1, 0)
        slice_tensor_11: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_3, 1, 1, 256)
        slice_tensor_12: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_11, 2, 1, 256)
        copy_default_3: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_12, slice_tensor_10);  slice_tensor_12 = slice_tensor_10 = None
        slice_scatter_default_5: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_11, copy_default_3, 2, 1, 256);  slice_tensor_11 = copy_default_3 = None
        slice_scatter_default_6: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_3, slice_scatter_default_5, 1, 1, 256);  select_int_3 = slice_scatter_default_5 = None
        select_scatter_default_1: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_4, slice_scatter_default_6, 1, 0);  slice_scatter_default_4 = slice_scatter_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        reshape_default_3: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_1, _shape_param_3);  select_scatter_default_1 = _shape_param_3 = None
        permute_default_1: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        slice_tensor_13: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_1, 1, 0, 256)
        slice_tensor_14: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_13, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_self: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_tensor_14);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_default_4: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_14, where_self);  slice_tensor_14 = where_self = None
        slice_scatter_default_7: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_13, copy_default_4, 3, 0, 257);  slice_tensor_13 = copy_default_4 = None
        slice_scatter_default_8: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_1, slice_scatter_default_7, 1, 0, 256);  permute_default_1 = slice_scatter_default_7 = None
        permute_default_2: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_8, [0, 2, 1, 3]);  slice_scatter_default_8 = None
        reshape_default_4: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_4);  permute_default_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        reshape_default_5: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_3: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None
        slice_tensor_15: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_3, 1, -256, 9223372036854775807)
        slice_tensor_16: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_15, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_self_1: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_tensor_16);  convert_element_type_1 = permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_default_5: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_16, where_self_1);  slice_tensor_16 = where_self_1 = None
        slice_scatter_default_9: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_15, copy_default_5, 3, -257, 9223372036854775807);  slice_tensor_15 = copy_default_5 = None
        slice_scatter_default_10: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_3, slice_scatter_default_9, 1, -256, 9223372036854775807);  permute_default_3 = slice_scatter_default_9 = None
        permute_default_4: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_10, [0, 2, 1, 3]);  slice_scatter_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_default_5: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_default_4, [0, 2, 1, 3]);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_tensor: "f32[2, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_default_5, permute_82);  permute_default_5 = permute_82 = None
        permute_default_6: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1, 3]);  add_tensor = None
        permute_default_7: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_default_6, [0, 2, 1, 3]);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_default: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        amax_default: "f32[2, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_default, [-1], True)
        sub_tensor: "f32[2, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_default, amax_default);  clone_default = amax_default = None
        exp_default: "f32[2, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[2, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_self_2: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_tensor);  unsqueeze_18 = full_default_1 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33);  inductor_seeds_default = None
        inductor_random_default: "f32[2, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([2, 1024, 12, 513], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[2, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_scalar, where_self_2);  gt_scalar = where_self_2 = None
        mul_tensor_1: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_default_8: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_tensor_1, [0, 2, 1, 3]);  mul_tensor_1 = None
        clone_default_1: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_6: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_1: "f32[24, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(reshape_default_6, [0, 257], 0.0);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        reshape_default_7: "f32[24, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_default_1, _shape_param_7);  constant_pad_nd_default_1 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_tensor_17: "f32[24, 4, 196864]" = torch.ops.aten.slice.Tensor(reshape_default_7, 2, 0, -256);  reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        reshape_default_8: "f32[24, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_tensor_17, _shape_param_8);  slice_tensor_17 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_tensor_18: "f32[24, 4, 256, 768]" = torch.ops.aten.slice.Tensor(reshape_default_8, 3, 0, -1);  reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_default: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_18, 4);  slice_tensor_18 = None
        reshape_default_9: "f32[96, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_9);  unsqueeze_default = _shape_param_9 = None
        return reshape_default_9



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

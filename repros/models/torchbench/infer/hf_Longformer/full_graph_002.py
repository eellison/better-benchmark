import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 4096, 768]", arg1_1: "f16[768, 768]", arg2_1: "f16[768]", arg3_1: "f16[768, 768]", arg4_1: "f16[768]", arg5_1: "f16[768, 768]", arg6_1: "f16[768]", arg7_1: "f16[1, 4096]", arg8_1: "b8[1, 4096]", arg9_1: "f16[768, 768]", arg10_1: "f16[768]", arg11_1: "f16[768]", arg12_1: "f16[768]", arg13_1: "f16[3072, 768]", arg14_1: "f16[3072]", arg15_1: "f16[768, 3072]", arg16_1: "f16[768]", arg17_1: "f16[768]", arg18_1: "f16[768]", arg19_1: "f16[768, 768]", arg20_1: "f16[768]", arg21_1: "f16[768, 768]", arg22_1: "f16[768]", arg23_1: "f16[768, 768]", arg24_1: "f16[768]", arg25_1: "f16[768, 768]", arg26_1: "f16[768]", arg27_1: "f16[768]", arg28_1: "f16[768]", arg29_1: "f16[3072, 768]", arg30_1: "f16[3072]", arg31_1: "f16[768, 3072]", arg32_1: "f16[768]", arg33_1: "f16[768]", arg34_1: "f16[768]", arg35_1: "f16[768, 768]", arg36_1: "f16[768]", arg37_1: "f16[768, 768]", arg38_1: "f16[768]", arg39_1: "f16[768, 768]", arg40_1: "f16[768]", arg41_1: "f16[768, 768]", arg42_1: "f16[768]", arg43_1: "f16[768]", arg44_1: "f16[768]", arg45_1: "f16[3072, 768]", arg46_1: "f16[3072]", arg47_1: "f16[768, 3072]", arg48_1: "f16[768]", arg49_1: "f16[768]", arg50_1: "f16[768]", arg51_1: "f16[768, 768]", arg52_1: "f16[768]", arg53_1: "f16[768, 768]", arg54_1: "f16[768]", arg55_1: "f16[768, 768]", arg56_1: "f16[768]", arg57_1: "f16[768, 768]", arg58_1: "f16[768]", arg59_1: "f16[768]", arg60_1: "f16[768]", arg61_1: "f16[3072, 768]", arg62_1: "f16[3072]", arg63_1: "f16[768, 3072]", arg64_1: "f16[768]", arg65_1: "f16[768]", arg66_1: "f16[768]", arg67_1: "f16[768, 768]", arg68_1: "f16[768]", arg69_1: "f16[768, 768]", arg70_1: "f16[768]", arg71_1: "f16[768, 768]", arg72_1: "f16[768]", arg73_1: "f16[768, 768]", arg74_1: "f16[768]", arg75_1: "f16[768]", arg76_1: "f16[768]", arg77_1: "f16[3072, 768]", arg78_1: "f16[3072]", arg79_1: "f16[768, 3072]", arg80_1: "f16[768]", arg81_1: "f16[768]", arg82_1: "f16[768]", arg83_1: "f16[768, 768]", arg84_1: "f16[768]", arg85_1: "f16[768, 768]", arg86_1: "f16[768]", arg87_1: "f16[768, 768]", arg88_1: "f16[768]", arg89_1: "f16[768, 768]", arg90_1: "f16[768]", arg91_1: "f16[768]", arg92_1: "f16[768]", arg93_1: "f16[3072, 768]", arg94_1: "f16[3072]", arg95_1: "f16[768, 3072]", arg96_1: "f16[768]", arg97_1: "f16[768]", arg98_1: "f16[768]", arg99_1: "f16[768, 768]", arg100_1: "f16[768]", arg101_1: "f16[768, 768]", arg102_1: "f16[768]", arg103_1: "f16[768, 768]", arg104_1: "f16[768]", arg105_1: "f16[768, 768]", arg106_1: "f16[768]", arg107_1: "f16[768]", arg108_1: "f16[768]", arg109_1: "f16[3072, 768]", arg110_1: "f16[3072]", arg111_1: "f16[768, 3072]", arg112_1: "f16[768]", arg113_1: "f16[768]", arg114_1: "f16[768]", arg115_1: "f16[768, 768]", arg116_1: "f16[768]", arg117_1: "f16[768, 768]", arg118_1: "f16[768]", arg119_1: "f16[768, 768]", arg120_1: "f16[768]", arg121_1: "f16[768, 768]", arg122_1: "f16[768]", arg123_1: "f16[768]", arg124_1: "f16[768]", arg125_1: "f16[3072, 768]", arg126_1: "f16[3072]", arg127_1: "f16[768, 3072]", arg128_1: "f16[768]", arg129_1: "f16[768]", arg130_1: "f16[768]", arg131_1: "f16[768, 768]", arg132_1: "f16[768]", arg133_1: "f16[768, 768]", arg134_1: "f16[768]", arg135_1: "f16[768, 768]", arg136_1: "f16[768]", arg137_1: "f16[768, 768]", arg138_1: "f16[768]", arg139_1: "f16[768]", arg140_1: "f16[768]", arg141_1: "f16[3072, 768]", arg142_1: "f16[3072]", arg143_1: "f16[768, 3072]", arg144_1: "f16[768]", arg145_1: "f16[768]", arg146_1: "f16[768]", arg147_1: "f16[768, 768]", arg148_1: "f16[768]", arg149_1: "f16[768, 768]", arg150_1: "f16[768]", arg151_1: "f16[768, 768]", arg152_1: "f16[768]", arg153_1: "f16[768, 768]", arg154_1: "f16[768]", arg155_1: "f16[768]", arg156_1: "f16[768]", arg157_1: "f16[3072, 768]", arg158_1: "f16[3072]", arg159_1: "f16[768, 3072]", arg160_1: "f16[768]", arg161_1: "f16[768]", arg162_1: "f16[768]", arg163_1: "f16[768, 768]", arg164_1: "f16[768]", arg165_1: "f16[768, 768]", arg166_1: "f16[768]", arg167_1: "f16[768, 768]", arg168_1: "f16[768]", arg169_1: "f16[768, 768]", arg170_1: "f16[768]", arg171_1: "f16[768]", arg172_1: "f16[768]", arg173_1: "f16[3072, 768]", arg174_1: "f16[3072]", arg175_1: "f16[768, 3072]", arg176_1: "f16[768]", arg177_1: "f16[768]", arg178_1: "f16[768]", arg179_1: "f16[768, 768]", arg180_1: "f16[768]", arg181_1: "f16[768, 768]", arg182_1: "f16[768]", arg183_1: "f16[768, 768]", arg184_1: "f16[768]", arg185_1: "f16[768, 768]", arg186_1: "f16[768]", arg187_1: "f16[768]", arg188_1: "f16[768]", arg189_1: "f16[3072, 768]", arg190_1: "f16[3072]", arg191_1: "f16[768, 3072]", arg192_1: "f16[768]", arg193_1: "f16[768]", arg194_1: "f16[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_7: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_8: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_17: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_18: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_5: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_3: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_4: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 256, 9223372036854775807);  slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute, [4096, 768])
        permute_1: "f16[768, 768]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg2_1, view, permute_1);  arg2_1 = view = permute_1 = None
        view_1: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm, [4096, 1, 768]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_1, 8.0);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_40: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div, [4096, 1, 12, 64]);  div = None
        permute_25: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_40, [1, 0, 2, 3]);  view_40 = None
        permute_26: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_25, [0, 2, 1, 3]);  permute_25 = None
        view_41: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_26, [12, 4096, 64]);  permute_26 = None
        view_42: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_41, [12, 8, 512, 64]);  view_41 = None
        as_strided_5: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_42, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_42 = None
        unsqueeze_4: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_5, 4);  as_strided_5 = None
        clone: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_4, memory_format = torch.contiguous_format);  unsqueeze_4 = None
        view_43: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone, [180, 512, 64]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_2: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute, [4096, 768])
        permute_2: "f16[768, 768]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg4_1, view_2, permute_2);  arg4_1 = view_2 = permute_2 = None
        view_3: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_1, [4096, 1, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_11: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_3, [4096, 1, 12, 64]);  view_3 = None
        permute_5: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_11, [1, 0, 2, 3]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_10: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        view_17: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_10, [12, 4096, 64]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_26: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_17, [12, 8, 512, 64]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_1: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_26, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_2: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        permute_20: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 4, 2, 3]);  unsqueeze_2 = None
        permute_29: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_20, [0, 1, 4, 3, 2]);  permute_20 = None
        clone_1: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_44: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_1, [180, 64, 512]);  clone_1 = None
        bmm: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
        view_45: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm, [12, 15, 512, 1, 512]);  bmm = None
        permute_30: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_45, [0, 1, 2, 4, 3]);  view_45 = None
        view_46: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_30, [12, 15, 512, 512]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_46, [0, 0, 0, 1], 0.0);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_47: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd, [12, 15, 512, 513]);  constant_pad_nd = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_47, 2, 0, 256)
        slice_2: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 257);  slice_1 = None
        copy: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_2);  slice_4 = slice_2 = None
        slice_scatter: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_5, copy, 3, 256, 9223372036854775807);  slice_5 = copy = None
        slice_scatter_1: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter, 1, 0, -1);  full = slice_scatter = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_4: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        select_3: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        slice_11: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_3, 2, 256, 9223372036854775807);  select_3 = None
        select: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_47, 1, -1)
        slice_8: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807);  select = None
        slice_9: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_8, 2, 0, 257);  slice_8 = None
        copy_1: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_11, slice_9);  slice_11 = slice_9 = None
        slice_scatter_2: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_4, copy_1, 2, 256, 9223372036854775807);  select_4 = copy_1 = None
        select_scatter: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_1, slice_scatter_2, 1, -1);  slice_scatter_1 = slice_scatter_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_20: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_18: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_19: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_18, 3, 0, 256);  slice_18 = None
        slice_13: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_47, 2, -257, -1)
        slice_14: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_13, 3, 257, 9223372036854775807);  slice_13 = None
        copy_2: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_19, slice_14);  slice_19 = slice_14 = None
        slice_scatter_3: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_20, copy_2, 3, 0, 256);  slice_20 = copy_2 = None
        slice_scatter_4: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_3, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_11: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_30: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_11, 1, 1, 256)
        select_10: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_28: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_10, 1, 1, 256);  select_10 = None
        slice_29: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_28, 2, 1, 256);  slice_28 = None
        select_6: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_47, 1, 0);  view_47 = None
        slice_23: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_6, 1, 0, 255);  select_6 = None
        slice_24: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_23, 2, -255, 9223372036854775807);  slice_23 = None
        copy_3: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_29, slice_24);  slice_29 = slice_24 = None
        slice_scatter_5: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_30, copy_3, 2, 1, 256);  slice_30 = copy_3 = None
        slice_scatter_6: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_11, slice_scatter_5, 1, 1, 256);  select_11 = slice_scatter_5 = None
        select_scatter_1: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_4, slice_scatter_6, 1, 0);  slice_scatter_4 = slice_scatter_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_55: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_1, [1, 12, 4096, 513])
        permute_38: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        slice_43: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_38, 1, 0, 256)
        view_54: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_1, [1, 12, 4096, 513])
        permute_37: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        slice_41: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_37, 1, 0, 256);  permute_37 = None
        slice_42: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_41, 3, 0, 257);  slice_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_5: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub_1: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_5, unsqueeze_6);  unsqueeze_5 = unsqueeze_6 = None
        le: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_1, 0);  sub_1 = None
        full_default: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[256, 257]" = torch.ops.aten.where.self(le, full_default, full_default_1);  le = full_default = full_default_1 = None
        rev: "f16[256, 257]" = torch.ops.prims.rev.default(where, [0]);  where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_7: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
        unsqueeze_8: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_8, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_11: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_2: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_35: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_2, [0, 2, 1, 3]);  full_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_52: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_1, [1, 12, 4096, 513]);  select_scatter_1 = None
        permute_34: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        slice_36: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_34, 1, 0, 256);  permute_34 = None
        slice_37: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_36, 3, 0, 257);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_1: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_11, permute_35, slice_37);  convert_element_type_11 = permute_35 = slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_4: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_42, where_1);  slice_42 = where_1 = None
        slice_scatter_7: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_43, copy_4, 3, 0, 257);  slice_43 = copy_4 = None
        slice_scatter_8: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_38, slice_scatter_7, 1, 0, 256);  permute_38 = slice_scatter_7 = None
        permute_39: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3]);  slice_scatter_8 = None
        view_56: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_39, [12, 16, 256, 513]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_63: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_56, [1, 12, 4096, 513])
        permute_47: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None
        slice_56: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_47, 1, -256, 9223372036854775807)
        view_62: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_56, [1, 12, 4096, 513])
        permute_46: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        slice_54: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_46, 1, -256, 9223372036854775807);  permute_46 = None
        slice_55: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_54, 3, -257, 9223372036854775807);  slice_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_1: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_8, [1, 3]);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_1: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, [1, 256, 12, 257]);  rev_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_12: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_3: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_44: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_3, [0, 2, 1, 3]);  full_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_60: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_56, [1, 12, 4096, 513]);  view_56 = None
        permute_43: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        slice_49: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_43, 1, -256, 9223372036854775807);  permute_43 = None
        slice_50: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_49, 3, -257, 9223372036854775807);  slice_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_2: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_12, permute_44, slice_50);  convert_element_type_12 = permute_44 = slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_5: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_55, where_2);  slice_55 = where_2 = None
        slice_scatter_9: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_56, copy_5, 3, -257, 9223372036854775807);  slice_56 = copy_5 = None
        slice_scatter_10: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_47, slice_scatter_9, 1, -256, 9223372036854775807);  permute_47 = slice_scatter_9 = None
        permute_48: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3]);  slice_scatter_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_73: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_48, [0, 2, 1, 3]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_5: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_63: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_61: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_62: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_61, 3, 256, 9223372036854775807);  slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_4: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_50: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
        view_66: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_50, [1, 4096, 1]);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_68: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_66, [1, 8, 512, 1]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_6: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_68, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_11: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_6, 4);  as_strided_6 = None
        permute_52: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_11, [0, 1, 2, 4, 3]);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_2: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_9: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne, 2);  ne = None
        unsqueeze_10: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_13: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_10, torch.float16)
        where_3: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_10, full_default_2, convert_element_type_13);  unsqueeze_10 = full_default_2 = convert_element_type_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_51: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
        view_67: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_51, [1, 4096, 1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_69: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_67, [1, 8, 512, 1]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_7: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_69, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_12: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_7, 4);  as_strided_7 = None
        permute_53: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_12, [0, 1, 4, 2, 3]);  unsqueeze_12 = None
        mul: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_52, permute_53);  permute_52 = permute_53 = None
        view_70: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul, [1, 15, 512, 512]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_70, [0, 0, 0, 1], 0.0);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_71: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_1, [1, 15, 512, 513]);  constant_pad_nd_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_59: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_71, 2, 0, 256)
        slice_60: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_59, 3, 0, 257);  slice_59 = None
        copy_6: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_62, slice_60);  slice_62 = slice_60 = None
        slice_scatter_11: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_63, copy_6, 3, 256, 9223372036854775807);  slice_63 = copy_6 = None
        slice_scatter_12: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_11, 1, 0, -1);  full_5 = slice_scatter_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_17: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        select_16: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        slice_69: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_16, 2, 256, 9223372036854775807);  select_16 = None
        select_13: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_71, 1, -1)
        slice_66: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_13, 1, 256, 9223372036854775807);  select_13 = None
        slice_67: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_66, 2, 0, 257);  slice_66 = None
        copy_7: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_69, slice_67);  slice_69 = slice_67 = None
        slice_scatter_13: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_17, copy_7, 2, 256, 9223372036854775807);  select_17 = copy_7 = None
        select_scatter_2: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_12, slice_scatter_13, 1, -1);  slice_scatter_12 = slice_scatter_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_78: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_76: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_77: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_76, 3, 0, 256);  slice_76 = None
        slice_71: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_71, 2, -257, -1)
        slice_72: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_71, 3, 257, 9223372036854775807);  slice_71 = None
        copy_8: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_77, slice_72);  slice_77 = slice_72 = None
        slice_scatter_14: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_78, copy_8, 3, 0, 256);  slice_78 = copy_8 = None
        slice_scatter_15: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_2, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter_2 = slice_scatter_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_24: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_88: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_24, 1, 1, 256)
        select_23: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_86: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_23, 1, 1, 256);  select_23 = None
        slice_87: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_86, 2, 1, 256);  slice_86 = None
        select_19: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_71, 1, 0);  view_71 = None
        slice_81: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_19, 1, 0, 255);  select_19 = None
        slice_82: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_81, 2, -255, 9223372036854775807);  slice_81 = None
        copy_9: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_87, slice_82);  slice_87 = slice_82 = None
        slice_scatter_16: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_88, copy_9, 2, 1, 256);  slice_88 = copy_9 = None
        slice_scatter_17: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_24, slice_scatter_16, 1, 1, 256);  select_24 = slice_scatter_16 = None
        select_scatter_3: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_17, 1, 0);  slice_scatter_15 = slice_scatter_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_79: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_3, [1, 1, 4096, 513])
        permute_61: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
        slice_101: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_61, 1, 0, 256)
        view_78: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_3, [1, 1, 4096, 513])
        permute_60: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        slice_99: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_60, 1, 0, 256);  permute_60 = None
        slice_100: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_99, 3, 0, 257);  slice_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_2: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_13: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_2, -2);  iota_2 = None
        iota_3: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_14: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
        sub_3: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_13, unsqueeze_14);  unsqueeze_13 = unsqueeze_14 = None
        le_1: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_3, 0);  sub_3 = None
        full_default_3: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f16[256, 257]" = torch.ops.aten.where.self(le_1, full_default_3, full_default_4);  le_1 = full_default_3 = full_default_4 = None
        rev_2: "f16[256, 257]" = torch.ops.prims.rev.default(where_4, [0]);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_15: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_2, 0);  rev_2 = None
        unsqueeze_16: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_2: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_16, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_14: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_5: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_76: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_3, [1, 1, 4096, 513]);  select_scatter_3 = None
        permute_57: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        slice_94: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_57, 1, 0, 256);  permute_57 = None
        slice_95: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_94, 3, 0, 257);  slice_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_5: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_14, full_default_5, slice_95);  convert_element_type_14 = full_default_5 = slice_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_10: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_100, where_5);  slice_100 = where_5 = None
        slice_scatter_18: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_101, copy_10, 3, 0, 257);  slice_101 = copy_10 = None
        slice_scatter_19: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_61, slice_scatter_18, 1, 0, 256);  permute_61 = slice_scatter_18 = None
        permute_62: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_19, [0, 2, 1, 3]);  slice_scatter_19 = None
        view_80: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_62, [1, 16, 256, 513]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_87: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_80, [1, 1, 4096, 513])
        permute_70: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        slice_114: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_70, 1, -256, 9223372036854775807)
        view_86: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_80, [1, 1, 4096, 513])
        permute_69: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        slice_112: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_69, 1, -256, 9223372036854775807);  permute_69 = None
        slice_113: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_112, 3, -257, 9223372036854775807);  slice_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_3: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_16, [1, 3]);  unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_3: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_3, [1, 256, 1, 257]);  rev_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_15: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_6: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_84: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_80, [1, 1, 4096, 513]);  view_80 = None
        permute_66: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        slice_107: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_66, 1, -256, 9223372036854775807);  permute_66 = None
        slice_108: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_107, 3, -257, 9223372036854775807);  slice_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_6: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_15, full_default_6, slice_108);  convert_element_type_15 = full_default_6 = slice_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_11: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_113, where_6);  slice_113 = where_6 = None
        slice_scatter_20: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_114, copy_11, 3, -257, 9223372036854775807);  slice_114 = copy_11 = None
        slice_scatter_21: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_70, slice_scatter_20, 1, -256, 9223372036854775807);  permute_70 = slice_scatter_20 = None
        permute_71: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_21, [0, 2, 1, 3]);  slice_scatter_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_74: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_71, [0, 2, 1, 3]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_2: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_73, permute_74);  permute_73 = permute_74 = None
        permute_75: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_2, [0, 2, 1, 3]);  add_2 = None
        permute_76: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_75, [0, 2, 1, 3]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_6: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None
        convert_element_type_16: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_6, torch.float32);  clone_6 = None
        amax: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_16, [-1], True)
        sub_4: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_16, amax);  convert_element_type_16 = amax = None
        exp: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_1: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_7: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_7: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_7, div_7);  unsqueeze_18 = full_default_7 = div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_17: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_7, torch.float16);  where_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_78: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_17, [0, 2, 1, 3]);  convert_element_type_17 = None
        view_96: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_78, [12, 16, 256, 513]);  permute_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_96, [0, 257], 0.0);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_98: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_3, [12, 16, -1]);  constant_pad_nd_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_117: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_98, 2, 0, -256);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_99: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_117, [12, 16, 256, 769]);  slice_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_118: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_99, 3, 0, -1);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_19: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_118, 4);  slice_118 = None
        view_100: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_19, [192, 256, 768]);  unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_4: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute, [4096, 768]);  permute = None
        permute_3: "f16[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_2: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg6_1, view_4, permute_3);  arg6_1 = view_4 = permute_3 = None
        view_5: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_2, [4096, 1, 768]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_95: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_5, [4096, 1, 12, 64]);  view_5 = None
        permute_77: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_95, [1, 0, 2, 3]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_79: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_77, [0, 2, 1, 3]);  permute_77 = None
        view_97: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_79, [12, 4096, 64]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_97, [0, 0, 256, 256], -1.0);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_8: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_2, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_20: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_8, 4);  as_strided_8 = None
        clone_8: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_20, memory_format = torch.contiguous_format);  unsqueeze_20 = None
        view_101: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_8, [192, 768, 64]);  clone_8 = None
        bmm_1: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
        view_102: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_1, [12, 16, 256, 1, 64]);  bmm_1 = None
        permute_84: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_102, [0, 1, 2, 4, 3]);  view_102 = None
        view_103: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_84, [12, 16, 256, 64]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_104: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_103, [1, 12, 4096, 64]);  view_103 = None
        permute_85: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_86: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_85, [1, 0, 2, 3]);  permute_85 = None
        clone_9: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
        view_105: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_9, [4096, 1, 768]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_87: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_105, [1, 0, 2]);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_87, [4096, 768]);  permute_87 = None
        permute_88: "f16[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_3: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg10_1, view_106, permute_88);  arg10_1 = view_106 = permute_88 = None
        view_107: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 4096, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_4: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_107, arg0_1);  view_107 = arg0_1 = None
        convert_element_type_23: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_23, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 4096, 1]" = var_mean[0]
        getitem_1: "f32[1, 4096, 1]" = var_mean[1];  var_mean = None
        sub_6: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_23, getitem_1);  convert_element_type_23 = getitem_1 = None
        add_5: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_1: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = rsqrt = None
        mul_2: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg11_1);  mul_1 = arg11_1 = None
        add_6: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_2, arg12_1);  mul_2 = arg12_1 = None
        convert_element_type_24: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_6, torch.float16);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_24, [4096, 768])
        permute_89: "f16[768, 3072]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_4: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg14_1, view_108, permute_89);  arg14_1 = view_108 = permute_89 = None
        view_109: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 4096, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_28: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        mul_3: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.5)
        mul_4: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.7071067811865476);  convert_element_type_28 = None
        erf: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_7: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_3, add_7);  mul_3 = add_7 = None
        convert_element_type_29: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_5, torch.float16);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_110: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_29, [4096, 3072]);  convert_element_type_29 = None
        permute_90: "f16[3072, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_5: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg16_1, view_110, permute_90);  arg16_1 = view_110 = permute_90 = None
        view_111: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 4096, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_8: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_111, convert_element_type_24);  view_111 = convert_element_type_24 = None
        convert_element_type_33: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_33, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 4096, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 4096, 1]" = var_mean_1[1];  var_mean_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_16: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_17: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_38: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_39: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 3);  unsqueeze_38 = None
        full_default_15: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_9: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_123: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_9, 1, 0, -1)
        slice_121: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_9, 1, 0, -1)
        slice_122: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_121, 3, 256, 9223372036854775807);  slice_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_7: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_3);  convert_element_type_33 = getitem_3 = None
        add_9: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_6: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_1);  sub_7 = rsqrt_1 = None
        mul_7: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_6, arg17_1);  mul_6 = arg17_1 = None
        add_10: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_7, arg18_1);  mul_7 = arg18_1 = None
        convert_element_type_34: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_10, torch.float16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_91: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_112: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_91, [4096, 768])
        permute_92: "f16[768, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_6: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg20_1, view_112, permute_92);  arg20_1 = view_112 = permute_92 = None
        view_113: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_6, [4096, 1, 768]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_10: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_113, 8.0);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_152: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_10, [4096, 1, 12, 64]);  div_10 = None
        permute_116: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_152, [1, 0, 2, 3]);  view_152 = None
        permute_117: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_116, [0, 2, 1, 3]);  permute_116 = None
        view_153: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_117, [12, 4096, 64]);  permute_117 = None
        view_154: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_153, [12, 8, 512, 64]);  view_153 = None
        as_strided_14: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_154, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_154 = None
        unsqueeze_25: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_14, 4);  as_strided_14 = None
        clone_12: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_25, memory_format = torch.contiguous_format);  unsqueeze_25 = None
        view_155: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_12, [180, 512, 64]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_114: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_91, [4096, 768])
        permute_93: "f16[768, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_7: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg22_1, view_114, permute_93);  arg22_1 = view_114 = permute_93 = None
        view_115: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_7, [4096, 1, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_123: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_115, [4096, 1, 12, 64]);  view_115 = None
        permute_96: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_123, [1, 0, 2, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_101: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_96, [0, 2, 1, 3]);  permute_96 = None
        view_129: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_101, [12, 4096, 64]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_138: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_129, [12, 8, 512, 64]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_10: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_138, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_23: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_10, 4);  as_strided_10 = None
        permute_111: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_23, [0, 1, 4, 2, 3]);  unsqueeze_23 = None
        permute_120: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_111, [0, 1, 4, 3, 2]);  permute_111 = None
        clone_13: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        view_156: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_13, [180, 64, 512]);  clone_13 = None
        bmm_2: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_155, view_156);  view_155 = view_156 = None
        view_157: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_2, [12, 15, 512, 1, 512]);  bmm_2 = None
        permute_121: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_157, [0, 1, 2, 4, 3]);  view_157 = None
        view_158: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_121, [12, 15, 512, 512]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_4: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_158, [0, 0, 0, 1], 0.0);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_159: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_4, [12, 15, 512, 513]);  constant_pad_nd_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_119: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_159, 2, 0, 256)
        slice_120: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_119, 3, 0, 257);  slice_119 = None
        copy_12: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_122, slice_120);  slice_122 = slice_120 = None
        slice_scatter_22: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_123, copy_12, 3, 256, 9223372036854775807);  slice_123 = copy_12 = None
        slice_scatter_23: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_22, 1, 0, -1);  full_9 = slice_scatter_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_30: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_23, 1, -1)
        select_29: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_23, 1, -1)
        slice_129: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_29, 2, 256, 9223372036854775807);  select_29 = None
        select_26: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_159, 1, -1)
        slice_126: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_26, 1, 256, 9223372036854775807);  select_26 = None
        slice_127: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_126, 2, 0, 257);  slice_126 = None
        copy_13: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_129, slice_127);  slice_129 = slice_127 = None
        slice_scatter_24: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_30, copy_13, 2, 256, 9223372036854775807);  select_30 = copy_13 = None
        select_scatter_4: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_23, slice_scatter_24, 1, -1);  slice_scatter_23 = slice_scatter_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_138: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_4, 1, 1, 9223372036854775807)
        slice_136: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_4, 1, 1, 9223372036854775807)
        slice_137: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_136, 3, 0, 256);  slice_136 = None
        slice_131: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_159, 2, -257, -1)
        slice_132: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_131, 3, 257, 9223372036854775807);  slice_131 = None
        copy_14: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_137, slice_132);  slice_137 = slice_132 = None
        slice_scatter_25: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_138, copy_14, 3, 0, 256);  slice_138 = copy_14 = None
        slice_scatter_26: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_4, slice_scatter_25, 1, 1, 9223372036854775807);  select_scatter_4 = slice_scatter_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_37: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_26, 1, 0)
        slice_148: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_37, 1, 1, 256)
        select_36: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_26, 1, 0)
        slice_146: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_36, 1, 1, 256);  select_36 = None
        slice_147: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_146, 2, 1, 256);  slice_146 = None
        select_32: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_159, 1, 0);  view_159 = None
        slice_141: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_32, 1, 0, 255);  select_32 = None
        slice_142: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_141, 2, -255, 9223372036854775807);  slice_141 = None
        copy_15: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_147, slice_142);  slice_147 = slice_142 = None
        slice_scatter_27: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_148, copy_15, 2, 1, 256);  slice_148 = copy_15 = None
        slice_scatter_28: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_37, slice_scatter_27, 1, 1, 256);  select_37 = slice_scatter_27 = None
        select_scatter_5: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_26, slice_scatter_28, 1, 0);  slice_scatter_26 = slice_scatter_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_167: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_5, [1, 12, 4096, 513])
        permute_129: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
        slice_161: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_129, 1, 0, 256)
        view_166: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_5, [1, 12, 4096, 513])
        permute_128: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        slice_159: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_128, 1, 0, 256);  permute_128 = None
        slice_160: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_159, 3, 0, 257);  slice_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_4: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_26: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_4, -2);  iota_4 = None
        iota_5: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_27: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
        sub_9: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_26, unsqueeze_27);  unsqueeze_26 = unsqueeze_27 = None
        le_2: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_9, 0);  sub_9 = None
        full_default_8: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f16[256, 257]" = torch.ops.aten.where.self(le_2, full_default_8, full_default_9);  le_2 = full_default_8 = full_default_9 = None
        rev_4: "f16[256, 257]" = torch.ops.prims.rev.default(where_8, [0]);  where_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_28: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_4, 0);  rev_4 = None
        unsqueeze_29: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 2);  unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_4: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_29, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_46: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_4, torch.bool);  expand_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_11: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_126: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_11, [0, 2, 1, 3]);  full_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_164: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_5, [1, 12, 4096, 513]);  select_scatter_5 = None
        permute_125: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        slice_154: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_125, 1, 0, 256);  permute_125 = None
        slice_155: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_154, 3, 0, 257);  slice_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_9: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_46, permute_126, slice_155);  convert_element_type_46 = permute_126 = slice_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_16: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_160, where_9);  slice_160 = where_9 = None
        slice_scatter_29: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_161, copy_16, 3, 0, 257);  slice_161 = copy_16 = None
        slice_scatter_30: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_129, slice_scatter_29, 1, 0, 256);  permute_129 = slice_scatter_29 = None
        permute_130: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_30, [0, 2, 1, 3]);  slice_scatter_30 = None
        view_168: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_130, [12, 16, 256, 513]);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_175: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_168, [1, 12, 4096, 513])
        permute_138: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
        slice_174: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_138, 1, -256, 9223372036854775807)
        view_174: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_168, [1, 12, 4096, 513])
        permute_137: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        slice_172: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_137, 1, -256, 9223372036854775807);  permute_137 = None
        slice_173: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_172, 3, -257, 9223372036854775807);  slice_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_5: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_29, [1, 3]);  unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_5: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_5, [1, 256, 12, 257]);  rev_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_47: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_5, torch.bool);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_12: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_135: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_12, [0, 2, 1, 3]);  full_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_172: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_168, [1, 12, 4096, 513]);  view_168 = None
        permute_134: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        slice_167: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_134, 1, -256, 9223372036854775807);  permute_134 = None
        slice_168: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_167, 3, -257, 9223372036854775807);  slice_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_10: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_47, permute_135, slice_168);  convert_element_type_47 = permute_135 = slice_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_17: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_173, where_10);  slice_173 = where_10 = None
        slice_scatter_31: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_174, copy_17, 3, -257, 9223372036854775807);  slice_174 = copy_17 = None
        slice_scatter_32: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_138, slice_scatter_31, 1, -256, 9223372036854775807);  permute_138 = slice_scatter_31 = None
        permute_139: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_32, [0, 2, 1, 3]);  slice_scatter_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_164: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_139, [0, 2, 1, 3]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_14: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_181: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_14, 1, 0, -1)
        slice_179: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_14, 1, 0, -1)
        slice_180: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_179, 3, 256, 9223372036854775807);  slice_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_13: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_141: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_13, [0, 2, 1, 3]);  full_13 = None
        view_178: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_141, [1, 4096, 1]);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_180: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_178, [1, 8, 512, 1]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_15: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_180, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_32: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_15, 4);  as_strided_15 = None
        permute_143: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_32, [0, 1, 2, 4, 3]);  unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_10: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_1: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_30: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_1, 2);  ne_1 = None
        unsqueeze_31: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 3);  unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_48: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_31, torch.float16)
        where_11: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_31, full_default_10, convert_element_type_48);  unsqueeze_31 = full_default_10 = convert_element_type_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_142: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_11, [0, 2, 1, 3]);  where_11 = None
        view_179: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_142, [1, 4096, 1]);  permute_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_181: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_179, [1, 8, 512, 1]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_16: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_181, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_33: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_16, 4);  as_strided_16 = None
        permute_144: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_33, [0, 1, 4, 2, 3]);  unsqueeze_33 = None
        mul_8: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_143, permute_144);  permute_143 = permute_144 = None
        view_182: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_8, [1, 15, 512, 512]);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_5: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_182, [0, 0, 0, 1], 0.0);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_183: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_5, [1, 15, 512, 513]);  constant_pad_nd_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_177: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_183, 2, 0, 256)
        slice_178: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_177, 3, 0, 257);  slice_177 = None
        copy_18: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_180, slice_178);  slice_180 = slice_178 = None
        slice_scatter_33: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_181, copy_18, 3, 256, 9223372036854775807);  slice_181 = copy_18 = None
        slice_scatter_34: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_14, slice_scatter_33, 1, 0, -1);  full_14 = slice_scatter_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_43: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_34, 1, -1)
        select_42: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_34, 1, -1)
        slice_187: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_42, 2, 256, 9223372036854775807);  select_42 = None
        select_39: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_183, 1, -1)
        slice_184: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_39, 1, 256, 9223372036854775807);  select_39 = None
        slice_185: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_184, 2, 0, 257);  slice_184 = None
        copy_19: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_187, slice_185);  slice_187 = slice_185 = None
        slice_scatter_35: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_43, copy_19, 2, 256, 9223372036854775807);  select_43 = copy_19 = None
        select_scatter_6: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_34, slice_scatter_35, 1, -1);  slice_scatter_34 = slice_scatter_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_196: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_6, 1, 1, 9223372036854775807)
        slice_194: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_6, 1, 1, 9223372036854775807)
        slice_195: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_194, 3, 0, 256);  slice_194 = None
        slice_189: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_183, 2, -257, -1)
        slice_190: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_189, 3, 257, 9223372036854775807);  slice_189 = None
        copy_20: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_195, slice_190);  slice_195 = slice_190 = None
        slice_scatter_36: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_196, copy_20, 3, 0, 256);  slice_196 = copy_20 = None
        slice_scatter_37: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_6, slice_scatter_36, 1, 1, 9223372036854775807);  select_scatter_6 = slice_scatter_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_50: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_37, 1, 0)
        slice_206: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_50, 1, 1, 256)
        select_49: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_37, 1, 0)
        slice_204: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_49, 1, 1, 256);  select_49 = None
        slice_205: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_204, 2, 1, 256);  slice_204 = None
        select_45: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_183, 1, 0);  view_183 = None
        slice_199: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_45, 1, 0, 255);  select_45 = None
        slice_200: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_199, 2, -255, 9223372036854775807);  slice_199 = None
        copy_21: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_205, slice_200);  slice_205 = slice_200 = None
        slice_scatter_38: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_206, copy_21, 2, 1, 256);  slice_206 = copy_21 = None
        slice_scatter_39: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_50, slice_scatter_38, 1, 1, 256);  select_50 = slice_scatter_38 = None
        select_scatter_7: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_37, slice_scatter_39, 1, 0);  slice_scatter_37 = slice_scatter_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_191: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_7, [1, 1, 4096, 513])
        permute_152: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        slice_219: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_152, 1, 0, 256)
        view_190: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_7, [1, 1, 4096, 513])
        permute_151: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        slice_217: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_151, 1, 0, 256);  permute_151 = None
        slice_218: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_217, 3, 0, 257);  slice_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_6: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_34: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_6, -2);  iota_6 = None
        iota_7: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_35: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        sub_11: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_34, unsqueeze_35);  unsqueeze_34 = unsqueeze_35 = None
        le_3: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_11, 0);  sub_11 = None
        full_default_11: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f16[256, 257]" = torch.ops.aten.where.self(le_3, full_default_11, full_default_12);  le_3 = full_default_11 = full_default_12 = None
        rev_6: "f16[256, 257]" = torch.ops.prims.rev.default(where_12, [0]);  where_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_36: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_6, 0);  rev_6 = None
        unsqueeze_37: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_6: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_37, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_49: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_6, torch.bool);  expand_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_13: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_188: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_7, [1, 1, 4096, 513]);  select_scatter_7 = None
        permute_148: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        slice_212: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_148, 1, 0, 256);  permute_148 = None
        slice_213: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_212, 3, 0, 257);  slice_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_13: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_49, full_default_13, slice_213);  convert_element_type_49 = full_default_13 = slice_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_22: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_218, where_13);  slice_218 = where_13 = None
        slice_scatter_40: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_219, copy_22, 3, 0, 257);  slice_219 = copy_22 = None
        slice_scatter_41: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_152, slice_scatter_40, 1, 0, 256);  permute_152 = slice_scatter_40 = None
        permute_153: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_41, [0, 2, 1, 3]);  slice_scatter_41 = None
        view_192: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_153, [1, 16, 256, 513]);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_199: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_192, [1, 1, 4096, 513])
        permute_161: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
        slice_232: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_161, 1, -256, 9223372036854775807)
        view_198: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_192, [1, 1, 4096, 513])
        permute_160: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
        slice_230: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_160, 1, -256, 9223372036854775807);  permute_160 = None
        slice_231: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_230, 3, -257, 9223372036854775807);  slice_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_7: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_37, [1, 3]);  unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_7: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_7, [1, 256, 1, 257]);  rev_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_50: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_7, torch.bool);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_14: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_196: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_192, [1, 1, 4096, 513]);  view_192 = None
        permute_157: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
        slice_225: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_157, 1, -256, 9223372036854775807);  permute_157 = None
        slice_226: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_225, 3, -257, 9223372036854775807);  slice_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_14: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_50, full_default_14, slice_226);  convert_element_type_50 = full_default_14 = slice_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_23: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_231, where_14);  slice_231 = where_14 = None
        slice_scatter_42: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_232, copy_23, 3, -257, 9223372036854775807);  slice_232 = copy_23 = None
        slice_scatter_43: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_161, slice_scatter_42, 1, -256, 9223372036854775807);  permute_161 = slice_scatter_42 = None
        permute_162: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_43, [0, 2, 1, 3]);  slice_scatter_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_165: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_162, [0, 2, 1, 3]);  permute_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_13: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_164, permute_165);  permute_164 = permute_165 = None
        permute_166: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_13, [0, 2, 1, 3]);  add_13 = None
        permute_167: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_166, [0, 2, 1, 3]);  permute_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_18: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        convert_element_type_51: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_18, torch.float32);  clone_18 = None
        amax_1: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_51, [-1], True)
        sub_12: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_51, amax_1);  convert_element_type_51 = amax_1 = None
        exp_1: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_2: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_17: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_15: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_39, full_default_15, div_17);  unsqueeze_39 = full_default_15 = div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_52: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_15, torch.float16);  where_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_169: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_52, [0, 2, 1, 3]);  convert_element_type_52 = None
        view_208: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_169, [12, 16, 256, 513]);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_7: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_208, [0, 257], 0.0);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_210: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_7, [12, 16, -1]);  constant_pad_nd_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_235: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_210, 2, 0, -256);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_211: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_235, [12, 16, 256, 769]);  slice_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_236: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_211, 3, 0, -1);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_40: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_236, 4);  slice_236 = None
        view_212: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_40, [192, 256, 768]);  unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_116: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_91, [4096, 768]);  permute_91 = None
        permute_94: "f16[768, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_8: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg24_1, view_116, permute_94);  arg24_1 = view_116 = permute_94 = None
        view_117: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_8, [4096, 1, 768]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_207: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_117, [4096, 1, 12, 64]);  view_117 = None
        permute_168: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_207, [1, 0, 2, 3]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_170: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_168, [0, 2, 1, 3]);  permute_168 = None
        view_209: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_170, [12, 4096, 64]);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_6: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_209, [0, 0, 256, 256], -1.0);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_17: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_6, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_41: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_17, 4);  as_strided_17 = None
        clone_20: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_41, memory_format = torch.contiguous_format);  unsqueeze_41 = None
        view_213: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_20, [192, 768, 64]);  clone_20 = None
        bmm_3: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_212, view_213);  view_212 = view_213 = None
        view_214: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_3, [12, 16, 256, 1, 64]);  bmm_3 = None
        permute_175: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_214, [0, 1, 2, 4, 3]);  view_214 = None
        view_215: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_175, [12, 16, 256, 64]);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_216: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_215, [1, 12, 4096, 64]);  view_215 = None
        permute_176: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_177: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_176, [1, 0, 2, 3]);  permute_176 = None
        clone_21: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_217: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_21, [4096, 1, 768]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_178: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_217, [1, 0, 2]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_178, [4096, 768]);  permute_178 = None
        permute_179: "f16[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_9: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg26_1, view_218, permute_179);  arg26_1 = view_218 = permute_179 = None
        view_219: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 4096, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_219, convert_element_type_34);  view_219 = convert_element_type_34 = None
        convert_element_type_58: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_58, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 4096, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 4096, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_14: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_58, getitem_5);  convert_element_type_58 = getitem_5 = None
        add_16: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_9: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_2);  sub_14 = rsqrt_2 = None
        mul_10: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg27_1);  mul_9 = arg27_1 = None
        add_17: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_10, arg28_1);  mul_10 = arg28_1 = None
        convert_element_type_59: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_17, torch.float16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_220: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_59, [4096, 768])
        permute_180: "f16[768, 3072]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_10: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg30_1, view_220, permute_180);  arg30_1 = view_220 = permute_180 = None
        view_221: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 4096, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_63: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_221, torch.float32);  view_221 = None
        mul_11: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 0.5)
        mul_12: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 0.7071067811865476);  convert_element_type_63 = None
        erf_1: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_18: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_18);  mul_11 = add_18 = None
        convert_element_type_64: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_13, torch.float16);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_222: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_64, [4096, 3072]);  convert_element_type_64 = None
        permute_181: "f16[3072, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_11: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg32_1, view_222, permute_181);  arg32_1 = view_222 = permute_181 = None
        view_223: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 4096, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_19: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_223, convert_element_type_59);  view_223 = convert_element_type_59 = None
        convert_element_type_68: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32);  add_19 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_68, [2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 4096, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 4096, 1]" = var_mean_3[1];  var_mean_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_25: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_26: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_59: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_60: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 3);  unsqueeze_59 = None
        full_default_23: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_18: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_241: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_18, 1, 0, -1)
        slice_239: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_18, 1, 0, -1)
        slice_240: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_239, 3, 256, 9223372036854775807);  slice_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_15: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_68, getitem_7);  convert_element_type_68 = getitem_7 = None
        add_20: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_14: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_3);  sub_15 = rsqrt_3 = None
        mul_15: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg33_1);  mul_14 = arg33_1 = None
        add_21: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_15, arg34_1);  mul_15 = arg34_1 = None
        convert_element_type_69: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_21, torch.float16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_182: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_69, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_224: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_182, [4096, 768])
        permute_183: "f16[768, 768]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_12: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg36_1, view_224, permute_183);  arg36_1 = view_224 = permute_183 = None
        view_225: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_12, [4096, 1, 768]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_20: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_225, 8.0);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_264: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_20, [4096, 1, 12, 64]);  div_20 = None
        permute_207: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_264, [1, 0, 2, 3]);  view_264 = None
        permute_208: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_207, [0, 2, 1, 3]);  permute_207 = None
        view_265: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_208, [12, 4096, 64]);  permute_208 = None
        view_266: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_265, [12, 8, 512, 64]);  view_265 = None
        as_strided_23: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_266, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_266 = None
        unsqueeze_46: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_23, 4);  as_strided_23 = None
        clone_24: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_46, memory_format = torch.contiguous_format);  unsqueeze_46 = None
        view_267: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_24, [180, 512, 64]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_226: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_182, [4096, 768])
        permute_184: "f16[768, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_13: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg38_1, view_226, permute_184);  arg38_1 = view_226 = permute_184 = None
        view_227: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_13, [4096, 1, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_235: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_227, [4096, 1, 12, 64]);  view_227 = None
        permute_187: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_235, [1, 0, 2, 3]);  view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_192: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_187, [0, 2, 1, 3]);  permute_187 = None
        view_241: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_192, [12, 4096, 64]);  permute_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_250: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_241, [12, 8, 512, 64]);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_19: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_250, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_44: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_19, 4);  as_strided_19 = None
        permute_202: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_44, [0, 1, 4, 2, 3]);  unsqueeze_44 = None
        permute_211: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_202, [0, 1, 4, 3, 2]);  permute_202 = None
        clone_25: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
        view_268: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_25, [180, 64, 512]);  clone_25 = None
        bmm_4: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_267, view_268);  view_267 = view_268 = None
        view_269: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_4, [12, 15, 512, 1, 512]);  bmm_4 = None
        permute_212: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_269, [0, 1, 2, 4, 3]);  view_269 = None
        view_270: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_212, [12, 15, 512, 512]);  permute_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_8: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_270, [0, 0, 0, 1], 0.0);  view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_271: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_8, [12, 15, 512, 513]);  constant_pad_nd_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_237: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_271, 2, 0, 256)
        slice_238: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_237, 3, 0, 257);  slice_237 = None
        copy_24: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_240, slice_238);  slice_240 = slice_238 = None
        slice_scatter_44: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_241, copy_24, 3, 256, 9223372036854775807);  slice_241 = copy_24 = None
        slice_scatter_45: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_18, slice_scatter_44, 1, 0, -1);  full_18 = slice_scatter_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_56: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_45, 1, -1)
        select_55: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_45, 1, -1)
        slice_247: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_55, 2, 256, 9223372036854775807);  select_55 = None
        select_52: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_271, 1, -1)
        slice_244: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_52, 1, 256, 9223372036854775807);  select_52 = None
        slice_245: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_244, 2, 0, 257);  slice_244 = None
        copy_25: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_247, slice_245);  slice_247 = slice_245 = None
        slice_scatter_46: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_56, copy_25, 2, 256, 9223372036854775807);  select_56 = copy_25 = None
        select_scatter_8: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_45, slice_scatter_46, 1, -1);  slice_scatter_45 = slice_scatter_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_256: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_8, 1, 1, 9223372036854775807)
        slice_254: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_8, 1, 1, 9223372036854775807)
        slice_255: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_254, 3, 0, 256);  slice_254 = None
        slice_249: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_271, 2, -257, -1)
        slice_250: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_249, 3, 257, 9223372036854775807);  slice_249 = None
        copy_26: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_255, slice_250);  slice_255 = slice_250 = None
        slice_scatter_47: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_256, copy_26, 3, 0, 256);  slice_256 = copy_26 = None
        slice_scatter_48: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_8, slice_scatter_47, 1, 1, 9223372036854775807);  select_scatter_8 = slice_scatter_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_63: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_48, 1, 0)
        slice_266: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_63, 1, 1, 256)
        select_62: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_48, 1, 0)
        slice_264: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_62, 1, 1, 256);  select_62 = None
        slice_265: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_264, 2, 1, 256);  slice_264 = None
        select_58: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_271, 1, 0);  view_271 = None
        slice_259: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_58, 1, 0, 255);  select_58 = None
        slice_260: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_259, 2, -255, 9223372036854775807);  slice_259 = None
        copy_27: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_265, slice_260);  slice_265 = slice_260 = None
        slice_scatter_49: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_266, copy_27, 2, 1, 256);  slice_266 = copy_27 = None
        slice_scatter_50: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_63, slice_scatter_49, 1, 1, 256);  select_63 = slice_scatter_49 = None
        select_scatter_9: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_48, slice_scatter_50, 1, 0);  slice_scatter_48 = slice_scatter_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_279: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_9, [1, 12, 4096, 513])
        permute_220: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
        slice_279: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_220, 1, 0, 256)
        view_278: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_9, [1, 12, 4096, 513])
        permute_219: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
        slice_277: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_219, 1, 0, 256);  permute_219 = None
        slice_278: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_277, 3, 0, 257);  slice_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_8: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_47: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_8, -2);  iota_8 = None
        iota_9: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_48: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
        sub_17: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_47, unsqueeze_48);  unsqueeze_47 = unsqueeze_48 = None
        le_4: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_17, 0);  sub_17 = None
        full_default_16: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f16[256, 257]" = torch.ops.aten.where.self(le_4, full_default_16, full_default_17);  le_4 = full_default_16 = full_default_17 = None
        rev_8: "f16[256, 257]" = torch.ops.prims.rev.default(where_16, [0]);  where_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_49: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_8, 0);  rev_8 = None
        unsqueeze_50: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 2);  unsqueeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_8: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_50, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_81: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_8, torch.bool);  expand_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_20: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_217: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_20, [0, 2, 1, 3]);  full_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_276: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_9, [1, 12, 4096, 513]);  select_scatter_9 = None
        permute_216: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_276, [0, 2, 1, 3]);  view_276 = None
        slice_272: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_216, 1, 0, 256);  permute_216 = None
        slice_273: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_272, 3, 0, 257);  slice_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_17: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_81, permute_217, slice_273);  convert_element_type_81 = permute_217 = slice_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_28: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_278, where_17);  slice_278 = where_17 = None
        slice_scatter_51: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_279, copy_28, 3, 0, 257);  slice_279 = copy_28 = None
        slice_scatter_52: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_220, slice_scatter_51, 1, 0, 256);  permute_220 = slice_scatter_51 = None
        permute_221: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_52, [0, 2, 1, 3]);  slice_scatter_52 = None
        view_280: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_221, [12, 16, 256, 513]);  permute_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_287: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_280, [1, 12, 4096, 513])
        permute_229: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None
        slice_292: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_229, 1, -256, 9223372036854775807)
        view_286: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_280, [1, 12, 4096, 513])
        permute_228: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
        slice_290: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_228, 1, -256, 9223372036854775807);  permute_228 = None
        slice_291: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_290, 3, -257, 9223372036854775807);  slice_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_9: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_50, [1, 3]);  unsqueeze_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_9: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_9, [1, 256, 12, 257]);  rev_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_82: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_9, torch.bool);  expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_21: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_226: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_21, [0, 2, 1, 3]);  full_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_284: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_280, [1, 12, 4096, 513]);  view_280 = None
        permute_225: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
        slice_285: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_225, 1, -256, 9223372036854775807);  permute_225 = None
        slice_286: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_285, 3, -257, 9223372036854775807);  slice_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_18: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_82, permute_226, slice_286);  convert_element_type_82 = permute_226 = slice_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_29: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_291, where_18);  slice_291 = where_18 = None
        slice_scatter_53: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_292, copy_29, 3, -257, 9223372036854775807);  slice_292 = copy_29 = None
        slice_scatter_54: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_229, slice_scatter_53, 1, -256, 9223372036854775807);  permute_229 = slice_scatter_53 = None
        permute_230: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_54, [0, 2, 1, 3]);  slice_scatter_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_255: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_230, [0, 2, 1, 3]);  permute_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_23: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_299: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1)
        slice_297: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1)
        slice_298: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_297, 3, 256, 9223372036854775807);  slice_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_22: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_232: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_22, [0, 2, 1, 3]);  full_22 = None
        view_290: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_232, [1, 4096, 1]);  permute_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_292: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_290, [1, 8, 512, 1]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_24: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_292, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_53: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_24, 4);  as_strided_24 = None
        permute_234: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_53, [0, 1, 2, 4, 3]);  unsqueeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_18: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_2: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_51: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_2, 2);  ne_2 = None
        unsqueeze_52: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 3);  unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_83: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_52, torch.float16)
        where_19: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_52, full_default_18, convert_element_type_83);  unsqueeze_52 = full_default_18 = convert_element_type_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_233: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_19, [0, 2, 1, 3]);  where_19 = None
        view_291: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_233, [1, 4096, 1]);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_293: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_291, [1, 8, 512, 1]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_25: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_293, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_54: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_25, 4);  as_strided_25 = None
        permute_235: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_54, [0, 1, 4, 2, 3]);  unsqueeze_54 = None
        mul_16: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_234, permute_235);  permute_234 = permute_235 = None
        view_294: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_16, [1, 15, 512, 512]);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_9: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_294, [0, 0, 0, 1], 0.0);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_295: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_9, [1, 15, 512, 513]);  constant_pad_nd_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_295: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_295, 2, 0, 256)
        slice_296: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_295, 3, 0, 257);  slice_295 = None
        copy_30: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_298, slice_296);  slice_298 = slice_296 = None
        slice_scatter_55: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_299, copy_30, 3, 256, 9223372036854775807);  slice_299 = copy_30 = None
        slice_scatter_56: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_23, slice_scatter_55, 1, 0, -1);  full_23 = slice_scatter_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_69: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_56, 1, -1)
        select_68: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_56, 1, -1)
        slice_305: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_68, 2, 256, 9223372036854775807);  select_68 = None
        select_65: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_295, 1, -1)
        slice_302: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_65, 1, 256, 9223372036854775807);  select_65 = None
        slice_303: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_302, 2, 0, 257);  slice_302 = None
        copy_31: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_305, slice_303);  slice_305 = slice_303 = None
        slice_scatter_57: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_69, copy_31, 2, 256, 9223372036854775807);  select_69 = copy_31 = None
        select_scatter_10: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_56, slice_scatter_57, 1, -1);  slice_scatter_56 = slice_scatter_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_314: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_10, 1, 1, 9223372036854775807)
        slice_312: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_10, 1, 1, 9223372036854775807)
        slice_313: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_312, 3, 0, 256);  slice_312 = None
        slice_307: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_295, 2, -257, -1)
        slice_308: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_307, 3, 257, 9223372036854775807);  slice_307 = None
        copy_32: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_313, slice_308);  slice_313 = slice_308 = None
        slice_scatter_58: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_314, copy_32, 3, 0, 256);  slice_314 = copy_32 = None
        slice_scatter_59: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_10, slice_scatter_58, 1, 1, 9223372036854775807);  select_scatter_10 = slice_scatter_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_76: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_59, 1, 0)
        slice_324: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_76, 1, 1, 256)
        select_75: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_59, 1, 0)
        slice_322: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_75, 1, 1, 256);  select_75 = None
        slice_323: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_322, 2, 1, 256);  slice_322 = None
        select_71: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_295, 1, 0);  view_295 = None
        slice_317: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_71, 1, 0, 255);  select_71 = None
        slice_318: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_317, 2, -255, 9223372036854775807);  slice_317 = None
        copy_33: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_323, slice_318);  slice_323 = slice_318 = None
        slice_scatter_60: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_324, copy_33, 2, 1, 256);  slice_324 = copy_33 = None
        slice_scatter_61: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_76, slice_scatter_60, 1, 1, 256);  select_76 = slice_scatter_60 = None
        select_scatter_11: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_59, slice_scatter_61, 1, 0);  slice_scatter_59 = slice_scatter_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_303: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_11, [1, 1, 4096, 513])
        permute_243: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        slice_337: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_243, 1, 0, 256)
        view_302: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_11, [1, 1, 4096, 513])
        permute_242: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        slice_335: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_242, 1, 0, 256);  permute_242 = None
        slice_336: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_335, 3, 0, 257);  slice_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_10: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_55: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_10, -2);  iota_10 = None
        iota_11: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_56: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
        sub_19: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_55, unsqueeze_56);  unsqueeze_55 = unsqueeze_56 = None
        le_5: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_19, 0);  sub_19 = None
        full_default_19: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f16[256, 257]" = torch.ops.aten.where.self(le_5, full_default_19, full_default_20);  le_5 = full_default_19 = full_default_20 = None
        rev_10: "f16[256, 257]" = torch.ops.prims.rev.default(where_20, [0]);  where_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_57: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_10, 0);  rev_10 = None
        unsqueeze_58: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_10: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_58, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_84: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_10, torch.bool);  expand_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_21: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_300: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_11, [1, 1, 4096, 513]);  select_scatter_11 = None
        permute_239: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        slice_330: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_239, 1, 0, 256);  permute_239 = None
        slice_331: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_330, 3, 0, 257);  slice_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_21: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_84, full_default_21, slice_331);  convert_element_type_84 = full_default_21 = slice_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_34: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_336, where_21);  slice_336 = where_21 = None
        slice_scatter_62: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_337, copy_34, 3, 0, 257);  slice_337 = copy_34 = None
        slice_scatter_63: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_243, slice_scatter_62, 1, 0, 256);  permute_243 = slice_scatter_62 = None
        permute_244: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_63, [0, 2, 1, 3]);  slice_scatter_63 = None
        view_304: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_244, [1, 16, 256, 513]);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_311: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_304, [1, 1, 4096, 513])
        permute_252: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None
        slice_350: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_252, 1, -256, 9223372036854775807)
        view_310: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_304, [1, 1, 4096, 513])
        permute_251: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
        slice_348: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_251, 1, -256, 9223372036854775807);  permute_251 = None
        slice_349: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_348, 3, -257, 9223372036854775807);  slice_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_11: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_58, [1, 3]);  unsqueeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_11: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_11, [1, 256, 1, 257]);  rev_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_85: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_11, torch.bool);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_22: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_308: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_304, [1, 1, 4096, 513]);  view_304 = None
        permute_248: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
        slice_343: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_248, 1, -256, 9223372036854775807);  permute_248 = None
        slice_344: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_343, 3, -257, 9223372036854775807);  slice_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_22: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_85, full_default_22, slice_344);  convert_element_type_85 = full_default_22 = slice_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_35: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_349, where_22);  slice_349 = where_22 = None
        slice_scatter_64: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_350, copy_35, 3, -257, 9223372036854775807);  slice_350 = copy_35 = None
        slice_scatter_65: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_252, slice_scatter_64, 1, -256, 9223372036854775807);  permute_252 = slice_scatter_64 = None
        permute_253: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_65, [0, 2, 1, 3]);  slice_scatter_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_256: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_253, [0, 2, 1, 3]);  permute_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_24: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_255, permute_256);  permute_255 = permute_256 = None
        permute_257: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_24, [0, 2, 1, 3]);  add_24 = None
        permute_258: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_257, [0, 2, 1, 3]);  permute_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_30: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        convert_element_type_86: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_30, torch.float32);  clone_30 = None
        amax_2: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_86, [-1], True)
        sub_20: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_86, amax_2);  convert_element_type_86 = amax_2 = None
        exp_2: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_3: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_27: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_23: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_60, full_default_23, div_27);  unsqueeze_60 = full_default_23 = div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_87: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_23, torch.float16);  where_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_260: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_87, [0, 2, 1, 3]);  convert_element_type_87 = None
        view_320: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_260, [12, 16, 256, 513]);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_11: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_320, [0, 257], 0.0);  view_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_322: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_11, [12, 16, -1]);  constant_pad_nd_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_353: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_322, 2, 0, -256);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_323: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_353, [12, 16, 256, 769]);  slice_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_354: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_323, 3, 0, -1);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_61: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_354, 4);  slice_354 = None
        view_324: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_61, [192, 256, 768]);  unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_228: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_182, [4096, 768]);  permute_182 = None
        permute_185: "f16[768, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_14: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg40_1, view_228, permute_185);  arg40_1 = view_228 = permute_185 = None
        view_229: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_14, [4096, 1, 768]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_319: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_229, [4096, 1, 12, 64]);  view_229 = None
        permute_259: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_319, [1, 0, 2, 3]);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_261: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_259, [0, 2, 1, 3]);  permute_259 = None
        view_321: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_261, [12, 4096, 64]);  permute_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_10: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_321, [0, 0, 256, 256], -1.0);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_26: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_10, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_62: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_26, 4);  as_strided_26 = None
        clone_32: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_62, memory_format = torch.contiguous_format);  unsqueeze_62 = None
        view_325: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_32, [192, 768, 64]);  clone_32 = None
        bmm_5: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_324, view_325);  view_324 = view_325 = None
        view_326: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_5, [12, 16, 256, 1, 64]);  bmm_5 = None
        permute_266: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_326, [0, 1, 2, 4, 3]);  view_326 = None
        view_327: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_266, [12, 16, 256, 64]);  permute_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_328: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_327, [1, 12, 4096, 64]);  view_327 = None
        permute_267: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_268: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_267, [1, 0, 2, 3]);  permute_267 = None
        clone_33: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
        view_329: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_33, [4096, 1, 768]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_269: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_329, [1, 0, 2]);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_330: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_269, [4096, 768]);  permute_269 = None
        permute_270: "f16[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_15: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg42_1, view_330, permute_270);  arg42_1 = view_330 = permute_270 = None
        view_331: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 4096, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_26: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_331, convert_element_type_69);  view_331 = convert_element_type_69 = None
        convert_element_type_93: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_26, torch.float32);  add_26 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_93, [2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 4096, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 4096, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_22: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_93, getitem_9);  convert_element_type_93 = getitem_9 = None
        add_27: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_17: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_4);  sub_22 = rsqrt_4 = None
        mul_18: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg43_1);  mul_17 = arg43_1 = None
        add_28: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_18, arg44_1);  mul_18 = arg44_1 = None
        convert_element_type_94: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_28, torch.float16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_332: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_94, [4096, 768])
        permute_271: "f16[768, 3072]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_16: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg46_1, view_332, permute_271);  arg46_1 = view_332 = permute_271 = None
        view_333: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 4096, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_98: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_333, torch.float32);  view_333 = None
        mul_19: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_98, 0.5)
        mul_20: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_98, 0.7071067811865476);  convert_element_type_98 = None
        erf_2: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_29: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_29);  mul_19 = add_29 = None
        convert_element_type_99: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_21, torch.float16);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_334: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_99, [4096, 3072]);  convert_element_type_99 = None
        permute_272: "f16[3072, 768]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_17: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg48_1, view_334, permute_272);  arg48_1 = view_334 = permute_272 = None
        view_335: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 4096, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_30: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_335, convert_element_type_94);  view_335 = convert_element_type_94 = None
        convert_element_type_103: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_103, [2], correction = 0, keepdim = True)
        getitem_10: "f32[1, 4096, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 4096, 1]" = var_mean_5[1];  var_mean_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_34: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_35: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_80: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_81: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 3);  unsqueeze_80 = None
        full_default_31: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_27: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_359: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1)
        slice_357: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1)
        slice_358: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_357, 3, 256, 9223372036854775807);  slice_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_23: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_103, getitem_11);  convert_element_type_103 = getitem_11 = None
        add_31: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_22: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_5);  sub_23 = rsqrt_5 = None
        mul_23: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg49_1);  mul_22 = arg49_1 = None
        add_32: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_23, arg50_1);  mul_23 = arg50_1 = None
        convert_element_type_104: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_32, torch.float16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_273: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_104, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_336: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_273, [4096, 768])
        permute_274: "f16[768, 768]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_18: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg52_1, view_336, permute_274);  arg52_1 = view_336 = permute_274 = None
        view_337: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_18, [4096, 1, 768]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_30: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_337, 8.0);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_376: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_30, [4096, 1, 12, 64]);  div_30 = None
        permute_298: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_376, [1, 0, 2, 3]);  view_376 = None
        permute_299: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_298, [0, 2, 1, 3]);  permute_298 = None
        view_377: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_299, [12, 4096, 64]);  permute_299 = None
        view_378: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_377, [12, 8, 512, 64]);  view_377 = None
        as_strided_32: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_378, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_378 = None
        unsqueeze_67: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_32, 4);  as_strided_32 = None
        clone_36: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_67, memory_format = torch.contiguous_format);  unsqueeze_67 = None
        view_379: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_36, [180, 512, 64]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_338: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_273, [4096, 768])
        permute_275: "f16[768, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_19: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg54_1, view_338, permute_275);  arg54_1 = view_338 = permute_275 = None
        view_339: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_19, [4096, 1, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_347: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_339, [4096, 1, 12, 64]);  view_339 = None
        permute_278: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_347, [1, 0, 2, 3]);  view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_283: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_278, [0, 2, 1, 3]);  permute_278 = None
        view_353: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_283, [12, 4096, 64]);  permute_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_362: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_353, [12, 8, 512, 64]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_28: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_362, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_65: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_28, 4);  as_strided_28 = None
        permute_293: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_65, [0, 1, 4, 2, 3]);  unsqueeze_65 = None
        permute_302: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_293, [0, 1, 4, 3, 2]);  permute_293 = None
        clone_37: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_380: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_37, [180, 64, 512]);  clone_37 = None
        bmm_6: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_379, view_380);  view_379 = view_380 = None
        view_381: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_6, [12, 15, 512, 1, 512]);  bmm_6 = None
        permute_303: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_381, [0, 1, 2, 4, 3]);  view_381 = None
        view_382: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_303, [12, 15, 512, 512]);  permute_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_12: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_382, [0, 0, 0, 1], 0.0);  view_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_383: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_12, [12, 15, 512, 513]);  constant_pad_nd_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_355: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_383, 2, 0, 256)
        slice_356: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_355, 3, 0, 257);  slice_355 = None
        copy_36: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_358, slice_356);  slice_358 = slice_356 = None
        slice_scatter_66: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_359, copy_36, 3, 256, 9223372036854775807);  slice_359 = copy_36 = None
        slice_scatter_67: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_27, slice_scatter_66, 1, 0, -1);  full_27 = slice_scatter_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_82: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_67, 1, -1)
        select_81: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_67, 1, -1)
        slice_365: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_81, 2, 256, 9223372036854775807);  select_81 = None
        select_78: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_383, 1, -1)
        slice_362: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_78, 1, 256, 9223372036854775807);  select_78 = None
        slice_363: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_362, 2, 0, 257);  slice_362 = None
        copy_37: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_365, slice_363);  slice_365 = slice_363 = None
        slice_scatter_68: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_82, copy_37, 2, 256, 9223372036854775807);  select_82 = copy_37 = None
        select_scatter_12: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_67, slice_scatter_68, 1, -1);  slice_scatter_67 = slice_scatter_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_374: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_12, 1, 1, 9223372036854775807)
        slice_372: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_12, 1, 1, 9223372036854775807)
        slice_373: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_372, 3, 0, 256);  slice_372 = None
        slice_367: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_383, 2, -257, -1)
        slice_368: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_367, 3, 257, 9223372036854775807);  slice_367 = None
        copy_38: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_373, slice_368);  slice_373 = slice_368 = None
        slice_scatter_69: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_374, copy_38, 3, 0, 256);  slice_374 = copy_38 = None
        slice_scatter_70: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_12, slice_scatter_69, 1, 1, 9223372036854775807);  select_scatter_12 = slice_scatter_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_89: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_70, 1, 0)
        slice_384: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_89, 1, 1, 256)
        select_88: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_70, 1, 0)
        slice_382: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_88, 1, 1, 256);  select_88 = None
        slice_383: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_382, 2, 1, 256);  slice_382 = None
        select_84: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_383, 1, 0);  view_383 = None
        slice_377: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_84, 1, 0, 255);  select_84 = None
        slice_378: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_377, 2, -255, 9223372036854775807);  slice_377 = None
        copy_39: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_383, slice_378);  slice_383 = slice_378 = None
        slice_scatter_71: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_384, copy_39, 2, 1, 256);  slice_384 = copy_39 = None
        slice_scatter_72: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_89, slice_scatter_71, 1, 1, 256);  select_89 = slice_scatter_71 = None
        select_scatter_13: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_70, slice_scatter_72, 1, 0);  slice_scatter_70 = slice_scatter_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_391: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_13, [1, 12, 4096, 513])
        permute_311: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
        slice_397: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_311, 1, 0, 256)
        view_390: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_13, [1, 12, 4096, 513])
        permute_310: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
        slice_395: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_310, 1, 0, 256);  permute_310 = None
        slice_396: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_395, 3, 0, 257);  slice_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_12: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_68: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_12, -2);  iota_12 = None
        iota_13: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_69: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
        sub_25: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_68, unsqueeze_69);  unsqueeze_68 = unsqueeze_69 = None
        le_6: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_25, 0);  sub_25 = None
        full_default_24: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "f16[256, 257]" = torch.ops.aten.where.self(le_6, full_default_24, full_default_25);  le_6 = full_default_24 = full_default_25 = None
        rev_12: "f16[256, 257]" = torch.ops.prims.rev.default(where_24, [0]);  where_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_70: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_12, 0);  rev_12 = None
        unsqueeze_71: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, 2);  unsqueeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_12: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_71, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_116: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_12, torch.bool);  expand_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_29: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_308: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_29, [0, 2, 1, 3]);  full_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_388: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_13, [1, 12, 4096, 513]);  select_scatter_13 = None
        permute_307: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        slice_390: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_307, 1, 0, 256);  permute_307 = None
        slice_391: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_390, 3, 0, 257);  slice_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_25: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_116, permute_308, slice_391);  convert_element_type_116 = permute_308 = slice_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_40: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_396, where_25);  slice_396 = where_25 = None
        slice_scatter_73: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_397, copy_40, 3, 0, 257);  slice_397 = copy_40 = None
        slice_scatter_74: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_311, slice_scatter_73, 1, 0, 256);  permute_311 = slice_scatter_73 = None
        permute_312: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_74, [0, 2, 1, 3]);  slice_scatter_74 = None
        view_392: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_312, [12, 16, 256, 513]);  permute_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_399: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_392, [1, 12, 4096, 513])
        permute_320: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        slice_410: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_320, 1, -256, 9223372036854775807)
        view_398: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_392, [1, 12, 4096, 513])
        permute_319: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        slice_408: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_319, 1, -256, 9223372036854775807);  permute_319 = None
        slice_409: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_408, 3, -257, 9223372036854775807);  slice_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_13: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_71, [1, 3]);  unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_13: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_13, [1, 256, 12, 257]);  rev_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_117: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_13, torch.bool);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_30: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_317: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_30, [0, 2, 1, 3]);  full_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_396: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_392, [1, 12, 4096, 513]);  view_392 = None
        permute_316: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None
        slice_403: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_316, 1, -256, 9223372036854775807);  permute_316 = None
        slice_404: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_403, 3, -257, 9223372036854775807);  slice_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_26: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_117, permute_317, slice_404);  convert_element_type_117 = permute_317 = slice_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_41: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_409, where_26);  slice_409 = where_26 = None
        slice_scatter_75: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_410, copy_41, 3, -257, 9223372036854775807);  slice_410 = copy_41 = None
        slice_scatter_76: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_320, slice_scatter_75, 1, -256, 9223372036854775807);  permute_320 = slice_scatter_75 = None
        permute_321: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_76, [0, 2, 1, 3]);  slice_scatter_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_346: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_321, [0, 2, 1, 3]);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_32: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_417: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_32, 1, 0, -1)
        slice_415: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_32, 1, 0, -1)
        slice_416: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_415, 3, 256, 9223372036854775807);  slice_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_31: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_323: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_31, [0, 2, 1, 3]);  full_31 = None
        view_402: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_323, [1, 4096, 1]);  permute_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_404: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_402, [1, 8, 512, 1]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_33: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_404, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_74: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_33, 4);  as_strided_33 = None
        permute_325: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_74, [0, 1, 2, 4, 3]);  unsqueeze_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_26: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_3: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_72: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_3, 2);  ne_3 = None
        unsqueeze_73: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_118: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_73, torch.float16)
        where_27: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_73, full_default_26, convert_element_type_118);  unsqueeze_73 = full_default_26 = convert_element_type_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_324: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_27, [0, 2, 1, 3]);  where_27 = None
        view_403: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_324, [1, 4096, 1]);  permute_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_405: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_403, [1, 8, 512, 1]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_34: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_405, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_75: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_34, 4);  as_strided_34 = None
        permute_326: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 2, 3]);  unsqueeze_75 = None
        mul_24: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_325, permute_326);  permute_325 = permute_326 = None
        view_406: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_24, [1, 15, 512, 512]);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_13: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_406, [0, 0, 0, 1], 0.0);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_407: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_13, [1, 15, 512, 513]);  constant_pad_nd_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_413: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_407, 2, 0, 256)
        slice_414: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_413, 3, 0, 257);  slice_413 = None
        copy_42: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_416, slice_414);  slice_416 = slice_414 = None
        slice_scatter_77: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_417, copy_42, 3, 256, 9223372036854775807);  slice_417 = copy_42 = None
        slice_scatter_78: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_32, slice_scatter_77, 1, 0, -1);  full_32 = slice_scatter_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_95: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_78, 1, -1)
        select_94: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_78, 1, -1)
        slice_423: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_94, 2, 256, 9223372036854775807);  select_94 = None
        select_91: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_407, 1, -1)
        slice_420: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_91, 1, 256, 9223372036854775807);  select_91 = None
        slice_421: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_420, 2, 0, 257);  slice_420 = None
        copy_43: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_423, slice_421);  slice_423 = slice_421 = None
        slice_scatter_79: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_95, copy_43, 2, 256, 9223372036854775807);  select_95 = copy_43 = None
        select_scatter_14: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_78, slice_scatter_79, 1, -1);  slice_scatter_78 = slice_scatter_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_432: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_14, 1, 1, 9223372036854775807)
        slice_430: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_14, 1, 1, 9223372036854775807)
        slice_431: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_430, 3, 0, 256);  slice_430 = None
        slice_425: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_407, 2, -257, -1)
        slice_426: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_425, 3, 257, 9223372036854775807);  slice_425 = None
        copy_44: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_431, slice_426);  slice_431 = slice_426 = None
        slice_scatter_80: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_432, copy_44, 3, 0, 256);  slice_432 = copy_44 = None
        slice_scatter_81: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_14, slice_scatter_80, 1, 1, 9223372036854775807);  select_scatter_14 = slice_scatter_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_102: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_81, 1, 0)
        slice_442: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_102, 1, 1, 256)
        select_101: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_81, 1, 0)
        slice_440: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_101, 1, 1, 256);  select_101 = None
        slice_441: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_440, 2, 1, 256);  slice_440 = None
        select_97: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_407, 1, 0);  view_407 = None
        slice_435: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_97, 1, 0, 255);  select_97 = None
        slice_436: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_435, 2, -255, 9223372036854775807);  slice_435 = None
        copy_45: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_441, slice_436);  slice_441 = slice_436 = None
        slice_scatter_82: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_442, copy_45, 2, 1, 256);  slice_442 = copy_45 = None
        slice_scatter_83: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_102, slice_scatter_82, 1, 1, 256);  select_102 = slice_scatter_82 = None
        select_scatter_15: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_81, slice_scatter_83, 1, 0);  slice_scatter_81 = slice_scatter_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_415: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_15, [1, 1, 4096, 513])
        permute_334: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
        slice_455: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_334, 1, 0, 256)
        view_414: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_15, [1, 1, 4096, 513])
        permute_333: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
        slice_453: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_333, 1, 0, 256);  permute_333 = None
        slice_454: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_453, 3, 0, 257);  slice_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_14: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_76: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_14, -2);  iota_14 = None
        iota_15: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_77: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
        sub_27: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_76, unsqueeze_77);  unsqueeze_76 = unsqueeze_77 = None
        le_7: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_27, 0);  sub_27 = None
        full_default_27: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "f16[256, 257]" = torch.ops.aten.where.self(le_7, full_default_27, full_default_28);  le_7 = full_default_27 = full_default_28 = None
        rev_14: "f16[256, 257]" = torch.ops.prims.rev.default(where_28, [0]);  where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_78: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_14, 0);  rev_14 = None
        unsqueeze_79: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 2);  unsqueeze_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_14: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_79, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_119: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_14, torch.bool);  expand_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_29: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_412: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_15, [1, 1, 4096, 513]);  select_scatter_15 = None
        permute_330: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        slice_448: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_330, 1, 0, 256);  permute_330 = None
        slice_449: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_448, 3, 0, 257);  slice_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_29: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_119, full_default_29, slice_449);  convert_element_type_119 = full_default_29 = slice_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_46: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_454, where_29);  slice_454 = where_29 = None
        slice_scatter_84: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_455, copy_46, 3, 0, 257);  slice_455 = copy_46 = None
        slice_scatter_85: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_334, slice_scatter_84, 1, 0, 256);  permute_334 = slice_scatter_84 = None
        permute_335: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_85, [0, 2, 1, 3]);  slice_scatter_85 = None
        view_416: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_335, [1, 16, 256, 513]);  permute_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_423: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_416, [1, 1, 4096, 513])
        permute_343: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None
        slice_468: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_343, 1, -256, 9223372036854775807)
        view_422: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_416, [1, 1, 4096, 513])
        permute_342: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
        slice_466: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_342, 1, -256, 9223372036854775807);  permute_342 = None
        slice_467: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_466, 3, -257, 9223372036854775807);  slice_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_15: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_79, [1, 3]);  unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_15: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_15, [1, 256, 1, 257]);  rev_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_120: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_15, torch.bool);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_30: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_420: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_416, [1, 1, 4096, 513]);  view_416 = None
        permute_339: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        slice_461: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_339, 1, -256, 9223372036854775807);  permute_339 = None
        slice_462: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_461, 3, -257, 9223372036854775807);  slice_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_30: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_120, full_default_30, slice_462);  convert_element_type_120 = full_default_30 = slice_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_47: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_467, where_30);  slice_467 = where_30 = None
        slice_scatter_86: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_468, copy_47, 3, -257, 9223372036854775807);  slice_468 = copy_47 = None
        slice_scatter_87: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_343, slice_scatter_86, 1, -256, 9223372036854775807);  permute_343 = slice_scatter_86 = None
        permute_344: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_87, [0, 2, 1, 3]);  slice_scatter_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_347: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_344, [0, 2, 1, 3]);  permute_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_35: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_346, permute_347);  permute_346 = permute_347 = None
        permute_348: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_35, [0, 2, 1, 3]);  add_35 = None
        permute_349: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_348, [0, 2, 1, 3]);  permute_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_42: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_349, memory_format = torch.contiguous_format);  permute_349 = None
        convert_element_type_121: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_42, torch.float32);  clone_42 = None
        amax_3: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_121, [-1], True)
        sub_28: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_121, amax_3);  convert_element_type_121 = amax_3 = None
        exp_3: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_4: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_37: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_31: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_81, full_default_31, div_37);  unsqueeze_81 = full_default_31 = div_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_122: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_31, torch.float16);  where_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_351: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_122, [0, 2, 1, 3]);  convert_element_type_122 = None
        view_432: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_351, [12, 16, 256, 513]);  permute_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_15: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_432, [0, 257], 0.0);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_434: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_15, [12, 16, -1]);  constant_pad_nd_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_471: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_434, 2, 0, -256);  view_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_435: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_471, [12, 16, 256, 769]);  slice_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_472: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_435, 3, 0, -1);  view_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_82: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_472, 4);  slice_472 = None
        view_436: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_82, [192, 256, 768]);  unsqueeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_340: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_273, [4096, 768]);  permute_273 = None
        permute_276: "f16[768, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_20: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg56_1, view_340, permute_276);  arg56_1 = view_340 = permute_276 = None
        view_341: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_20, [4096, 1, 768]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_431: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_341, [4096, 1, 12, 64]);  view_341 = None
        permute_350: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_431, [1, 0, 2, 3]);  view_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_352: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_350, [0, 2, 1, 3]);  permute_350 = None
        view_433: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_352, [12, 4096, 64]);  permute_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_14: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_433, [0, 0, 256, 256], -1.0);  view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_35: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_14, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_83: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_35, 4);  as_strided_35 = None
        clone_44: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_83, memory_format = torch.contiguous_format);  unsqueeze_83 = None
        view_437: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_44, [192, 768, 64]);  clone_44 = None
        bmm_7: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_436, view_437);  view_436 = view_437 = None
        view_438: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_7, [12, 16, 256, 1, 64]);  bmm_7 = None
        permute_357: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_438, [0, 1, 2, 4, 3]);  view_438 = None
        view_439: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_357, [12, 16, 256, 64]);  permute_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_440: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_439, [1, 12, 4096, 64]);  view_439 = None
        permute_358: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_359: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_358, [1, 0, 2, 3]);  permute_358 = None
        clone_45: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_441: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_45, [4096, 1, 768]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_360: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_441, [1, 0, 2]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_442: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_360, [4096, 768]);  permute_360 = None
        permute_361: "f16[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_21: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg58_1, view_442, permute_361);  arg58_1 = view_442 = permute_361 = None
        view_443: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 4096, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_37: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_443, convert_element_type_104);  view_443 = convert_element_type_104 = None
        convert_element_type_128: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_128, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 4096, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 4096, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_30: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_128, getitem_13);  convert_element_type_128 = getitem_13 = None
        add_38: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_25: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_6);  sub_30 = rsqrt_6 = None
        mul_26: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg59_1);  mul_25 = arg59_1 = None
        add_39: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_26, arg60_1);  mul_26 = arg60_1 = None
        convert_element_type_129: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_39, torch.float16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_444: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_129, [4096, 768])
        permute_362: "f16[768, 3072]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_22: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg62_1, view_444, permute_362);  arg62_1 = view_444 = permute_362 = None
        view_445: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 4096, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_133: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_445, torch.float32);  view_445 = None
        mul_27: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_133, 0.5)
        mul_28: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_133, 0.7071067811865476);  convert_element_type_133 = None
        erf_3: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_40: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_29: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_40);  mul_27 = add_40 = None
        convert_element_type_134: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_29, torch.float16);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_446: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_134, [4096, 3072]);  convert_element_type_134 = None
        permute_363: "f16[3072, 768]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_23: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg64_1, view_446, permute_363);  arg64_1 = view_446 = permute_363 = None
        view_447: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 4096, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_41: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_447, convert_element_type_129);  view_447 = convert_element_type_129 = None
        convert_element_type_138: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_138, [2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 4096, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 4096, 1]" = var_mean_7[1];  var_mean_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_43: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_44: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_101: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_102: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 3);  unsqueeze_101 = None
        full_default_39: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_36: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_477: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_36, 1, 0, -1)
        slice_475: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_36, 1, 0, -1)
        slice_476: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_475, 3, 256, 9223372036854775807);  slice_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_31: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_138, getitem_15);  convert_element_type_138 = getitem_15 = None
        add_42: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_30: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_7);  sub_31 = rsqrt_7 = None
        mul_31: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg65_1);  mul_30 = arg65_1 = None
        add_43: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_31, arg66_1);  mul_31 = arg66_1 = None
        convert_element_type_139: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.float16);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_364: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_448: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_364, [4096, 768])
        permute_365: "f16[768, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_24: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg68_1, view_448, permute_365);  arg68_1 = view_448 = permute_365 = None
        view_449: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_24, [4096, 1, 768]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_40: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_449, 8.0);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_488: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_40, [4096, 1, 12, 64]);  div_40 = None
        permute_389: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_488, [1, 0, 2, 3]);  view_488 = None
        permute_390: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_389, [0, 2, 1, 3]);  permute_389 = None
        view_489: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_390, [12, 4096, 64]);  permute_390 = None
        view_490: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_489, [12, 8, 512, 64]);  view_489 = None
        as_strided_41: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_490, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_490 = None
        unsqueeze_88: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_41, 4);  as_strided_41 = None
        clone_48: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_88, memory_format = torch.contiguous_format);  unsqueeze_88 = None
        view_491: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_48, [180, 512, 64]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_450: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_364, [4096, 768])
        permute_366: "f16[768, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_25: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg70_1, view_450, permute_366);  arg70_1 = view_450 = permute_366 = None
        view_451: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_25, [4096, 1, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_459: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_451, [4096, 1, 12, 64]);  view_451 = None
        permute_369: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_459, [1, 0, 2, 3]);  view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_374: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_369, [0, 2, 1, 3]);  permute_369 = None
        view_465: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_374, [12, 4096, 64]);  permute_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_474: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_465, [12, 8, 512, 64]);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_37: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_474, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_86: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_37, 4);  as_strided_37 = None
        permute_384: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_86, [0, 1, 4, 2, 3]);  unsqueeze_86 = None
        permute_393: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_384, [0, 1, 4, 3, 2]);  permute_384 = None
        clone_49: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
        view_492: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_49, [180, 64, 512]);  clone_49 = None
        bmm_8: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_491, view_492);  view_491 = view_492 = None
        view_493: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_8, [12, 15, 512, 1, 512]);  bmm_8 = None
        permute_394: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_493, [0, 1, 2, 4, 3]);  view_493 = None
        view_494: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_394, [12, 15, 512, 512]);  permute_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_16: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_494, [0, 0, 0, 1], 0.0);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_495: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_16, [12, 15, 512, 513]);  constant_pad_nd_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_473: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_495, 2, 0, 256)
        slice_474: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_473, 3, 0, 257);  slice_473 = None
        copy_48: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_476, slice_474);  slice_476 = slice_474 = None
        slice_scatter_88: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_477, copy_48, 3, 256, 9223372036854775807);  slice_477 = copy_48 = None
        slice_scatter_89: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_36, slice_scatter_88, 1, 0, -1);  full_36 = slice_scatter_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_108: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_89, 1, -1)
        select_107: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_89, 1, -1)
        slice_483: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_107, 2, 256, 9223372036854775807);  select_107 = None
        select_104: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_495, 1, -1)
        slice_480: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_104, 1, 256, 9223372036854775807);  select_104 = None
        slice_481: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_480, 2, 0, 257);  slice_480 = None
        copy_49: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_483, slice_481);  slice_483 = slice_481 = None
        slice_scatter_90: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_108, copy_49, 2, 256, 9223372036854775807);  select_108 = copy_49 = None
        select_scatter_16: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_89, slice_scatter_90, 1, -1);  slice_scatter_89 = slice_scatter_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_492: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_16, 1, 1, 9223372036854775807)
        slice_490: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_16, 1, 1, 9223372036854775807)
        slice_491: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_490, 3, 0, 256);  slice_490 = None
        slice_485: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_495, 2, -257, -1)
        slice_486: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_485, 3, 257, 9223372036854775807);  slice_485 = None
        copy_50: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_491, slice_486);  slice_491 = slice_486 = None
        slice_scatter_91: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_492, copy_50, 3, 0, 256);  slice_492 = copy_50 = None
        slice_scatter_92: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_16, slice_scatter_91, 1, 1, 9223372036854775807);  select_scatter_16 = slice_scatter_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_115: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_92, 1, 0)
        slice_502: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_115, 1, 1, 256)
        select_114: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_92, 1, 0)
        slice_500: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_114, 1, 1, 256);  select_114 = None
        slice_501: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_500, 2, 1, 256);  slice_500 = None
        select_110: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_495, 1, 0);  view_495 = None
        slice_495: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_110, 1, 0, 255);  select_110 = None
        slice_496: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_495, 2, -255, 9223372036854775807);  slice_495 = None
        copy_51: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_501, slice_496);  slice_501 = slice_496 = None
        slice_scatter_93: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_502, copy_51, 2, 1, 256);  slice_502 = copy_51 = None
        slice_scatter_94: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_115, slice_scatter_93, 1, 1, 256);  select_115 = slice_scatter_93 = None
        select_scatter_17: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_92, slice_scatter_94, 1, 0);  slice_scatter_92 = slice_scatter_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_503: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_17, [1, 12, 4096, 513])
        permute_402: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_503, [0, 2, 1, 3]);  view_503 = None
        slice_515: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_402, 1, 0, 256)
        view_502: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_17, [1, 12, 4096, 513])
        permute_401: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
        slice_513: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_401, 1, 0, 256);  permute_401 = None
        slice_514: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_513, 3, 0, 257);  slice_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_16: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_89: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_16, -2);  iota_16 = None
        iota_17: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_90: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
        sub_33: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_89, unsqueeze_90);  unsqueeze_89 = unsqueeze_90 = None
        le_8: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_33, 0);  sub_33 = None
        full_default_32: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32: "f16[256, 257]" = torch.ops.aten.where.self(le_8, full_default_32, full_default_33);  le_8 = full_default_32 = full_default_33 = None
        rev_16: "f16[256, 257]" = torch.ops.prims.rev.default(where_32, [0]);  where_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_91: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_16, 0);  rev_16 = None
        unsqueeze_92: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 2);  unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_16: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_92, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_151: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_16, torch.bool);  expand_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_38: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_399: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_38, [0, 2, 1, 3]);  full_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_500: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_17, [1, 12, 4096, 513]);  select_scatter_17 = None
        permute_398: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
        slice_508: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_398, 1, 0, 256);  permute_398 = None
        slice_509: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_508, 3, 0, 257);  slice_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_33: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_151, permute_399, slice_509);  convert_element_type_151 = permute_399 = slice_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_52: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_514, where_33);  slice_514 = where_33 = None
        slice_scatter_95: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_515, copy_52, 3, 0, 257);  slice_515 = copy_52 = None
        slice_scatter_96: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_402, slice_scatter_95, 1, 0, 256);  permute_402 = slice_scatter_95 = None
        permute_403: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_96, [0, 2, 1, 3]);  slice_scatter_96 = None
        view_504: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_403, [12, 16, 256, 513]);  permute_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_511: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_504, [1, 12, 4096, 513])
        permute_411: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
        slice_528: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_411, 1, -256, 9223372036854775807)
        view_510: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_504, [1, 12, 4096, 513])
        permute_410: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
        slice_526: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_410, 1, -256, 9223372036854775807);  permute_410 = None
        slice_527: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_526, 3, -257, 9223372036854775807);  slice_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_17: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_92, [1, 3]);  unsqueeze_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_17: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_17, [1, 256, 12, 257]);  rev_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_152: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_17, torch.bool);  expand_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_39: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_408: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_39, [0, 2, 1, 3]);  full_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_508: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_504, [1, 12, 4096, 513]);  view_504 = None
        permute_407: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        slice_521: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_407, 1, -256, 9223372036854775807);  permute_407 = None
        slice_522: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_521, 3, -257, 9223372036854775807);  slice_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_34: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_152, permute_408, slice_522);  convert_element_type_152 = permute_408 = slice_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_53: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_527, where_34);  slice_527 = where_34 = None
        slice_scatter_97: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_528, copy_53, 3, -257, 9223372036854775807);  slice_528 = copy_53 = None
        slice_scatter_98: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_411, slice_scatter_97, 1, -256, 9223372036854775807);  permute_411 = slice_scatter_97 = None
        permute_412: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_98, [0, 2, 1, 3]);  slice_scatter_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_437: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_412, [0, 2, 1, 3]);  permute_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_41: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_535: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_41, 1, 0, -1)
        slice_533: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_41, 1, 0, -1)
        slice_534: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_533, 3, 256, 9223372036854775807);  slice_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_40: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_414: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_40, [0, 2, 1, 3]);  full_40 = None
        view_514: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_414, [1, 4096, 1]);  permute_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_516: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_514, [1, 8, 512, 1]);  view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_42: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_516, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_95: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_42, 4);  as_strided_42 = None
        permute_416: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_95, [0, 1, 2, 4, 3]);  unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_34: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_4: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_93: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_4, 2);  ne_4 = None
        unsqueeze_94: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_153: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_94, torch.float16)
        where_35: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_94, full_default_34, convert_element_type_153);  unsqueeze_94 = full_default_34 = convert_element_type_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_415: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_35, [0, 2, 1, 3]);  where_35 = None
        view_515: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_415, [1, 4096, 1]);  permute_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_517: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_515, [1, 8, 512, 1]);  view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_43: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_517, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_96: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_43, 4);  as_strided_43 = None
        permute_417: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_96, [0, 1, 4, 2, 3]);  unsqueeze_96 = None
        mul_32: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_416, permute_417);  permute_416 = permute_417 = None
        view_518: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_32, [1, 15, 512, 512]);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_17: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_518, [0, 0, 0, 1], 0.0);  view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_519: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_17, [1, 15, 512, 513]);  constant_pad_nd_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_531: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_519, 2, 0, 256)
        slice_532: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_531, 3, 0, 257);  slice_531 = None
        copy_54: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_534, slice_532);  slice_534 = slice_532 = None
        slice_scatter_99: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_535, copy_54, 3, 256, 9223372036854775807);  slice_535 = copy_54 = None
        slice_scatter_100: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_41, slice_scatter_99, 1, 0, -1);  full_41 = slice_scatter_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_121: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_100, 1, -1)
        select_120: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_100, 1, -1)
        slice_541: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_120, 2, 256, 9223372036854775807);  select_120 = None
        select_117: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_519, 1, -1)
        slice_538: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_117, 1, 256, 9223372036854775807);  select_117 = None
        slice_539: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_538, 2, 0, 257);  slice_538 = None
        copy_55: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_541, slice_539);  slice_541 = slice_539 = None
        slice_scatter_101: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_121, copy_55, 2, 256, 9223372036854775807);  select_121 = copy_55 = None
        select_scatter_18: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_100, slice_scatter_101, 1, -1);  slice_scatter_100 = slice_scatter_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_550: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_18, 1, 1, 9223372036854775807)
        slice_548: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_18, 1, 1, 9223372036854775807)
        slice_549: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_548, 3, 0, 256);  slice_548 = None
        slice_543: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_519, 2, -257, -1)
        slice_544: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_543, 3, 257, 9223372036854775807);  slice_543 = None
        copy_56: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_549, slice_544);  slice_549 = slice_544 = None
        slice_scatter_102: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_550, copy_56, 3, 0, 256);  slice_550 = copy_56 = None
        slice_scatter_103: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_18, slice_scatter_102, 1, 1, 9223372036854775807);  select_scatter_18 = slice_scatter_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_128: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_103, 1, 0)
        slice_560: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_128, 1, 1, 256)
        select_127: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_103, 1, 0)
        slice_558: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_127, 1, 1, 256);  select_127 = None
        slice_559: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_558, 2, 1, 256);  slice_558 = None
        select_123: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_519, 1, 0);  view_519 = None
        slice_553: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_123, 1, 0, 255);  select_123 = None
        slice_554: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_553, 2, -255, 9223372036854775807);  slice_553 = None
        copy_57: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_559, slice_554);  slice_559 = slice_554 = None
        slice_scatter_104: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_560, copy_57, 2, 1, 256);  slice_560 = copy_57 = None
        slice_scatter_105: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_128, slice_scatter_104, 1, 1, 256);  select_128 = slice_scatter_104 = None
        select_scatter_19: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_103, slice_scatter_105, 1, 0);  slice_scatter_103 = slice_scatter_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_527: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_19, [1, 1, 4096, 513])
        permute_425: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None
        slice_573: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_425, 1, 0, 256)
        view_526: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_19, [1, 1, 4096, 513])
        permute_424: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
        slice_571: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_424, 1, 0, 256);  permute_424 = None
        slice_572: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_571, 3, 0, 257);  slice_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_18: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_97: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_18, -2);  iota_18 = None
        iota_19: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_98: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
        sub_35: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_97, unsqueeze_98);  unsqueeze_97 = unsqueeze_98 = None
        le_9: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_35, 0);  sub_35 = None
        full_default_35: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_36: "f16[256, 257]" = torch.ops.aten.where.self(le_9, full_default_35, full_default_36);  le_9 = full_default_35 = full_default_36 = None
        rev_18: "f16[256, 257]" = torch.ops.prims.rev.default(where_36, [0]);  where_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_99: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_18, 0);  rev_18 = None
        unsqueeze_100: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_18: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_100, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_154: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_18, torch.bool);  expand_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_37: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_524: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_19, [1, 1, 4096, 513]);  select_scatter_19 = None
        permute_421: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_524, [0, 2, 1, 3]);  view_524 = None
        slice_566: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_421, 1, 0, 256);  permute_421 = None
        slice_567: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_566, 3, 0, 257);  slice_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_37: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_154, full_default_37, slice_567);  convert_element_type_154 = full_default_37 = slice_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_58: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_572, where_37);  slice_572 = where_37 = None
        slice_scatter_106: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_573, copy_58, 3, 0, 257);  slice_573 = copy_58 = None
        slice_scatter_107: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_425, slice_scatter_106, 1, 0, 256);  permute_425 = slice_scatter_106 = None
        permute_426: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_107, [0, 2, 1, 3]);  slice_scatter_107 = None
        view_528: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_426, [1, 16, 256, 513]);  permute_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_535: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_528, [1, 1, 4096, 513])
        permute_434: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
        slice_586: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_434, 1, -256, 9223372036854775807)
        view_534: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_528, [1, 1, 4096, 513])
        permute_433: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
        slice_584: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_433, 1, -256, 9223372036854775807);  permute_433 = None
        slice_585: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_584, 3, -257, 9223372036854775807);  slice_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_19: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_100, [1, 3]);  unsqueeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_19: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_19, [1, 256, 1, 257]);  rev_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_155: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_19, torch.bool);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_38: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_532: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_528, [1, 1, 4096, 513]);  view_528 = None
        permute_430: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
        slice_579: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_430, 1, -256, 9223372036854775807);  permute_430 = None
        slice_580: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_579, 3, -257, 9223372036854775807);  slice_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_38: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_155, full_default_38, slice_580);  convert_element_type_155 = full_default_38 = slice_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_59: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_585, where_38);  slice_585 = where_38 = None
        slice_scatter_108: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_586, copy_59, 3, -257, 9223372036854775807);  slice_586 = copy_59 = None
        slice_scatter_109: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_434, slice_scatter_108, 1, -256, 9223372036854775807);  permute_434 = slice_scatter_108 = None
        permute_435: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_109, [0, 2, 1, 3]);  slice_scatter_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_438: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_435, [0, 2, 1, 3]);  permute_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_46: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_437, permute_438);  permute_437 = permute_438 = None
        permute_439: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_46, [0, 2, 1, 3]);  add_46 = None
        permute_440: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_439, [0, 2, 1, 3]);  permute_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_54: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_440, memory_format = torch.contiguous_format);  permute_440 = None
        convert_element_type_156: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_54, torch.float32);  clone_54 = None
        amax_4: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_156, [-1], True)
        sub_36: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_156, amax_4);  convert_element_type_156 = amax_4 = None
        exp_4: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_5: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_47: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_39: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_102, full_default_39, div_47);  unsqueeze_102 = full_default_39 = div_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_157: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_39, torch.float16);  where_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_442: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_157, [0, 2, 1, 3]);  convert_element_type_157 = None
        view_544: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_442, [12, 16, 256, 513]);  permute_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_19: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_544, [0, 257], 0.0);  view_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_546: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_19, [12, 16, -1]);  constant_pad_nd_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_589: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_546, 2, 0, -256);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_547: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_589, [12, 16, 256, 769]);  slice_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_590: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_547, 3, 0, -1);  view_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_103: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_590, 4);  slice_590 = None
        view_548: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_103, [192, 256, 768]);  unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_452: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_364, [4096, 768]);  permute_364 = None
        permute_367: "f16[768, 768]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_26: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg72_1, view_452, permute_367);  arg72_1 = view_452 = permute_367 = None
        view_453: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_26, [4096, 1, 768]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_543: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_453, [4096, 1, 12, 64]);  view_453 = None
        permute_441: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_543, [1, 0, 2, 3]);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_443: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_441, [0, 2, 1, 3]);  permute_441 = None
        view_545: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_443, [12, 4096, 64]);  permute_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_18: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_545, [0, 0, 256, 256], -1.0);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_44: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_18, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_104: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_44, 4);  as_strided_44 = None
        clone_56: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_104, memory_format = torch.contiguous_format);  unsqueeze_104 = None
        view_549: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_56, [192, 768, 64]);  clone_56 = None
        bmm_9: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_548, view_549);  view_548 = view_549 = None
        view_550: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_9, [12, 16, 256, 1, 64]);  bmm_9 = None
        permute_448: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_550, [0, 1, 2, 4, 3]);  view_550 = None
        view_551: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_448, [12, 16, 256, 64]);  permute_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_552: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_551, [1, 12, 4096, 64]);  view_551 = None
        permute_449: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_450: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_449, [1, 0, 2, 3]);  permute_449 = None
        clone_57: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None
        view_553: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_57, [4096, 1, 768]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_451: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_553, [1, 0, 2]);  view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_554: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_451, [4096, 768]);  permute_451 = None
        permute_452: "f16[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_27: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg74_1, view_554, permute_452);  arg74_1 = view_554 = permute_452 = None
        view_555: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 4096, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_48: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_555, convert_element_type_139);  view_555 = convert_element_type_139 = None
        convert_element_type_163: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_163, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 4096, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 4096, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_38: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_163, getitem_17);  convert_element_type_163 = getitem_17 = None
        add_49: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_33: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_8);  sub_38 = rsqrt_8 = None
        mul_34: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg75_1);  mul_33 = arg75_1 = None
        add_50: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_34, arg76_1);  mul_34 = arg76_1 = None
        convert_element_type_164: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_50, torch.float16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_556: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_164, [4096, 768])
        permute_453: "f16[768, 3072]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_28: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg78_1, view_556, permute_453);  arg78_1 = view_556 = permute_453 = None
        view_557: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 4096, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_168: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_557, torch.float32);  view_557 = None
        mul_35: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_168, 0.5)
        mul_36: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_168, 0.7071067811865476);  convert_element_type_168 = None
        erf_4: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_51: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_37: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_35, add_51);  mul_35 = add_51 = None
        convert_element_type_169: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_37, torch.float16);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_558: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_169, [4096, 3072]);  convert_element_type_169 = None
        permute_454: "f16[3072, 768]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_29: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg80_1, view_558, permute_454);  arg80_1 = view_558 = permute_454 = None
        view_559: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_29, [1, 4096, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_52: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_559, convert_element_type_164);  view_559 = convert_element_type_164 = None
        convert_element_type_173: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_52, torch.float32);  add_52 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_173, [2], correction = 0, keepdim = True)
        getitem_18: "f32[1, 4096, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 4096, 1]" = var_mean_9[1];  var_mean_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_52: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_53: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_122: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_123: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 3);  unsqueeze_122 = None
        full_default_47: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_45: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_595: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1)
        slice_593: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1)
        slice_594: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_593, 3, 256, 9223372036854775807);  slice_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_39: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_173, getitem_19);  convert_element_type_173 = getitem_19 = None
        add_53: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_38: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_9);  sub_39 = rsqrt_9 = None
        mul_39: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_38, arg81_1);  mul_38 = arg81_1 = None
        add_54: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_39, arg82_1);  mul_39 = arg82_1 = None
        convert_element_type_174: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_54, torch.float16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_455: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_560: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_455, [4096, 768])
        permute_456: "f16[768, 768]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_30: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg84_1, view_560, permute_456);  arg84_1 = view_560 = permute_456 = None
        view_561: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_30, [4096, 1, 768]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_50: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_561, 8.0);  view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_600: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_50, [4096, 1, 12, 64]);  div_50 = None
        permute_480: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_600, [1, 0, 2, 3]);  view_600 = None
        permute_481: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_480, [0, 2, 1, 3]);  permute_480 = None
        view_601: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_481, [12, 4096, 64]);  permute_481 = None
        view_602: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_601, [12, 8, 512, 64]);  view_601 = None
        as_strided_50: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_602, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_602 = None
        unsqueeze_109: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_50, 4);  as_strided_50 = None
        clone_60: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_109, memory_format = torch.contiguous_format);  unsqueeze_109 = None
        view_603: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_60, [180, 512, 64]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_562: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_455, [4096, 768])
        permute_457: "f16[768, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_31: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg86_1, view_562, permute_457);  arg86_1 = view_562 = permute_457 = None
        view_563: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_31, [4096, 1, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_571: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_563, [4096, 1, 12, 64]);  view_563 = None
        permute_460: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_571, [1, 0, 2, 3]);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_465: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_460, [0, 2, 1, 3]);  permute_460 = None
        view_577: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_465, [12, 4096, 64]);  permute_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_586: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_577, [12, 8, 512, 64]);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_46: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_586, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_107: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_46, 4);  as_strided_46 = None
        permute_475: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_107, [0, 1, 4, 2, 3]);  unsqueeze_107 = None
        permute_484: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_475, [0, 1, 4, 3, 2]);  permute_475 = None
        clone_61: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        view_604: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_61, [180, 64, 512]);  clone_61 = None
        bmm_10: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_603, view_604);  view_603 = view_604 = None
        view_605: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_10, [12, 15, 512, 1, 512]);  bmm_10 = None
        permute_485: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_605, [0, 1, 2, 4, 3]);  view_605 = None
        view_606: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_485, [12, 15, 512, 512]);  permute_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_20: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_606, [0, 0, 0, 1], 0.0);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_607: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_20, [12, 15, 512, 513]);  constant_pad_nd_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_591: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_607, 2, 0, 256)
        slice_592: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_591, 3, 0, 257);  slice_591 = None
        copy_60: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_594, slice_592);  slice_594 = slice_592 = None
        slice_scatter_110: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_595, copy_60, 3, 256, 9223372036854775807);  slice_595 = copy_60 = None
        slice_scatter_111: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_45, slice_scatter_110, 1, 0, -1);  full_45 = slice_scatter_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_134: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_111, 1, -1)
        select_133: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_111, 1, -1)
        slice_601: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_133, 2, 256, 9223372036854775807);  select_133 = None
        select_130: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_607, 1, -1)
        slice_598: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_130, 1, 256, 9223372036854775807);  select_130 = None
        slice_599: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_598, 2, 0, 257);  slice_598 = None
        copy_61: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_601, slice_599);  slice_601 = slice_599 = None
        slice_scatter_112: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_134, copy_61, 2, 256, 9223372036854775807);  select_134 = copy_61 = None
        select_scatter_20: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_111, slice_scatter_112, 1, -1);  slice_scatter_111 = slice_scatter_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_610: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_20, 1, 1, 9223372036854775807)
        slice_608: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_20, 1, 1, 9223372036854775807)
        slice_609: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_608, 3, 0, 256);  slice_608 = None
        slice_603: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_607, 2, -257, -1)
        slice_604: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_603, 3, 257, 9223372036854775807);  slice_603 = None
        copy_62: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_609, slice_604);  slice_609 = slice_604 = None
        slice_scatter_113: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_610, copy_62, 3, 0, 256);  slice_610 = copy_62 = None
        slice_scatter_114: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_20, slice_scatter_113, 1, 1, 9223372036854775807);  select_scatter_20 = slice_scatter_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_141: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_114, 1, 0)
        slice_620: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_141, 1, 1, 256)
        select_140: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_114, 1, 0)
        slice_618: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_140, 1, 1, 256);  select_140 = None
        slice_619: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_618, 2, 1, 256);  slice_618 = None
        select_136: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_607, 1, 0);  view_607 = None
        slice_613: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_136, 1, 0, 255);  select_136 = None
        slice_614: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_613, 2, -255, 9223372036854775807);  slice_613 = None
        copy_63: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_619, slice_614);  slice_619 = slice_614 = None
        slice_scatter_115: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_620, copy_63, 2, 1, 256);  slice_620 = copy_63 = None
        slice_scatter_116: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_141, slice_scatter_115, 1, 1, 256);  select_141 = slice_scatter_115 = None
        select_scatter_21: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_114, slice_scatter_116, 1, 0);  slice_scatter_114 = slice_scatter_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_615: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_21, [1, 12, 4096, 513])
        permute_493: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_615, [0, 2, 1, 3]);  view_615 = None
        slice_633: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_493, 1, 0, 256)
        view_614: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_21, [1, 12, 4096, 513])
        permute_492: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        slice_631: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_492, 1, 0, 256);  permute_492 = None
        slice_632: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_631, 3, 0, 257);  slice_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_20: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_110: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_20, -2);  iota_20 = None
        iota_21: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_111: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
        sub_41: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_110, unsqueeze_111);  unsqueeze_110 = unsqueeze_111 = None
        le_10: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_41, 0);  sub_41 = None
        full_default_40: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40: "f16[256, 257]" = torch.ops.aten.where.self(le_10, full_default_40, full_default_41);  le_10 = full_default_40 = full_default_41 = None
        rev_20: "f16[256, 257]" = torch.ops.prims.rev.default(where_40, [0]);  where_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_112: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_20, 0);  rev_20 = None
        unsqueeze_113: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 2);  unsqueeze_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_20: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_113, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_186: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_20, torch.bool);  expand_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_47: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_490: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_47, [0, 2, 1, 3]);  full_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_612: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_21, [1, 12, 4096, 513]);  select_scatter_21 = None
        permute_489: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
        slice_626: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_489, 1, 0, 256);  permute_489 = None
        slice_627: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_626, 3, 0, 257);  slice_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_41: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_186, permute_490, slice_627);  convert_element_type_186 = permute_490 = slice_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_64: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_632, where_41);  slice_632 = where_41 = None
        slice_scatter_117: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_633, copy_64, 3, 0, 257);  slice_633 = copy_64 = None
        slice_scatter_118: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_493, slice_scatter_117, 1, 0, 256);  permute_493 = slice_scatter_117 = None
        permute_494: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_118, [0, 2, 1, 3]);  slice_scatter_118 = None
        view_616: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_494, [12, 16, 256, 513]);  permute_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_623: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_616, [1, 12, 4096, 513])
        permute_502: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_623, [0, 2, 1, 3]);  view_623 = None
        slice_646: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_502, 1, -256, 9223372036854775807)
        view_622: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_616, [1, 12, 4096, 513])
        permute_501: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_622, [0, 2, 1, 3]);  view_622 = None
        slice_644: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_501, 1, -256, 9223372036854775807);  permute_501 = None
        slice_645: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_644, 3, -257, 9223372036854775807);  slice_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_21: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_113, [1, 3]);  unsqueeze_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_21: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_21, [1, 256, 12, 257]);  rev_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_187: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_21, torch.bool);  expand_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_48: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_499: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_48, [0, 2, 1, 3]);  full_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_620: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_616, [1, 12, 4096, 513]);  view_616 = None
        permute_498: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_620, [0, 2, 1, 3]);  view_620 = None
        slice_639: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_498, 1, -256, 9223372036854775807);  permute_498 = None
        slice_640: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_639, 3, -257, 9223372036854775807);  slice_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_42: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_187, permute_499, slice_640);  convert_element_type_187 = permute_499 = slice_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_65: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_645, where_42);  slice_645 = where_42 = None
        slice_scatter_119: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_646, copy_65, 3, -257, 9223372036854775807);  slice_646 = copy_65 = None
        slice_scatter_120: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_502, slice_scatter_119, 1, -256, 9223372036854775807);  permute_502 = slice_scatter_119 = None
        permute_503: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_120, [0, 2, 1, 3]);  slice_scatter_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_528: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_503, [0, 2, 1, 3]);  permute_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_50: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_653: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_50, 1, 0, -1)
        slice_651: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_50, 1, 0, -1)
        slice_652: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_651, 3, 256, 9223372036854775807);  slice_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_49: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_505: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_49, [0, 2, 1, 3]);  full_49 = None
        view_626: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_505, [1, 4096, 1]);  permute_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_628: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_626, [1, 8, 512, 1]);  view_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_51: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_628, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_116: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_51, 4);  as_strided_51 = None
        permute_507: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_116, [0, 1, 2, 4, 3]);  unsqueeze_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_42: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_5: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_114: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_5, 2);  ne_5 = None
        unsqueeze_115: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 3);  unsqueeze_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_188: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_115, torch.float16)
        where_43: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_115, full_default_42, convert_element_type_188);  unsqueeze_115 = full_default_42 = convert_element_type_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_506: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_43, [0, 2, 1, 3]);  where_43 = None
        view_627: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_506, [1, 4096, 1]);  permute_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_629: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_627, [1, 8, 512, 1]);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_52: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_629, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_117: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_52, 4);  as_strided_52 = None
        permute_508: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_117, [0, 1, 4, 2, 3]);  unsqueeze_117 = None
        mul_40: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_507, permute_508);  permute_507 = permute_508 = None
        view_630: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_40, [1, 15, 512, 512]);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_21: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_630, [0, 0, 0, 1], 0.0);  view_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_631: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_21, [1, 15, 512, 513]);  constant_pad_nd_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_649: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_631, 2, 0, 256)
        slice_650: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_649, 3, 0, 257);  slice_649 = None
        copy_66: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_652, slice_650);  slice_652 = slice_650 = None
        slice_scatter_121: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_653, copy_66, 3, 256, 9223372036854775807);  slice_653 = copy_66 = None
        slice_scatter_122: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_50, slice_scatter_121, 1, 0, -1);  full_50 = slice_scatter_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_147: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_122, 1, -1)
        select_146: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_122, 1, -1)
        slice_659: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_146, 2, 256, 9223372036854775807);  select_146 = None
        select_143: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_631, 1, -1)
        slice_656: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_143, 1, 256, 9223372036854775807);  select_143 = None
        slice_657: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_656, 2, 0, 257);  slice_656 = None
        copy_67: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_659, slice_657);  slice_659 = slice_657 = None
        slice_scatter_123: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_147, copy_67, 2, 256, 9223372036854775807);  select_147 = copy_67 = None
        select_scatter_22: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_122, slice_scatter_123, 1, -1);  slice_scatter_122 = slice_scatter_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_668: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_22, 1, 1, 9223372036854775807)
        slice_666: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_22, 1, 1, 9223372036854775807)
        slice_667: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_666, 3, 0, 256);  slice_666 = None
        slice_661: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_631, 2, -257, -1)
        slice_662: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_661, 3, 257, 9223372036854775807);  slice_661 = None
        copy_68: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_667, slice_662);  slice_667 = slice_662 = None
        slice_scatter_124: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_668, copy_68, 3, 0, 256);  slice_668 = copy_68 = None
        slice_scatter_125: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_22, slice_scatter_124, 1, 1, 9223372036854775807);  select_scatter_22 = slice_scatter_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_154: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_125, 1, 0)
        slice_678: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_154, 1, 1, 256)
        select_153: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_125, 1, 0)
        slice_676: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_153, 1, 1, 256);  select_153 = None
        slice_677: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_676, 2, 1, 256);  slice_676 = None
        select_149: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_631, 1, 0);  view_631 = None
        slice_671: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_149, 1, 0, 255);  select_149 = None
        slice_672: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_671, 2, -255, 9223372036854775807);  slice_671 = None
        copy_69: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_677, slice_672);  slice_677 = slice_672 = None
        slice_scatter_126: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_678, copy_69, 2, 1, 256);  slice_678 = copy_69 = None
        slice_scatter_127: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_154, slice_scatter_126, 1, 1, 256);  select_154 = slice_scatter_126 = None
        select_scatter_23: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_125, slice_scatter_127, 1, 0);  slice_scatter_125 = slice_scatter_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_639: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_23, [1, 1, 4096, 513])
        permute_516: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_639, [0, 2, 1, 3]);  view_639 = None
        slice_691: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_516, 1, 0, 256)
        view_638: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_23, [1, 1, 4096, 513])
        permute_515: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None
        slice_689: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_515, 1, 0, 256);  permute_515 = None
        slice_690: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_689, 3, 0, 257);  slice_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_22: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_118: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_22, -2);  iota_22 = None
        iota_23: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_119: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
        sub_43: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_118, unsqueeze_119);  unsqueeze_118 = unsqueeze_119 = None
        le_11: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_43, 0);  sub_43 = None
        full_default_43: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_44: "f16[256, 257]" = torch.ops.aten.where.self(le_11, full_default_43, full_default_44);  le_11 = full_default_43 = full_default_44 = None
        rev_22: "f16[256, 257]" = torch.ops.prims.rev.default(where_44, [0]);  where_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_120: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_22, 0);  rev_22 = None
        unsqueeze_121: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_22: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_121, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_189: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_22, torch.bool);  expand_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_45: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_636: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_23, [1, 1, 4096, 513]);  select_scatter_23 = None
        permute_512: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_636, [0, 2, 1, 3]);  view_636 = None
        slice_684: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_512, 1, 0, 256);  permute_512 = None
        slice_685: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_684, 3, 0, 257);  slice_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_45: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_189, full_default_45, slice_685);  convert_element_type_189 = full_default_45 = slice_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_70: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_690, where_45);  slice_690 = where_45 = None
        slice_scatter_128: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_691, copy_70, 3, 0, 257);  slice_691 = copy_70 = None
        slice_scatter_129: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_516, slice_scatter_128, 1, 0, 256);  permute_516 = slice_scatter_128 = None
        permute_517: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_129, [0, 2, 1, 3]);  slice_scatter_129 = None
        view_640: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_517, [1, 16, 256, 513]);  permute_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_647: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_640, [1, 1, 4096, 513])
        permute_525: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_647, [0, 2, 1, 3]);  view_647 = None
        slice_704: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_525, 1, -256, 9223372036854775807)
        view_646: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_640, [1, 1, 4096, 513])
        permute_524: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_646, [0, 2, 1, 3]);  view_646 = None
        slice_702: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_524, 1, -256, 9223372036854775807);  permute_524 = None
        slice_703: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_702, 3, -257, 9223372036854775807);  slice_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_23: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_121, [1, 3]);  unsqueeze_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_23: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_23, [1, 256, 1, 257]);  rev_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_190: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_23, torch.bool);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_46: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_644: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_640, [1, 1, 4096, 513]);  view_640 = None
        permute_521: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_644, [0, 2, 1, 3]);  view_644 = None
        slice_697: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_521, 1, -256, 9223372036854775807);  permute_521 = None
        slice_698: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_697, 3, -257, 9223372036854775807);  slice_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_46: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_190, full_default_46, slice_698);  convert_element_type_190 = full_default_46 = slice_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_71: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_703, where_46);  slice_703 = where_46 = None
        slice_scatter_130: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_704, copy_71, 3, -257, 9223372036854775807);  slice_704 = copy_71 = None
        slice_scatter_131: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_525, slice_scatter_130, 1, -256, 9223372036854775807);  permute_525 = slice_scatter_130 = None
        permute_526: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_131, [0, 2, 1, 3]);  slice_scatter_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_529: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_526, [0, 2, 1, 3]);  permute_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_57: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_528, permute_529);  permute_528 = permute_529 = None
        permute_530: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_57, [0, 2, 1, 3]);  add_57 = None
        permute_531: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_530, [0, 2, 1, 3]);  permute_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_66: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_531, memory_format = torch.contiguous_format);  permute_531 = None
        convert_element_type_191: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_66, torch.float32);  clone_66 = None
        amax_5: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_191, [-1], True)
        sub_44: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_191, amax_5);  convert_element_type_191 = amax_5 = None
        exp_5: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
        sum_6: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_57: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_47: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_123, full_default_47, div_57);  unsqueeze_123 = full_default_47 = div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_192: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_47, torch.float16);  where_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_533: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_192, [0, 2, 1, 3]);  convert_element_type_192 = None
        view_656: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_533, [12, 16, 256, 513]);  permute_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_23: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_656, [0, 257], 0.0);  view_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_658: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_23, [12, 16, -1]);  constant_pad_nd_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_707: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_658, 2, 0, -256);  view_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_659: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_707, [12, 16, 256, 769]);  slice_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_708: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_659, 3, 0, -1);  view_659 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_124: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_708, 4);  slice_708 = None
        view_660: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_124, [192, 256, 768]);  unsqueeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_564: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_455, [4096, 768]);  permute_455 = None
        permute_458: "f16[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_32: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg88_1, view_564, permute_458);  arg88_1 = view_564 = permute_458 = None
        view_565: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_32, [4096, 1, 768]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_655: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_565, [4096, 1, 12, 64]);  view_565 = None
        permute_532: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_655, [1, 0, 2, 3]);  view_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_534: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_532, [0, 2, 1, 3]);  permute_532 = None
        view_657: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_534, [12, 4096, 64]);  permute_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_22: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_657, [0, 0, 256, 256], -1.0);  view_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_53: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_22, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_125: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_53, 4);  as_strided_53 = None
        clone_68: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_125, memory_format = torch.contiguous_format);  unsqueeze_125 = None
        view_661: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_68, [192, 768, 64]);  clone_68 = None
        bmm_11: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_660, view_661);  view_660 = view_661 = None
        view_662: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_11, [12, 16, 256, 1, 64]);  bmm_11 = None
        permute_539: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_662, [0, 1, 2, 4, 3]);  view_662 = None
        view_663: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_539, [12, 16, 256, 64]);  permute_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_664: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_663, [1, 12, 4096, 64]);  view_663 = None
        permute_540: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_664, [0, 2, 1, 3]);  view_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_541: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_540, [1, 0, 2, 3]);  permute_540 = None
        clone_69: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_541, memory_format = torch.contiguous_format);  permute_541 = None
        view_665: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_69, [4096, 1, 768]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_542: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_665, [1, 0, 2]);  view_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_666: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_542, [4096, 768]);  permute_542 = None
        permute_543: "f16[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_33: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg90_1, view_666, permute_543);  arg90_1 = view_666 = permute_543 = None
        view_667: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_33, [1, 4096, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_667, convert_element_type_174);  view_667 = convert_element_type_174 = None
        convert_element_type_198: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_198, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 4096, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 4096, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_46: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_198, getitem_21);  convert_element_type_198 = getitem_21 = None
        add_60: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_41: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_10);  sub_46 = rsqrt_10 = None
        mul_42: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg91_1);  mul_41 = arg91_1 = None
        add_61: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_42, arg92_1);  mul_42 = arg92_1 = None
        convert_element_type_199: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_61, torch.float16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_668: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_199, [4096, 768])
        permute_544: "f16[768, 3072]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_34: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg94_1, view_668, permute_544);  arg94_1 = view_668 = permute_544 = None
        view_669: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 4096, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_203: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_669, torch.float32);  view_669 = None
        mul_43: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_203, 0.5)
        mul_44: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_203, 0.7071067811865476);  convert_element_type_203 = None
        erf_5: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
        add_62: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_45: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_43, add_62);  mul_43 = add_62 = None
        convert_element_type_204: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_45, torch.float16);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_670: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_204, [4096, 3072]);  convert_element_type_204 = None
        permute_545: "f16[3072, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_35: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg96_1, view_670, permute_545);  arg96_1 = view_670 = permute_545 = None
        view_671: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_35, [1, 4096, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_671, convert_element_type_199);  view_671 = convert_element_type_199 = None
        convert_element_type_208: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_208, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 4096, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 4096, 1]" = var_mean_11[1];  var_mean_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_61: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_62: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_143: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_144: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
        full_default_55: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_54: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_713: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_54, 1, 0, -1)
        slice_711: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_54, 1, 0, -1)
        slice_712: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_711, 3, 256, 9223372036854775807);  slice_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_47: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_208, getitem_23);  convert_element_type_208 = getitem_23 = None
        add_64: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_46: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_11);  sub_47 = rsqrt_11 = None
        mul_47: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_46, arg97_1);  mul_46 = arg97_1 = None
        add_65: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_47, arg98_1);  mul_47 = arg98_1 = None
        convert_element_type_209: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_546: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_209, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_672: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_546, [4096, 768])
        permute_547: "f16[768, 768]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_36: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg100_1, view_672, permute_547);  arg100_1 = view_672 = permute_547 = None
        view_673: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_36, [4096, 1, 768]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_60: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_673, 8.0);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_712: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_60, [4096, 1, 12, 64]);  div_60 = None
        permute_571: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_712, [1, 0, 2, 3]);  view_712 = None
        permute_572: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_571, [0, 2, 1, 3]);  permute_571 = None
        view_713: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_572, [12, 4096, 64]);  permute_572 = None
        view_714: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_713, [12, 8, 512, 64]);  view_713 = None
        as_strided_59: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_714, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_714 = None
        unsqueeze_130: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_59, 4);  as_strided_59 = None
        clone_72: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_130, memory_format = torch.contiguous_format);  unsqueeze_130 = None
        view_715: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_72, [180, 512, 64]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_674: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_546, [4096, 768])
        permute_548: "f16[768, 768]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_37: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg102_1, view_674, permute_548);  arg102_1 = view_674 = permute_548 = None
        view_675: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_37, [4096, 1, 768]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_683: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_675, [4096, 1, 12, 64]);  view_675 = None
        permute_551: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_683, [1, 0, 2, 3]);  view_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_556: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_551, [0, 2, 1, 3]);  permute_551 = None
        view_689: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_556, [12, 4096, 64]);  permute_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_698: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_689, [12, 8, 512, 64]);  view_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_55: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_698, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_128: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_55, 4);  as_strided_55 = None
        permute_566: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_128, [0, 1, 4, 2, 3]);  unsqueeze_128 = None
        permute_575: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_566, [0, 1, 4, 3, 2]);  permute_566 = None
        clone_73: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
        view_716: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_73, [180, 64, 512]);  clone_73 = None
        bmm_12: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_715, view_716);  view_715 = view_716 = None
        view_717: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_12, [12, 15, 512, 1, 512]);  bmm_12 = None
        permute_576: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_717, [0, 1, 2, 4, 3]);  view_717 = None
        view_718: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_576, [12, 15, 512, 512]);  permute_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_24: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_718, [0, 0, 0, 1], 0.0);  view_718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_719: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_24, [12, 15, 512, 513]);  constant_pad_nd_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_709: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_719, 2, 0, 256)
        slice_710: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_709, 3, 0, 257);  slice_709 = None
        copy_72: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_712, slice_710);  slice_712 = slice_710 = None
        slice_scatter_132: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_713, copy_72, 3, 256, 9223372036854775807);  slice_713 = copy_72 = None
        slice_scatter_133: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_54, slice_scatter_132, 1, 0, -1);  full_54 = slice_scatter_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_160: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_133, 1, -1)
        select_159: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_133, 1, -1)
        slice_719: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_159, 2, 256, 9223372036854775807);  select_159 = None
        select_156: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_719, 1, -1)
        slice_716: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_156, 1, 256, 9223372036854775807);  select_156 = None
        slice_717: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_716, 2, 0, 257);  slice_716 = None
        copy_73: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_719, slice_717);  slice_719 = slice_717 = None
        slice_scatter_134: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_160, copy_73, 2, 256, 9223372036854775807);  select_160 = copy_73 = None
        select_scatter_24: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_133, slice_scatter_134, 1, -1);  slice_scatter_133 = slice_scatter_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_728: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_24, 1, 1, 9223372036854775807)
        slice_726: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_24, 1, 1, 9223372036854775807)
        slice_727: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_726, 3, 0, 256);  slice_726 = None
        slice_721: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_719, 2, -257, -1)
        slice_722: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_721, 3, 257, 9223372036854775807);  slice_721 = None
        copy_74: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_727, slice_722);  slice_727 = slice_722 = None
        slice_scatter_135: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_728, copy_74, 3, 0, 256);  slice_728 = copy_74 = None
        slice_scatter_136: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_24, slice_scatter_135, 1, 1, 9223372036854775807);  select_scatter_24 = slice_scatter_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_167: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_136, 1, 0)
        slice_738: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_167, 1, 1, 256)
        select_166: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_136, 1, 0)
        slice_736: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_166, 1, 1, 256);  select_166 = None
        slice_737: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_736, 2, 1, 256);  slice_736 = None
        select_162: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_719, 1, 0);  view_719 = None
        slice_731: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_162, 1, 0, 255);  select_162 = None
        slice_732: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_731, 2, -255, 9223372036854775807);  slice_731 = None
        copy_75: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_737, slice_732);  slice_737 = slice_732 = None
        slice_scatter_137: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_738, copy_75, 2, 1, 256);  slice_738 = copy_75 = None
        slice_scatter_138: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_167, slice_scatter_137, 1, 1, 256);  select_167 = slice_scatter_137 = None
        select_scatter_25: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_136, slice_scatter_138, 1, 0);  slice_scatter_136 = slice_scatter_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_727: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_25, [1, 12, 4096, 513])
        permute_584: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_727, [0, 2, 1, 3]);  view_727 = None
        slice_751: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_584, 1, 0, 256)
        view_726: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_25, [1, 12, 4096, 513])
        permute_583: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_726, [0, 2, 1, 3]);  view_726 = None
        slice_749: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_583, 1, 0, 256);  permute_583 = None
        slice_750: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_749, 3, 0, 257);  slice_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_24: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_131: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_24, -2);  iota_24 = None
        iota_25: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_132: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
        sub_49: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_131, unsqueeze_132);  unsqueeze_131 = unsqueeze_132 = None
        le_12: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_49, 0);  sub_49 = None
        full_default_48: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_48: "f16[256, 257]" = torch.ops.aten.where.self(le_12, full_default_48, full_default_49);  le_12 = full_default_48 = full_default_49 = None
        rev_24: "f16[256, 257]" = torch.ops.prims.rev.default(where_48, [0]);  where_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_133: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_24, 0);  rev_24 = None
        unsqueeze_134: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_24: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_134, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_221: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_24, torch.bool);  expand_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_56: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_581: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_56, [0, 2, 1, 3]);  full_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_724: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_25, [1, 12, 4096, 513]);  select_scatter_25 = None
        permute_580: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_724, [0, 2, 1, 3]);  view_724 = None
        slice_744: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_580, 1, 0, 256);  permute_580 = None
        slice_745: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_744, 3, 0, 257);  slice_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_49: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_221, permute_581, slice_745);  convert_element_type_221 = permute_581 = slice_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_76: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_750, where_49);  slice_750 = where_49 = None
        slice_scatter_139: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_751, copy_76, 3, 0, 257);  slice_751 = copy_76 = None
        slice_scatter_140: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_584, slice_scatter_139, 1, 0, 256);  permute_584 = slice_scatter_139 = None
        permute_585: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_140, [0, 2, 1, 3]);  slice_scatter_140 = None
        view_728: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_585, [12, 16, 256, 513]);  permute_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_735: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_728, [1, 12, 4096, 513])
        permute_593: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_735, [0, 2, 1, 3]);  view_735 = None
        slice_764: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_593, 1, -256, 9223372036854775807)
        view_734: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_728, [1, 12, 4096, 513])
        permute_592: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
        slice_762: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_592, 1, -256, 9223372036854775807);  permute_592 = None
        slice_763: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_762, 3, -257, 9223372036854775807);  slice_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_25: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_134, [1, 3]);  unsqueeze_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_25: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_25, [1, 256, 12, 257]);  rev_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_222: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_25, torch.bool);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_57: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_590: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_57, [0, 2, 1, 3]);  full_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_732: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_728, [1, 12, 4096, 513]);  view_728 = None
        permute_589: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None
        slice_757: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_589, 1, -256, 9223372036854775807);  permute_589 = None
        slice_758: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_757, 3, -257, 9223372036854775807);  slice_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_50: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_222, permute_590, slice_758);  convert_element_type_222 = permute_590 = slice_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_77: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_763, where_50);  slice_763 = where_50 = None
        slice_scatter_141: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_764, copy_77, 3, -257, 9223372036854775807);  slice_764 = copy_77 = None
        slice_scatter_142: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_593, slice_scatter_141, 1, -256, 9223372036854775807);  permute_593 = slice_scatter_141 = None
        permute_594: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_142, [0, 2, 1, 3]);  slice_scatter_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_619: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_594, [0, 2, 1, 3]);  permute_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_59: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_771: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_59, 1, 0, -1)
        slice_769: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_59, 1, 0, -1)
        slice_770: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_769, 3, 256, 9223372036854775807);  slice_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_58: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_596: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_58, [0, 2, 1, 3]);  full_58 = None
        view_738: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_596, [1, 4096, 1]);  permute_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_740: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_738, [1, 8, 512, 1]);  view_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_60: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_740, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_137: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_60, 4);  as_strided_60 = None
        permute_598: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_137, [0, 1, 2, 4, 3]);  unsqueeze_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_50: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_6: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_135: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_6, 2);  ne_6 = None
        unsqueeze_136: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_223: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_136, torch.float16)
        where_51: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_136, full_default_50, convert_element_type_223);  unsqueeze_136 = full_default_50 = convert_element_type_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_597: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_51, [0, 2, 1, 3]);  where_51 = None
        view_739: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_597, [1, 4096, 1]);  permute_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_741: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_739, [1, 8, 512, 1]);  view_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_61: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_741, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_138: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_61, 4);  as_strided_61 = None
        permute_599: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_138, [0, 1, 4, 2, 3]);  unsqueeze_138 = None
        mul_48: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_598, permute_599);  permute_598 = permute_599 = None
        view_742: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_48, [1, 15, 512, 512]);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_25: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_742, [0, 0, 0, 1], 0.0);  view_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_743: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_25, [1, 15, 512, 513]);  constant_pad_nd_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_767: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_743, 2, 0, 256)
        slice_768: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_767, 3, 0, 257);  slice_767 = None
        copy_78: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_770, slice_768);  slice_770 = slice_768 = None
        slice_scatter_143: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_771, copy_78, 3, 256, 9223372036854775807);  slice_771 = copy_78 = None
        slice_scatter_144: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_59, slice_scatter_143, 1, 0, -1);  full_59 = slice_scatter_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_173: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_144, 1, -1)
        select_172: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_144, 1, -1)
        slice_777: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_172, 2, 256, 9223372036854775807);  select_172 = None
        select_169: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_743, 1, -1)
        slice_774: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_169, 1, 256, 9223372036854775807);  select_169 = None
        slice_775: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_774, 2, 0, 257);  slice_774 = None
        copy_79: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_777, slice_775);  slice_777 = slice_775 = None
        slice_scatter_145: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_173, copy_79, 2, 256, 9223372036854775807);  select_173 = copy_79 = None
        select_scatter_26: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_144, slice_scatter_145, 1, -1);  slice_scatter_144 = slice_scatter_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_786: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_26, 1, 1, 9223372036854775807)
        slice_784: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_26, 1, 1, 9223372036854775807)
        slice_785: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_784, 3, 0, 256);  slice_784 = None
        slice_779: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_743, 2, -257, -1)
        slice_780: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_779, 3, 257, 9223372036854775807);  slice_779 = None
        copy_80: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_785, slice_780);  slice_785 = slice_780 = None
        slice_scatter_146: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_786, copy_80, 3, 0, 256);  slice_786 = copy_80 = None
        slice_scatter_147: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_26, slice_scatter_146, 1, 1, 9223372036854775807);  select_scatter_26 = slice_scatter_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_180: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_147, 1, 0)
        slice_796: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_180, 1, 1, 256)
        select_179: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_147, 1, 0)
        slice_794: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_179, 1, 1, 256);  select_179 = None
        slice_795: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_794, 2, 1, 256);  slice_794 = None
        select_175: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_743, 1, 0);  view_743 = None
        slice_789: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_175, 1, 0, 255);  select_175 = None
        slice_790: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_789, 2, -255, 9223372036854775807);  slice_789 = None
        copy_81: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_795, slice_790);  slice_795 = slice_790 = None
        slice_scatter_148: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_796, copy_81, 2, 1, 256);  slice_796 = copy_81 = None
        slice_scatter_149: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_180, slice_scatter_148, 1, 1, 256);  select_180 = slice_scatter_148 = None
        select_scatter_27: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_147, slice_scatter_149, 1, 0);  slice_scatter_147 = slice_scatter_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_751: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_27, [1, 1, 4096, 513])
        permute_607: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_751, [0, 2, 1, 3]);  view_751 = None
        slice_809: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_607, 1, 0, 256)
        view_750: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_27, [1, 1, 4096, 513])
        permute_606: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_750, [0, 2, 1, 3]);  view_750 = None
        slice_807: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_606, 1, 0, 256);  permute_606 = None
        slice_808: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_807, 3, 0, 257);  slice_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_26: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_139: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_26, -2);  iota_26 = None
        iota_27: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_140: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
        sub_51: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_139, unsqueeze_140);  unsqueeze_139 = unsqueeze_140 = None
        le_13: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_51, 0);  sub_51 = None
        full_default_51: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_52: "f16[256, 257]" = torch.ops.aten.where.self(le_13, full_default_51, full_default_52);  le_13 = full_default_51 = full_default_52 = None
        rev_26: "f16[256, 257]" = torch.ops.prims.rev.default(where_52, [0]);  where_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_141: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_26, 0);  rev_26 = None
        unsqueeze_142: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_26: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_142, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_224: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_26, torch.bool);  expand_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_53: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_748: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_27, [1, 1, 4096, 513]);  select_scatter_27 = None
        permute_603: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None
        slice_802: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_603, 1, 0, 256);  permute_603 = None
        slice_803: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_802, 3, 0, 257);  slice_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_53: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_224, full_default_53, slice_803);  convert_element_type_224 = full_default_53 = slice_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_82: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_808, where_53);  slice_808 = where_53 = None
        slice_scatter_150: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_809, copy_82, 3, 0, 257);  slice_809 = copy_82 = None
        slice_scatter_151: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_607, slice_scatter_150, 1, 0, 256);  permute_607 = slice_scatter_150 = None
        permute_608: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_151, [0, 2, 1, 3]);  slice_scatter_151 = None
        view_752: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_608, [1, 16, 256, 513]);  permute_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_759: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_752, [1, 1, 4096, 513])
        permute_616: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_759, [0, 2, 1, 3]);  view_759 = None
        slice_822: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_616, 1, -256, 9223372036854775807)
        view_758: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_752, [1, 1, 4096, 513])
        permute_615: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_758, [0, 2, 1, 3]);  view_758 = None
        slice_820: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_615, 1, -256, 9223372036854775807);  permute_615 = None
        slice_821: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_820, 3, -257, 9223372036854775807);  slice_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_27: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_142, [1, 3]);  unsqueeze_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_27: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_27, [1, 256, 1, 257]);  rev_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_225: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_27, torch.bool);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_54: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_756: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_752, [1, 1, 4096, 513]);  view_752 = None
        permute_612: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_756, [0, 2, 1, 3]);  view_756 = None
        slice_815: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_612, 1, -256, 9223372036854775807);  permute_612 = None
        slice_816: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_815, 3, -257, 9223372036854775807);  slice_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_54: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_225, full_default_54, slice_816);  convert_element_type_225 = full_default_54 = slice_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_83: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_821, where_54);  slice_821 = where_54 = None
        slice_scatter_152: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_822, copy_83, 3, -257, 9223372036854775807);  slice_822 = copy_83 = None
        slice_scatter_153: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_616, slice_scatter_152, 1, -256, 9223372036854775807);  permute_616 = slice_scatter_152 = None
        permute_617: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_153, [0, 2, 1, 3]);  slice_scatter_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_620: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_617, [0, 2, 1, 3]);  permute_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_68: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_619, permute_620);  permute_619 = permute_620 = None
        permute_621: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_68, [0, 2, 1, 3]);  add_68 = None
        permute_622: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_621, [0, 2, 1, 3]);  permute_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_78: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
        convert_element_type_226: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_78, torch.float32);  clone_78 = None
        amax_6: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_226, [-1], True)
        sub_52: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_226, amax_6);  convert_element_type_226 = amax_6 = None
        exp_6: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_7: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_67: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_55: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_144, full_default_55, div_67);  unsqueeze_144 = full_default_55 = div_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_227: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_55, torch.float16);  where_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_624: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_227, [0, 2, 1, 3]);  convert_element_type_227 = None
        view_768: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_624, [12, 16, 256, 513]);  permute_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_27: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_768, [0, 257], 0.0);  view_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_770: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_27, [12, 16, -1]);  constant_pad_nd_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_825: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_770, 2, 0, -256);  view_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_771: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_825, [12, 16, 256, 769]);  slice_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_826: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_771, 3, 0, -1);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_145: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_826, 4);  slice_826 = None
        view_772: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_145, [192, 256, 768]);  unsqueeze_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_676: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_546, [4096, 768]);  permute_546 = None
        permute_549: "f16[768, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_38: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg104_1, view_676, permute_549);  arg104_1 = view_676 = permute_549 = None
        view_677: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_38, [4096, 1, 768]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_767: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_677, [4096, 1, 12, 64]);  view_677 = None
        permute_623: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_767, [1, 0, 2, 3]);  view_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_625: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_623, [0, 2, 1, 3]);  permute_623 = None
        view_769: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_625, [12, 4096, 64]);  permute_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_26: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_769, [0, 0, 256, 256], -1.0);  view_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_62: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_26, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_146: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_62, 4);  as_strided_62 = None
        clone_80: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_146, memory_format = torch.contiguous_format);  unsqueeze_146 = None
        view_773: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_80, [192, 768, 64]);  clone_80 = None
        bmm_13: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_772, view_773);  view_772 = view_773 = None
        view_774: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_13, [12, 16, 256, 1, 64]);  bmm_13 = None
        permute_630: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_774, [0, 1, 2, 4, 3]);  view_774 = None
        view_775: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_630, [12, 16, 256, 64]);  permute_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_776: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_775, [1, 12, 4096, 64]);  view_775 = None
        permute_631: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_776, [0, 2, 1, 3]);  view_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_632: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_631, [1, 0, 2, 3]);  permute_631 = None
        clone_81: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_632, memory_format = torch.contiguous_format);  permute_632 = None
        view_777: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_81, [4096, 1, 768]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_633: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_777, [1, 0, 2]);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_778: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_633, [4096, 768]);  permute_633 = None
        permute_634: "f16[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_39: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg106_1, view_778, permute_634);  arg106_1 = view_778 = permute_634 = None
        view_779: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_39, [1, 4096, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_70: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_779, convert_element_type_209);  view_779 = convert_element_type_209 = None
        convert_element_type_233: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_70, torch.float32);  add_70 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_233, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 4096, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 4096, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_54: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_233, getitem_25);  convert_element_type_233 = getitem_25 = None
        add_71: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_49: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_12);  sub_54 = rsqrt_12 = None
        mul_50: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg107_1);  mul_49 = arg107_1 = None
        add_72: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_50, arg108_1);  mul_50 = arg108_1 = None
        convert_element_type_234: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_72, torch.float16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_780: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_234, [4096, 768])
        permute_635: "f16[768, 3072]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_40: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg110_1, view_780, permute_635);  arg110_1 = view_780 = permute_635 = None
        view_781: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_40, [1, 4096, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_238: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_781, torch.float32);  view_781 = None
        mul_51: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_238, 0.5)
        mul_52: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_238, 0.7071067811865476);  convert_element_type_238 = None
        erf_6: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_73: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_53: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_73);  mul_51 = add_73 = None
        convert_element_type_239: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_53, torch.float16);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_782: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_239, [4096, 3072]);  convert_element_type_239 = None
        permute_636: "f16[3072, 768]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_41: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg112_1, view_782, permute_636);  arg112_1 = view_782 = permute_636 = None
        view_783: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_41, [1, 4096, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_74: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_783, convert_element_type_234);  view_783 = convert_element_type_234 = None
        convert_element_type_243: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_74, torch.float32);  add_74 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_243, [2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 4096, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 4096, 1]" = var_mean_13[1];  var_mean_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_70: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_71: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_164: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_165: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        full_default_63: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_63: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_831: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_63, 1, 0, -1)
        slice_829: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_63, 1, 0, -1)
        slice_830: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_829, 3, 256, 9223372036854775807);  slice_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_55: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_243, getitem_27);  convert_element_type_243 = getitem_27 = None
        add_75: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_54: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_13);  sub_55 = rsqrt_13 = None
        mul_55: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg113_1);  mul_54 = arg113_1 = None
        add_76: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_55, arg114_1);  mul_55 = arg114_1 = None
        convert_element_type_244: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_76, torch.float16);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_637: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_784: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_637, [4096, 768])
        permute_638: "f16[768, 768]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_42: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg116_1, view_784, permute_638);  arg116_1 = view_784 = permute_638 = None
        view_785: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_42, [4096, 1, 768]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_70: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_785, 8.0);  view_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_824: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_70, [4096, 1, 12, 64]);  div_70 = None
        permute_662: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_824, [1, 0, 2, 3]);  view_824 = None
        permute_663: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_662, [0, 2, 1, 3]);  permute_662 = None
        view_825: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_663, [12, 4096, 64]);  permute_663 = None
        view_826: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_825, [12, 8, 512, 64]);  view_825 = None
        as_strided_68: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_826, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_826 = None
        unsqueeze_151: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_68, 4);  as_strided_68 = None
        clone_84: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_151, memory_format = torch.contiguous_format);  unsqueeze_151 = None
        view_827: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_84, [180, 512, 64]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_786: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_637, [4096, 768])
        permute_639: "f16[768, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_43: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg118_1, view_786, permute_639);  arg118_1 = view_786 = permute_639 = None
        view_787: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_43, [4096, 1, 768]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_795: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_787, [4096, 1, 12, 64]);  view_787 = None
        permute_642: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_795, [1, 0, 2, 3]);  view_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_647: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_642, [0, 2, 1, 3]);  permute_642 = None
        view_801: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_647, [12, 4096, 64]);  permute_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_810: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_801, [12, 8, 512, 64]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_64: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_810, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_149: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_64, 4);  as_strided_64 = None
        permute_657: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_149, [0, 1, 4, 2, 3]);  unsqueeze_149 = None
        permute_666: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_657, [0, 1, 4, 3, 2]);  permute_657 = None
        clone_85: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_666, memory_format = torch.contiguous_format);  permute_666 = None
        view_828: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_85, [180, 64, 512]);  clone_85 = None
        bmm_14: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_827, view_828);  view_827 = view_828 = None
        view_829: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_14, [12, 15, 512, 1, 512]);  bmm_14 = None
        permute_667: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_829, [0, 1, 2, 4, 3]);  view_829 = None
        view_830: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_667, [12, 15, 512, 512]);  permute_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_28: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_830, [0, 0, 0, 1], 0.0);  view_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_831: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_28, [12, 15, 512, 513]);  constant_pad_nd_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_827: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_831, 2, 0, 256)
        slice_828: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_827, 3, 0, 257);  slice_827 = None
        copy_84: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_830, slice_828);  slice_830 = slice_828 = None
        slice_scatter_154: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_831, copy_84, 3, 256, 9223372036854775807);  slice_831 = copy_84 = None
        slice_scatter_155: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_63, slice_scatter_154, 1, 0, -1);  full_63 = slice_scatter_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_186: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_155, 1, -1)
        select_185: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_155, 1, -1)
        slice_837: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_185, 2, 256, 9223372036854775807);  select_185 = None
        select_182: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_831, 1, -1)
        slice_834: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_182, 1, 256, 9223372036854775807);  select_182 = None
        slice_835: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_834, 2, 0, 257);  slice_834 = None
        copy_85: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_837, slice_835);  slice_837 = slice_835 = None
        slice_scatter_156: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_186, copy_85, 2, 256, 9223372036854775807);  select_186 = copy_85 = None
        select_scatter_28: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_155, slice_scatter_156, 1, -1);  slice_scatter_155 = slice_scatter_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_846: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_28, 1, 1, 9223372036854775807)
        slice_844: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_28, 1, 1, 9223372036854775807)
        slice_845: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_844, 3, 0, 256);  slice_844 = None
        slice_839: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_831, 2, -257, -1)
        slice_840: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_839, 3, 257, 9223372036854775807);  slice_839 = None
        copy_86: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_845, slice_840);  slice_845 = slice_840 = None
        slice_scatter_157: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_846, copy_86, 3, 0, 256);  slice_846 = copy_86 = None
        slice_scatter_158: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_28, slice_scatter_157, 1, 1, 9223372036854775807);  select_scatter_28 = slice_scatter_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_193: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_158, 1, 0)
        slice_856: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_193, 1, 1, 256)
        select_192: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_158, 1, 0)
        slice_854: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_192, 1, 1, 256);  select_192 = None
        slice_855: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_854, 2, 1, 256);  slice_854 = None
        select_188: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_831, 1, 0);  view_831 = None
        slice_849: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_188, 1, 0, 255);  select_188 = None
        slice_850: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_849, 2, -255, 9223372036854775807);  slice_849 = None
        copy_87: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_855, slice_850);  slice_855 = slice_850 = None
        slice_scatter_159: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_856, copy_87, 2, 1, 256);  slice_856 = copy_87 = None
        slice_scatter_160: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_193, slice_scatter_159, 1, 1, 256);  select_193 = slice_scatter_159 = None
        select_scatter_29: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_158, slice_scatter_160, 1, 0);  slice_scatter_158 = slice_scatter_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_839: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_29, [1, 12, 4096, 513])
        permute_675: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_839, [0, 2, 1, 3]);  view_839 = None
        slice_869: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_675, 1, 0, 256)
        view_838: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_29, [1, 12, 4096, 513])
        permute_674: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_838, [0, 2, 1, 3]);  view_838 = None
        slice_867: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_674, 1, 0, 256);  permute_674 = None
        slice_868: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_867, 3, 0, 257);  slice_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_28: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_152: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_28, -2);  iota_28 = None
        iota_29: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_153: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
        sub_57: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_152, unsqueeze_153);  unsqueeze_152 = unsqueeze_153 = None
        le_14: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_57, 0);  sub_57 = None
        full_default_56: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_56: "f16[256, 257]" = torch.ops.aten.where.self(le_14, full_default_56, full_default_57);  le_14 = full_default_56 = full_default_57 = None
        rev_28: "f16[256, 257]" = torch.ops.prims.rev.default(where_56, [0]);  where_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_154: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_28, 0);  rev_28 = None
        unsqueeze_155: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_28: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_155, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_256: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_28, torch.bool);  expand_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_65: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_672: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_65, [0, 2, 1, 3]);  full_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_836: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_29, [1, 12, 4096, 513]);  select_scatter_29 = None
        permute_671: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_836, [0, 2, 1, 3]);  view_836 = None
        slice_862: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_671, 1, 0, 256);  permute_671 = None
        slice_863: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_862, 3, 0, 257);  slice_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_57: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_256, permute_672, slice_863);  convert_element_type_256 = permute_672 = slice_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_88: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_868, where_57);  slice_868 = where_57 = None
        slice_scatter_161: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_869, copy_88, 3, 0, 257);  slice_869 = copy_88 = None
        slice_scatter_162: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_675, slice_scatter_161, 1, 0, 256);  permute_675 = slice_scatter_161 = None
        permute_676: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_162, [0, 2, 1, 3]);  slice_scatter_162 = None
        view_840: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_676, [12, 16, 256, 513]);  permute_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_847: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_840, [1, 12, 4096, 513])
        permute_684: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_847, [0, 2, 1, 3]);  view_847 = None
        slice_882: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_684, 1, -256, 9223372036854775807)
        view_846: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_840, [1, 12, 4096, 513])
        permute_683: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_846, [0, 2, 1, 3]);  view_846 = None
        slice_880: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_683, 1, -256, 9223372036854775807);  permute_683 = None
        slice_881: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_880, 3, -257, 9223372036854775807);  slice_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_29: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_155, [1, 3]);  unsqueeze_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_29: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_29, [1, 256, 12, 257]);  rev_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_257: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_29, torch.bool);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_66: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_681: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_66, [0, 2, 1, 3]);  full_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_844: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_840, [1, 12, 4096, 513]);  view_840 = None
        permute_680: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_844, [0, 2, 1, 3]);  view_844 = None
        slice_875: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_680, 1, -256, 9223372036854775807);  permute_680 = None
        slice_876: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_875, 3, -257, 9223372036854775807);  slice_875 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_58: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_257, permute_681, slice_876);  convert_element_type_257 = permute_681 = slice_876 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_89: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_881, where_58);  slice_881 = where_58 = None
        slice_scatter_163: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_882, copy_89, 3, -257, 9223372036854775807);  slice_882 = copy_89 = None
        slice_scatter_164: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_684, slice_scatter_163, 1, -256, 9223372036854775807);  permute_684 = slice_scatter_163 = None
        permute_685: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_164, [0, 2, 1, 3]);  slice_scatter_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_710: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_685, [0, 2, 1, 3]);  permute_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_68: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_889: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_68, 1, 0, -1)
        slice_887: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_68, 1, 0, -1)
        slice_888: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_887, 3, 256, 9223372036854775807);  slice_887 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_67: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_687: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_67, [0, 2, 1, 3]);  full_67 = None
        view_850: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_687, [1, 4096, 1]);  permute_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_852: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_850, [1, 8, 512, 1]);  view_850 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_69: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_852, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_158: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_69, 4);  as_strided_69 = None
        permute_689: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_158, [0, 1, 2, 4, 3]);  unsqueeze_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_58: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_7: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_156: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_7, 2);  ne_7 = None
        unsqueeze_157: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_258: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_157, torch.float16)
        where_59: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_157, full_default_58, convert_element_type_258);  unsqueeze_157 = full_default_58 = convert_element_type_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_688: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_59, [0, 2, 1, 3]);  where_59 = None
        view_851: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_688, [1, 4096, 1]);  permute_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_853: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_851, [1, 8, 512, 1]);  view_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_70: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_853, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_159: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_70, 4);  as_strided_70 = None
        permute_690: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_159, [0, 1, 4, 2, 3]);  unsqueeze_159 = None
        mul_56: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_689, permute_690);  permute_689 = permute_690 = None
        view_854: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_56, [1, 15, 512, 512]);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_29: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_854, [0, 0, 0, 1], 0.0);  view_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_855: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_29, [1, 15, 512, 513]);  constant_pad_nd_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_885: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_855, 2, 0, 256)
        slice_886: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_885, 3, 0, 257);  slice_885 = None
        copy_90: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_888, slice_886);  slice_888 = slice_886 = None
        slice_scatter_165: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_889, copy_90, 3, 256, 9223372036854775807);  slice_889 = copy_90 = None
        slice_scatter_166: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_68, slice_scatter_165, 1, 0, -1);  full_68 = slice_scatter_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_199: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_166, 1, -1)
        select_198: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_166, 1, -1)
        slice_895: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_198, 2, 256, 9223372036854775807);  select_198 = None
        select_195: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_855, 1, -1)
        slice_892: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_195, 1, 256, 9223372036854775807);  select_195 = None
        slice_893: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_892, 2, 0, 257);  slice_892 = None
        copy_91: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_895, slice_893);  slice_895 = slice_893 = None
        slice_scatter_167: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_199, copy_91, 2, 256, 9223372036854775807);  select_199 = copy_91 = None
        select_scatter_30: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_166, slice_scatter_167, 1, -1);  slice_scatter_166 = slice_scatter_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_904: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_30, 1, 1, 9223372036854775807)
        slice_902: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_30, 1, 1, 9223372036854775807)
        slice_903: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_902, 3, 0, 256);  slice_902 = None
        slice_897: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_855, 2, -257, -1)
        slice_898: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_897, 3, 257, 9223372036854775807);  slice_897 = None
        copy_92: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_903, slice_898);  slice_903 = slice_898 = None
        slice_scatter_168: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_904, copy_92, 3, 0, 256);  slice_904 = copy_92 = None
        slice_scatter_169: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_30, slice_scatter_168, 1, 1, 9223372036854775807);  select_scatter_30 = slice_scatter_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_206: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_169, 1, 0)
        slice_914: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_206, 1, 1, 256)
        select_205: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_169, 1, 0)
        slice_912: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_205, 1, 1, 256);  select_205 = None
        slice_913: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_912, 2, 1, 256);  slice_912 = None
        select_201: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_855, 1, 0);  view_855 = None
        slice_907: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_201, 1, 0, 255);  select_201 = None
        slice_908: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_907, 2, -255, 9223372036854775807);  slice_907 = None
        copy_93: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_913, slice_908);  slice_913 = slice_908 = None
        slice_scatter_170: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_914, copy_93, 2, 1, 256);  slice_914 = copy_93 = None
        slice_scatter_171: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_206, slice_scatter_170, 1, 1, 256);  select_206 = slice_scatter_170 = None
        select_scatter_31: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_169, slice_scatter_171, 1, 0);  slice_scatter_169 = slice_scatter_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_863: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_31, [1, 1, 4096, 513])
        permute_698: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_863, [0, 2, 1, 3]);  view_863 = None
        slice_927: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_698, 1, 0, 256)
        view_862: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_31, [1, 1, 4096, 513])
        permute_697: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_862, [0, 2, 1, 3]);  view_862 = None
        slice_925: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_697, 1, 0, 256);  permute_697 = None
        slice_926: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_925, 3, 0, 257);  slice_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_30: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_160: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_30, -2);  iota_30 = None
        iota_31: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_161: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
        sub_59: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_160, unsqueeze_161);  unsqueeze_160 = unsqueeze_161 = None
        le_15: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_59, 0);  sub_59 = None
        full_default_59: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_60: "f16[256, 257]" = torch.ops.aten.where.self(le_15, full_default_59, full_default_60);  le_15 = full_default_59 = full_default_60 = None
        rev_30: "f16[256, 257]" = torch.ops.prims.rev.default(where_60, [0]);  where_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_162: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_30, 0);  rev_30 = None
        unsqueeze_163: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 2);  unsqueeze_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_30: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_163, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_259: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_30, torch.bool);  expand_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_61: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_860: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_31, [1, 1, 4096, 513]);  select_scatter_31 = None
        permute_694: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_860, [0, 2, 1, 3]);  view_860 = None
        slice_920: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_694, 1, 0, 256);  permute_694 = None
        slice_921: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_920, 3, 0, 257);  slice_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_61: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_259, full_default_61, slice_921);  convert_element_type_259 = full_default_61 = slice_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_94: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_926, where_61);  slice_926 = where_61 = None
        slice_scatter_172: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_927, copy_94, 3, 0, 257);  slice_927 = copy_94 = None
        slice_scatter_173: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_698, slice_scatter_172, 1, 0, 256);  permute_698 = slice_scatter_172 = None
        permute_699: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_173, [0, 2, 1, 3]);  slice_scatter_173 = None
        view_864: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_699, [1, 16, 256, 513]);  permute_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_871: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_864, [1, 1, 4096, 513])
        permute_707: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_871, [0, 2, 1, 3]);  view_871 = None
        slice_940: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_707, 1, -256, 9223372036854775807)
        view_870: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_864, [1, 1, 4096, 513])
        permute_706: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_870, [0, 2, 1, 3]);  view_870 = None
        slice_938: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_706, 1, -256, 9223372036854775807);  permute_706 = None
        slice_939: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_938, 3, -257, 9223372036854775807);  slice_938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_31: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_163, [1, 3]);  unsqueeze_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_31: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_31, [1, 256, 1, 257]);  rev_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_260: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_31, torch.bool);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_62: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_868: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_864, [1, 1, 4096, 513]);  view_864 = None
        permute_703: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_868, [0, 2, 1, 3]);  view_868 = None
        slice_933: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_703, 1, -256, 9223372036854775807);  permute_703 = None
        slice_934: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_933, 3, -257, 9223372036854775807);  slice_933 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_62: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_260, full_default_62, slice_934);  convert_element_type_260 = full_default_62 = slice_934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_95: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_939, where_62);  slice_939 = where_62 = None
        slice_scatter_174: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_940, copy_95, 3, -257, 9223372036854775807);  slice_940 = copy_95 = None
        slice_scatter_175: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_707, slice_scatter_174, 1, -256, 9223372036854775807);  permute_707 = slice_scatter_174 = None
        permute_708: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_175, [0, 2, 1, 3]);  slice_scatter_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_711: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_708, [0, 2, 1, 3]);  permute_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_79: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_710, permute_711);  permute_710 = permute_711 = None
        permute_712: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_79, [0, 2, 1, 3]);  add_79 = None
        permute_713: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_712, [0, 2, 1, 3]);  permute_712 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_90: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_713, memory_format = torch.contiguous_format);  permute_713 = None
        convert_element_type_261: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_90, torch.float32);  clone_90 = None
        amax_7: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_261, [-1], True)
        sub_60: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_261, amax_7);  convert_element_type_261 = amax_7 = None
        exp_7: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_8: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_77: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_63: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_165, full_default_63, div_77);  unsqueeze_165 = full_default_63 = div_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_262: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_63, torch.float16);  where_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_715: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_262, [0, 2, 1, 3]);  convert_element_type_262 = None
        view_880: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_715, [12, 16, 256, 513]);  permute_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_31: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_880, [0, 257], 0.0);  view_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_882: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_31, [12, 16, -1]);  constant_pad_nd_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_943: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_882, 2, 0, -256);  view_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_883: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_943, [12, 16, 256, 769]);  slice_943 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_944: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_883, 3, 0, -1);  view_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_166: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_944, 4);  slice_944 = None
        view_884: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_166, [192, 256, 768]);  unsqueeze_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_788: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_637, [4096, 768]);  permute_637 = None
        permute_640: "f16[768, 768]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_44: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg120_1, view_788, permute_640);  arg120_1 = view_788 = permute_640 = None
        view_789: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_44, [4096, 1, 768]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_879: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_789, [4096, 1, 12, 64]);  view_789 = None
        permute_714: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_879, [1, 0, 2, 3]);  view_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_716: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_714, [0, 2, 1, 3]);  permute_714 = None
        view_881: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_716, [12, 4096, 64]);  permute_716 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_30: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_881, [0, 0, 256, 256], -1.0);  view_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_71: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_30, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_167: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_71, 4);  as_strided_71 = None
        clone_92: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_167, memory_format = torch.contiguous_format);  unsqueeze_167 = None
        view_885: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_92, [192, 768, 64]);  clone_92 = None
        bmm_15: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_884, view_885);  view_884 = view_885 = None
        view_886: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_15, [12, 16, 256, 1, 64]);  bmm_15 = None
        permute_721: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_886, [0, 1, 2, 4, 3]);  view_886 = None
        view_887: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_721, [12, 16, 256, 64]);  permute_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_888: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_887, [1, 12, 4096, 64]);  view_887 = None
        permute_722: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_888, [0, 2, 1, 3]);  view_888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_723: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_722, [1, 0, 2, 3]);  permute_722 = None
        clone_93: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_723, memory_format = torch.contiguous_format);  permute_723 = None
        view_889: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_93, [4096, 1, 768]);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_724: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_889, [1, 0, 2]);  view_889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_890: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_724, [4096, 768]);  permute_724 = None
        permute_725: "f16[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_45: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg122_1, view_890, permute_725);  arg122_1 = view_890 = permute_725 = None
        view_891: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_45, [1, 4096, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_81: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_891, convert_element_type_244);  view_891 = convert_element_type_244 = None
        convert_element_type_268: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_268, [2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 4096, 1]" = var_mean_14[0]
        getitem_29: "f32[1, 4096, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_62: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_268, getitem_29);  convert_element_type_268 = getitem_29 = None
        add_82: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_57: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = rsqrt_14 = None
        mul_58: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg123_1);  mul_57 = arg123_1 = None
        add_83: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_58, arg124_1);  mul_58 = arg124_1 = None
        convert_element_type_269: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_83, torch.float16);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_892: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_269, [4096, 768])
        permute_726: "f16[768, 3072]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_46: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg126_1, view_892, permute_726);  arg126_1 = view_892 = permute_726 = None
        view_893: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_46, [1, 4096, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_273: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_893, torch.float32);  view_893 = None
        mul_59: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_273, 0.5)
        mul_60: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_273, 0.7071067811865476);  convert_element_type_273 = None
        erf_7: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_84: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_61: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_59, add_84);  mul_59 = add_84 = None
        convert_element_type_274: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_61, torch.float16);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_894: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_274, [4096, 3072]);  convert_element_type_274 = None
        permute_727: "f16[3072, 768]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_47: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg128_1, view_894, permute_727);  arg128_1 = view_894 = permute_727 = None
        view_895: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_47, [1, 4096, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_85: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_895, convert_element_type_269);  view_895 = convert_element_type_269 = None
        convert_element_type_278: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_85, torch.float32);  add_85 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_278, [2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 4096, 1]" = var_mean_15[0]
        getitem_31: "f32[1, 4096, 1]" = var_mean_15[1];  var_mean_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_79: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_80: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_185: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_186: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
        full_default_71: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_72: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_949: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_72, 1, 0, -1)
        slice_947: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_72, 1, 0, -1)
        slice_948: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_947, 3, 256, 9223372036854775807);  slice_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_63: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_278, getitem_31);  convert_element_type_278 = getitem_31 = None
        add_86: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_62: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_15);  sub_63 = rsqrt_15 = None
        mul_63: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_62, arg129_1);  mul_62 = arg129_1 = None
        add_87: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_63, arg130_1);  mul_63 = arg130_1 = None
        convert_element_type_279: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_87, torch.float16);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_728: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_279, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_896: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_728, [4096, 768])
        permute_729: "f16[768, 768]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_48: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg132_1, view_896, permute_729);  arg132_1 = view_896 = permute_729 = None
        view_897: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_48, [4096, 1, 768]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_80: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_897, 8.0);  view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_936: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_80, [4096, 1, 12, 64]);  div_80 = None
        permute_753: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_936, [1, 0, 2, 3]);  view_936 = None
        permute_754: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_753, [0, 2, 1, 3]);  permute_753 = None
        view_937: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_754, [12, 4096, 64]);  permute_754 = None
        view_938: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_937, [12, 8, 512, 64]);  view_937 = None
        as_strided_77: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_938, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_938 = None
        unsqueeze_172: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_77, 4);  as_strided_77 = None
        clone_96: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_172, memory_format = torch.contiguous_format);  unsqueeze_172 = None
        view_939: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_96, [180, 512, 64]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_898: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_728, [4096, 768])
        permute_730: "f16[768, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_49: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg134_1, view_898, permute_730);  arg134_1 = view_898 = permute_730 = None
        view_899: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_49, [4096, 1, 768]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_907: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_899, [4096, 1, 12, 64]);  view_899 = None
        permute_733: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_907, [1, 0, 2, 3]);  view_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_738: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_733, [0, 2, 1, 3]);  permute_733 = None
        view_913: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_738, [12, 4096, 64]);  permute_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_922: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_913, [12, 8, 512, 64]);  view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_73: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_922, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_922 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_170: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_73, 4);  as_strided_73 = None
        permute_748: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_170, [0, 1, 4, 2, 3]);  unsqueeze_170 = None
        permute_757: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_748, [0, 1, 4, 3, 2]);  permute_748 = None
        clone_97: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_757, memory_format = torch.contiguous_format);  permute_757 = None
        view_940: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_97, [180, 64, 512]);  clone_97 = None
        bmm_16: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_939, view_940);  view_939 = view_940 = None
        view_941: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_16, [12, 15, 512, 1, 512]);  bmm_16 = None
        permute_758: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_941, [0, 1, 2, 4, 3]);  view_941 = None
        view_942: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_758, [12, 15, 512, 512]);  permute_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_32: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_942, [0, 0, 0, 1], 0.0);  view_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_943: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_32, [12, 15, 512, 513]);  constant_pad_nd_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_945: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_943, 2, 0, 256)
        slice_946: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_945, 3, 0, 257);  slice_945 = None
        copy_96: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_948, slice_946);  slice_948 = slice_946 = None
        slice_scatter_176: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_949, copy_96, 3, 256, 9223372036854775807);  slice_949 = copy_96 = None
        slice_scatter_177: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_72, slice_scatter_176, 1, 0, -1);  full_72 = slice_scatter_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_212: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_177, 1, -1)
        select_211: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_177, 1, -1)
        slice_955: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_211, 2, 256, 9223372036854775807);  select_211 = None
        select_208: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_943, 1, -1)
        slice_952: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_208, 1, 256, 9223372036854775807);  select_208 = None
        slice_953: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_952, 2, 0, 257);  slice_952 = None
        copy_97: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_955, slice_953);  slice_955 = slice_953 = None
        slice_scatter_178: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_212, copy_97, 2, 256, 9223372036854775807);  select_212 = copy_97 = None
        select_scatter_32: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_177, slice_scatter_178, 1, -1);  slice_scatter_177 = slice_scatter_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_964: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_32, 1, 1, 9223372036854775807)
        slice_962: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_32, 1, 1, 9223372036854775807)
        slice_963: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_962, 3, 0, 256);  slice_962 = None
        slice_957: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_943, 2, -257, -1)
        slice_958: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_957, 3, 257, 9223372036854775807);  slice_957 = None
        copy_98: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_963, slice_958);  slice_963 = slice_958 = None
        slice_scatter_179: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_964, copy_98, 3, 0, 256);  slice_964 = copy_98 = None
        slice_scatter_180: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_32, slice_scatter_179, 1, 1, 9223372036854775807);  select_scatter_32 = slice_scatter_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_219: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_180, 1, 0)
        slice_974: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_219, 1, 1, 256)
        select_218: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_180, 1, 0)
        slice_972: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_218, 1, 1, 256);  select_218 = None
        slice_973: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_972, 2, 1, 256);  slice_972 = None
        select_214: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_943, 1, 0);  view_943 = None
        slice_967: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_214, 1, 0, 255);  select_214 = None
        slice_968: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_967, 2, -255, 9223372036854775807);  slice_967 = None
        copy_99: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_973, slice_968);  slice_973 = slice_968 = None
        slice_scatter_181: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_974, copy_99, 2, 1, 256);  slice_974 = copy_99 = None
        slice_scatter_182: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_219, slice_scatter_181, 1, 1, 256);  select_219 = slice_scatter_181 = None
        select_scatter_33: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_180, slice_scatter_182, 1, 0);  slice_scatter_180 = slice_scatter_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_951: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_33, [1, 12, 4096, 513])
        permute_766: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_951, [0, 2, 1, 3]);  view_951 = None
        slice_987: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_766, 1, 0, 256)
        view_950: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_33, [1, 12, 4096, 513])
        permute_765: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_950, [0, 2, 1, 3]);  view_950 = None
        slice_985: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_765, 1, 0, 256);  permute_765 = None
        slice_986: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_985, 3, 0, 257);  slice_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_32: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_173: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_32, -2);  iota_32 = None
        iota_33: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_174: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
        sub_65: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_173, unsqueeze_174);  unsqueeze_173 = unsqueeze_174 = None
        le_16: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_65, 0);  sub_65 = None
        full_default_64: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_65: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_64: "f16[256, 257]" = torch.ops.aten.where.self(le_16, full_default_64, full_default_65);  le_16 = full_default_64 = full_default_65 = None
        rev_32: "f16[256, 257]" = torch.ops.prims.rev.default(where_64, [0]);  where_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_175: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_32, 0);  rev_32 = None
        unsqueeze_176: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_32: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_176, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_291: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_32, torch.bool);  expand_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_74: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_763: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_74, [0, 2, 1, 3]);  full_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_948: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_33, [1, 12, 4096, 513]);  select_scatter_33 = None
        permute_762: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_948, [0, 2, 1, 3]);  view_948 = None
        slice_980: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_762, 1, 0, 256);  permute_762 = None
        slice_981: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_980, 3, 0, 257);  slice_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_65: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_291, permute_763, slice_981);  convert_element_type_291 = permute_763 = slice_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_100: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_986, where_65);  slice_986 = where_65 = None
        slice_scatter_183: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_987, copy_100, 3, 0, 257);  slice_987 = copy_100 = None
        slice_scatter_184: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_766, slice_scatter_183, 1, 0, 256);  permute_766 = slice_scatter_183 = None
        permute_767: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_184, [0, 2, 1, 3]);  slice_scatter_184 = None
        view_952: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_767, [12, 16, 256, 513]);  permute_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_959: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_952, [1, 12, 4096, 513])
        permute_775: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_959, [0, 2, 1, 3]);  view_959 = None
        slice_1000: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_775, 1, -256, 9223372036854775807)
        view_958: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_952, [1, 12, 4096, 513])
        permute_774: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_958, [0, 2, 1, 3]);  view_958 = None
        slice_998: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_774, 1, -256, 9223372036854775807);  permute_774 = None
        slice_999: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_998, 3, -257, 9223372036854775807);  slice_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_33: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_176, [1, 3]);  unsqueeze_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_33: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_33, [1, 256, 12, 257]);  rev_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_292: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_33, torch.bool);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_75: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_772: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_75, [0, 2, 1, 3]);  full_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_956: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_952, [1, 12, 4096, 513]);  view_952 = None
        permute_771: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_956, [0, 2, 1, 3]);  view_956 = None
        slice_993: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_771, 1, -256, 9223372036854775807);  permute_771 = None
        slice_994: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_993, 3, -257, 9223372036854775807);  slice_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_66: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_292, permute_772, slice_994);  convert_element_type_292 = permute_772 = slice_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_101: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_999, where_66);  slice_999 = where_66 = None
        slice_scatter_185: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1000, copy_101, 3, -257, 9223372036854775807);  slice_1000 = copy_101 = None
        slice_scatter_186: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_775, slice_scatter_185, 1, -256, 9223372036854775807);  permute_775 = slice_scatter_185 = None
        permute_776: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_186, [0, 2, 1, 3]);  slice_scatter_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_801: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_776, [0, 2, 1, 3]);  permute_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_77: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1007: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_77, 1, 0, -1)
        slice_1005: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_77, 1, 0, -1)
        slice_1006: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1005, 3, 256, 9223372036854775807);  slice_1005 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_76: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_778: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_76, [0, 2, 1, 3]);  full_76 = None
        view_962: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_778, [1, 4096, 1]);  permute_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_964: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_962, [1, 8, 512, 1]);  view_962 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_78: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_964, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_964 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_179: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_78, 4);  as_strided_78 = None
        permute_780: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_179, [0, 1, 2, 4, 3]);  unsqueeze_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_66: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_8: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_177: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_8, 2);  ne_8 = None
        unsqueeze_178: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_293: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_178, torch.float16)
        where_67: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_178, full_default_66, convert_element_type_293);  unsqueeze_178 = full_default_66 = convert_element_type_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_779: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_67, [0, 2, 1, 3]);  where_67 = None
        view_963: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_779, [1, 4096, 1]);  permute_779 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_965: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_963, [1, 8, 512, 1]);  view_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_79: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_965, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_965 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_180: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_79, 4);  as_strided_79 = None
        permute_781: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_180, [0, 1, 4, 2, 3]);  unsqueeze_180 = None
        mul_64: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_780, permute_781);  permute_780 = permute_781 = None
        view_966: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_64, [1, 15, 512, 512]);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_33: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_966, [0, 0, 0, 1], 0.0);  view_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_967: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_33, [1, 15, 512, 513]);  constant_pad_nd_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1003: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_967, 2, 0, 256)
        slice_1004: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1003, 3, 0, 257);  slice_1003 = None
        copy_102: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_1006, slice_1004);  slice_1006 = slice_1004 = None
        slice_scatter_187: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1007, copy_102, 3, 256, 9223372036854775807);  slice_1007 = copy_102 = None
        slice_scatter_188: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_77, slice_scatter_187, 1, 0, -1);  full_77 = slice_scatter_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_225: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_188, 1, -1)
        select_224: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_188, 1, -1)
        slice_1013: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_224, 2, 256, 9223372036854775807);  select_224 = None
        select_221: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_967, 1, -1)
        slice_1010: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_221, 1, 256, 9223372036854775807);  select_221 = None
        slice_1011: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1010, 2, 0, 257);  slice_1010 = None
        copy_103: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_1013, slice_1011);  slice_1013 = slice_1011 = None
        slice_scatter_189: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_225, copy_103, 2, 256, 9223372036854775807);  select_225 = copy_103 = None
        select_scatter_34: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_188, slice_scatter_189, 1, -1);  slice_scatter_188 = slice_scatter_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1022: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_34, 1, 1, 9223372036854775807)
        slice_1020: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_34, 1, 1, 9223372036854775807)
        slice_1021: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1020, 3, 0, 256);  slice_1020 = None
        slice_1015: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_967, 2, -257, -1)
        slice_1016: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1015, 3, 257, 9223372036854775807);  slice_1015 = None
        copy_104: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_1021, slice_1016);  slice_1021 = slice_1016 = None
        slice_scatter_190: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1022, copy_104, 3, 0, 256);  slice_1022 = copy_104 = None
        slice_scatter_191: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_34, slice_scatter_190, 1, 1, 9223372036854775807);  select_scatter_34 = slice_scatter_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_232: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_191, 1, 0)
        slice_1032: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_232, 1, 1, 256)
        select_231: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_191, 1, 0)
        slice_1030: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_231, 1, 1, 256);  select_231 = None
        slice_1031: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1030, 2, 1, 256);  slice_1030 = None
        select_227: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_967, 1, 0);  view_967 = None
        slice_1025: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_227, 1, 0, 255);  select_227 = None
        slice_1026: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1025, 2, -255, 9223372036854775807);  slice_1025 = None
        copy_105: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_1031, slice_1026);  slice_1031 = slice_1026 = None
        slice_scatter_192: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1032, copy_105, 2, 1, 256);  slice_1032 = copy_105 = None
        slice_scatter_193: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_232, slice_scatter_192, 1, 1, 256);  select_232 = slice_scatter_192 = None
        select_scatter_35: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_191, slice_scatter_193, 1, 0);  slice_scatter_191 = slice_scatter_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_975: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_35, [1, 1, 4096, 513])
        permute_789: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_975, [0, 2, 1, 3]);  view_975 = None
        slice_1045: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_789, 1, 0, 256)
        view_974: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_35, [1, 1, 4096, 513])
        permute_788: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_974, [0, 2, 1, 3]);  view_974 = None
        slice_1043: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_788, 1, 0, 256);  permute_788 = None
        slice_1044: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1043, 3, 0, 257);  slice_1043 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_34: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_181: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_34, -2);  iota_34 = None
        iota_35: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_182: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
        sub_67: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_181, unsqueeze_182);  unsqueeze_181 = unsqueeze_182 = None
        le_17: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_67, 0);  sub_67 = None
        full_default_67: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_68: "f16[256, 257]" = torch.ops.aten.where.self(le_17, full_default_67, full_default_68);  le_17 = full_default_67 = full_default_68 = None
        rev_34: "f16[256, 257]" = torch.ops.prims.rev.default(where_68, [0]);  where_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_183: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_34, 0);  rev_34 = None
        unsqueeze_184: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_34: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_184, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_294: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_34, torch.bool);  expand_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_69: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_972: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_35, [1, 1, 4096, 513]);  select_scatter_35 = None
        permute_785: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_972, [0, 2, 1, 3]);  view_972 = None
        slice_1038: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_785, 1, 0, 256);  permute_785 = None
        slice_1039: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1038, 3, 0, 257);  slice_1038 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_69: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_294, full_default_69, slice_1039);  convert_element_type_294 = full_default_69 = slice_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_106: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1044, where_69);  slice_1044 = where_69 = None
        slice_scatter_194: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1045, copy_106, 3, 0, 257);  slice_1045 = copy_106 = None
        slice_scatter_195: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_789, slice_scatter_194, 1, 0, 256);  permute_789 = slice_scatter_194 = None
        permute_790: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_195, [0, 2, 1, 3]);  slice_scatter_195 = None
        view_976: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_790, [1, 16, 256, 513]);  permute_790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_983: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_976, [1, 1, 4096, 513])
        permute_798: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_983, [0, 2, 1, 3]);  view_983 = None
        slice_1058: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_798, 1, -256, 9223372036854775807)
        view_982: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_976, [1, 1, 4096, 513])
        permute_797: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_982, [0, 2, 1, 3]);  view_982 = None
        slice_1056: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_797, 1, -256, 9223372036854775807);  permute_797 = None
        slice_1057: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1056, 3, -257, 9223372036854775807);  slice_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_35: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_184, [1, 3]);  unsqueeze_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_35: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_35, [1, 256, 1, 257]);  rev_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_295: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_35, torch.bool);  expand_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_70: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_980: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_976, [1, 1, 4096, 513]);  view_976 = None
        permute_794: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_980, [0, 2, 1, 3]);  view_980 = None
        slice_1051: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_794, 1, -256, 9223372036854775807);  permute_794 = None
        slice_1052: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1051, 3, -257, 9223372036854775807);  slice_1051 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_70: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_295, full_default_70, slice_1052);  convert_element_type_295 = full_default_70 = slice_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_107: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1057, where_70);  slice_1057 = where_70 = None
        slice_scatter_196: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1058, copy_107, 3, -257, 9223372036854775807);  slice_1058 = copy_107 = None
        slice_scatter_197: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_798, slice_scatter_196, 1, -256, 9223372036854775807);  permute_798 = slice_scatter_196 = None
        permute_799: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_197, [0, 2, 1, 3]);  slice_scatter_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_802: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_799, [0, 2, 1, 3]);  permute_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_90: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_801, permute_802);  permute_801 = permute_802 = None
        permute_803: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_90, [0, 2, 1, 3]);  add_90 = None
        permute_804: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_803, [0, 2, 1, 3]);  permute_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_102: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_804, memory_format = torch.contiguous_format);  permute_804 = None
        convert_element_type_296: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_102, torch.float32);  clone_102 = None
        amax_8: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_296, [-1], True)
        sub_68: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_296, amax_8);  convert_element_type_296 = amax_8 = None
        exp_8: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
        sum_9: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_87: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_71: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_186, full_default_71, div_87);  unsqueeze_186 = full_default_71 = div_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_297: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_71, torch.float16);  where_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_806: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_297, [0, 2, 1, 3]);  convert_element_type_297 = None
        view_992: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_806, [12, 16, 256, 513]);  permute_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_35: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_992, [0, 257], 0.0);  view_992 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_994: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_35, [12, 16, -1]);  constant_pad_nd_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1061: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_994, 2, 0, -256);  view_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_995: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_1061, [12, 16, 256, 769]);  slice_1061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1062: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_995, 3, 0, -1);  view_995 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_187: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1062, 4);  slice_1062 = None
        view_996: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_187, [192, 256, 768]);  unsqueeze_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_900: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_728, [4096, 768]);  permute_728 = None
        permute_731: "f16[768, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_50: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg136_1, view_900, permute_731);  arg136_1 = view_900 = permute_731 = None
        view_901: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_50, [4096, 1, 768]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_991: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_901, [4096, 1, 12, 64]);  view_901 = None
        permute_805: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_991, [1, 0, 2, 3]);  view_991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_807: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_805, [0, 2, 1, 3]);  permute_805 = None
        view_993: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_807, [12, 4096, 64]);  permute_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_34: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_993, [0, 0, 256, 256], -1.0);  view_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_80: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_34, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_188: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_80, 4);  as_strided_80 = None
        clone_104: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_188, memory_format = torch.contiguous_format);  unsqueeze_188 = None
        view_997: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_104, [192, 768, 64]);  clone_104 = None
        bmm_17: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_996, view_997);  view_996 = view_997 = None
        view_998: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_17, [12, 16, 256, 1, 64]);  bmm_17 = None
        permute_812: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_998, [0, 1, 2, 4, 3]);  view_998 = None
        view_999: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_812, [12, 16, 256, 64]);  permute_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1000: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_999, [1, 12, 4096, 64]);  view_999 = None
        permute_813: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1000, [0, 2, 1, 3]);  view_1000 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_814: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_813, [1, 0, 2, 3]);  permute_813 = None
        clone_105: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_814, memory_format = torch.contiguous_format);  permute_814 = None
        view_1001: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_105, [4096, 1, 768]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_815: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_1001, [1, 0, 2]);  view_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_1002: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_815, [4096, 768]);  permute_815 = None
        permute_816: "f16[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_51: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg138_1, view_1002, permute_816);  arg138_1 = view_1002 = permute_816 = None
        view_1003: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_51, [1, 4096, 768]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_92: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_1003, convert_element_type_279);  view_1003 = convert_element_type_279 = None
        convert_element_type_303: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_92, torch.float32);  add_92 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_303, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 4096, 1]" = var_mean_16[0]
        getitem_33: "f32[1, 4096, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_70: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_303, getitem_33);  convert_element_type_303 = getitem_33 = None
        add_93: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_65: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_16);  sub_70 = rsqrt_16 = None
        mul_66: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg139_1);  mul_65 = arg139_1 = None
        add_94: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_66, arg140_1);  mul_66 = arg140_1 = None
        convert_element_type_304: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_94, torch.float16);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1004: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_304, [4096, 768])
        permute_817: "f16[768, 3072]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_52: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg142_1, view_1004, permute_817);  arg142_1 = view_1004 = permute_817 = None
        view_1005: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_52, [1, 4096, 3072]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_308: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_1005, torch.float32);  view_1005 = None
        mul_67: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_308, 0.5)
        mul_68: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_308, 0.7071067811865476);  convert_element_type_308 = None
        erf_8: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_95: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_69: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_95);  mul_67 = add_95 = None
        convert_element_type_309: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_69, torch.float16);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1006: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_309, [4096, 3072]);  convert_element_type_309 = None
        permute_818: "f16[3072, 768]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_53: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg144_1, view_1006, permute_818);  arg144_1 = view_1006 = permute_818 = None
        view_1007: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_53, [1, 4096, 768]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_96: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_1007, convert_element_type_304);  view_1007 = convert_element_type_304 = None
        convert_element_type_313: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_96, torch.float32);  add_96 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_313, [2], correction = 0, keepdim = True)
        getitem_34: "f32[1, 4096, 1]" = var_mean_17[0]
        getitem_35: "f32[1, 4096, 1]" = var_mean_17[1];  var_mean_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_88: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_89: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_206: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_207: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
        full_default_79: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_81: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1067: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_81, 1, 0, -1)
        slice_1065: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_81, 1, 0, -1)
        slice_1066: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1065, 3, 256, 9223372036854775807);  slice_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_71: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_313, getitem_35);  convert_element_type_313 = getitem_35 = None
        add_97: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_70: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_17);  sub_71 = rsqrt_17 = None
        mul_71: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg145_1);  mul_70 = arg145_1 = None
        add_98: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_71, arg146_1);  mul_71 = arg146_1 = None
        convert_element_type_314: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_98, torch.float16);  add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_819: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_314, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_1008: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_819, [4096, 768])
        permute_820: "f16[768, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_54: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg148_1, view_1008, permute_820);  arg148_1 = view_1008 = permute_820 = None
        view_1009: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_54, [4096, 1, 768]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_90: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_1009, 8.0);  view_1009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1048: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_90, [4096, 1, 12, 64]);  div_90 = None
        permute_844: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1048, [1, 0, 2, 3]);  view_1048 = None
        permute_845: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_844, [0, 2, 1, 3]);  permute_844 = None
        view_1049: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_845, [12, 4096, 64]);  permute_845 = None
        view_1050: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_1049, [12, 8, 512, 64]);  view_1049 = None
        as_strided_86: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_1050, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_1050 = None
        unsqueeze_193: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_86, 4);  as_strided_86 = None
        clone_108: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_193, memory_format = torch.contiguous_format);  unsqueeze_193 = None
        view_1051: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_108, [180, 512, 64]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_1010: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_819, [4096, 768])
        permute_821: "f16[768, 768]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_55: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg150_1, view_1010, permute_821);  arg150_1 = view_1010 = permute_821 = None
        view_1011: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_55, [4096, 1, 768]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1019: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_1011, [4096, 1, 12, 64]);  view_1011 = None
        permute_824: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1019, [1, 0, 2, 3]);  view_1019 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_829: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_824, [0, 2, 1, 3]);  permute_824 = None
        view_1025: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_829, [12, 4096, 64]);  permute_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1034: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_1025, [12, 8, 512, 64]);  view_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_82: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_1034, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_1034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_191: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_82, 4);  as_strided_82 = None
        permute_839: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_191, [0, 1, 4, 2, 3]);  unsqueeze_191 = None
        permute_848: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_839, [0, 1, 4, 3, 2]);  permute_839 = None
        clone_109: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_848, memory_format = torch.contiguous_format);  permute_848 = None
        view_1052: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_109, [180, 64, 512]);  clone_109 = None
        bmm_18: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_1051, view_1052);  view_1051 = view_1052 = None
        view_1053: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_18, [12, 15, 512, 1, 512]);  bmm_18 = None
        permute_849: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_1053, [0, 1, 2, 4, 3]);  view_1053 = None
        view_1054: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_849, [12, 15, 512, 512]);  permute_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_36: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1054, [0, 0, 0, 1], 0.0);  view_1054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1055: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_36, [12, 15, 512, 513]);  constant_pad_nd_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1063: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1055, 2, 0, 256)
        slice_1064: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1063, 3, 0, 257);  slice_1063 = None
        copy_108: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_1066, slice_1064);  slice_1066 = slice_1064 = None
        slice_scatter_198: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1067, copy_108, 3, 256, 9223372036854775807);  slice_1067 = copy_108 = None
        slice_scatter_199: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_81, slice_scatter_198, 1, 0, -1);  full_81 = slice_scatter_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_238: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_199, 1, -1)
        select_237: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_199, 1, -1)
        slice_1073: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_237, 2, 256, 9223372036854775807);  select_237 = None
        select_234: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_1055, 1, -1)
        slice_1070: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_234, 1, 256, 9223372036854775807);  select_234 = None
        slice_1071: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1070, 2, 0, 257);  slice_1070 = None
        copy_109: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_1073, slice_1071);  slice_1073 = slice_1071 = None
        slice_scatter_200: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_238, copy_109, 2, 256, 9223372036854775807);  select_238 = copy_109 = None
        select_scatter_36: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_199, slice_scatter_200, 1, -1);  slice_scatter_199 = slice_scatter_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1082: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_36, 1, 1, 9223372036854775807)
        slice_1080: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_36, 1, 1, 9223372036854775807)
        slice_1081: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1080, 3, 0, 256);  slice_1080 = None
        slice_1075: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1055, 2, -257, -1)
        slice_1076: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1075, 3, 257, 9223372036854775807);  slice_1075 = None
        copy_110: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_1081, slice_1076);  slice_1081 = slice_1076 = None
        slice_scatter_201: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1082, copy_110, 3, 0, 256);  slice_1082 = copy_110 = None
        slice_scatter_202: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_36, slice_scatter_201, 1, 1, 9223372036854775807);  select_scatter_36 = slice_scatter_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_245: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_202, 1, 0)
        slice_1092: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_245, 1, 1, 256)
        select_244: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_202, 1, 0)
        slice_1090: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_244, 1, 1, 256);  select_244 = None
        slice_1091: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1090, 2, 1, 256);  slice_1090 = None
        select_240: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_1055, 1, 0);  view_1055 = None
        slice_1085: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_240, 1, 0, 255);  select_240 = None
        slice_1086: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1085, 2, -255, 9223372036854775807);  slice_1085 = None
        copy_111: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_1091, slice_1086);  slice_1091 = slice_1086 = None
        slice_scatter_203: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1092, copy_111, 2, 1, 256);  slice_1092 = copy_111 = None
        slice_scatter_204: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_245, slice_scatter_203, 1, 1, 256);  select_245 = slice_scatter_203 = None
        select_scatter_37: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_202, slice_scatter_204, 1, 0);  slice_scatter_202 = slice_scatter_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1063: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_37, [1, 12, 4096, 513])
        permute_857: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1063, [0, 2, 1, 3]);  view_1063 = None
        slice_1105: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_857, 1, 0, 256)
        view_1062: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_37, [1, 12, 4096, 513])
        permute_856: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1062, [0, 2, 1, 3]);  view_1062 = None
        slice_1103: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_856, 1, 0, 256);  permute_856 = None
        slice_1104: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1103, 3, 0, 257);  slice_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_36: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_194: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_36, -2);  iota_36 = None
        iota_37: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_195: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
        sub_73: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_194, unsqueeze_195);  unsqueeze_194 = unsqueeze_195 = None
        le_18: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_73, 0);  sub_73 = None
        full_default_72: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_73: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_72: "f16[256, 257]" = torch.ops.aten.where.self(le_18, full_default_72, full_default_73);  le_18 = full_default_72 = full_default_73 = None
        rev_36: "f16[256, 257]" = torch.ops.prims.rev.default(where_72, [0]);  where_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_196: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_36, 0);  rev_36 = None
        unsqueeze_197: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_36: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_197, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_326: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_36, torch.bool);  expand_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_83: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_854: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_83, [0, 2, 1, 3]);  full_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1060: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_37, [1, 12, 4096, 513]);  select_scatter_37 = None
        permute_853: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1060, [0, 2, 1, 3]);  view_1060 = None
        slice_1098: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_853, 1, 0, 256);  permute_853 = None
        slice_1099: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1098, 3, 0, 257);  slice_1098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_73: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_326, permute_854, slice_1099);  convert_element_type_326 = permute_854 = slice_1099 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_112: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1104, where_73);  slice_1104 = where_73 = None
        slice_scatter_205: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1105, copy_112, 3, 0, 257);  slice_1105 = copy_112 = None
        slice_scatter_206: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_857, slice_scatter_205, 1, 0, 256);  permute_857 = slice_scatter_205 = None
        permute_858: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_206, [0, 2, 1, 3]);  slice_scatter_206 = None
        view_1064: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_858, [12, 16, 256, 513]);  permute_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1071: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_1064, [1, 12, 4096, 513])
        permute_866: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1071, [0, 2, 1, 3]);  view_1071 = None
        slice_1118: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_866, 1, -256, 9223372036854775807)
        view_1070: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_1064, [1, 12, 4096, 513])
        permute_865: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1070, [0, 2, 1, 3]);  view_1070 = None
        slice_1116: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_865, 1, -256, 9223372036854775807);  permute_865 = None
        slice_1117: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1116, 3, -257, 9223372036854775807);  slice_1116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_37: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_197, [1, 3]);  unsqueeze_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_37: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_37, [1, 256, 12, 257]);  rev_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_327: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_37, torch.bool);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_84: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_863: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_84, [0, 2, 1, 3]);  full_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1068: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_1064, [1, 12, 4096, 513]);  view_1064 = None
        permute_862: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1068, [0, 2, 1, 3]);  view_1068 = None
        slice_1111: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_862, 1, -256, 9223372036854775807);  permute_862 = None
        slice_1112: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1111, 3, -257, 9223372036854775807);  slice_1111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_74: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_327, permute_863, slice_1112);  convert_element_type_327 = permute_863 = slice_1112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_113: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1117, where_74);  slice_1117 = where_74 = None
        slice_scatter_207: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1118, copy_113, 3, -257, 9223372036854775807);  slice_1118 = copy_113 = None
        slice_scatter_208: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_866, slice_scatter_207, 1, -256, 9223372036854775807);  permute_866 = slice_scatter_207 = None
        permute_867: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_208, [0, 2, 1, 3]);  slice_scatter_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_892: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_867, [0, 2, 1, 3]);  permute_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_86: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1125: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_86, 1, 0, -1)
        slice_1123: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_86, 1, 0, -1)
        slice_1124: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1123, 3, 256, 9223372036854775807);  slice_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_85: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_869: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_85, [0, 2, 1, 3]);  full_85 = None
        view_1074: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_869, [1, 4096, 1]);  permute_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1076: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_1074, [1, 8, 512, 1]);  view_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_87: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_1076, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_1076 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_200: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_87, 4);  as_strided_87 = None
        permute_871: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_200, [0, 1, 2, 4, 3]);  unsqueeze_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_74: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_9: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_198: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_9, 2);  ne_9 = None
        unsqueeze_199: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_328: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_199, torch.float16)
        where_75: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_199, full_default_74, convert_element_type_328);  unsqueeze_199 = full_default_74 = convert_element_type_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_870: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_75, [0, 2, 1, 3]);  where_75 = None
        view_1075: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_870, [1, 4096, 1]);  permute_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1077: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_1075, [1, 8, 512, 1]);  view_1075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_88: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_1077, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_1077 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_201: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_88, 4);  as_strided_88 = None
        permute_872: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_201, [0, 1, 4, 2, 3]);  unsqueeze_201 = None
        mul_72: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_871, permute_872);  permute_871 = permute_872 = None
        view_1078: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_72, [1, 15, 512, 512]);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_37: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1078, [0, 0, 0, 1], 0.0);  view_1078 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1079: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_37, [1, 15, 512, 513]);  constant_pad_nd_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1121: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1079, 2, 0, 256)
        slice_1122: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1121, 3, 0, 257);  slice_1121 = None
        copy_114: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_1124, slice_1122);  slice_1124 = slice_1122 = None
        slice_scatter_209: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1125, copy_114, 3, 256, 9223372036854775807);  slice_1125 = copy_114 = None
        slice_scatter_210: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_86, slice_scatter_209, 1, 0, -1);  full_86 = slice_scatter_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_251: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_210, 1, -1)
        select_250: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_210, 1, -1)
        slice_1131: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_250, 2, 256, 9223372036854775807);  select_250 = None
        select_247: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_1079, 1, -1)
        slice_1128: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_247, 1, 256, 9223372036854775807);  select_247 = None
        slice_1129: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1128, 2, 0, 257);  slice_1128 = None
        copy_115: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_1131, slice_1129);  slice_1131 = slice_1129 = None
        slice_scatter_211: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_251, copy_115, 2, 256, 9223372036854775807);  select_251 = copy_115 = None
        select_scatter_38: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_210, slice_scatter_211, 1, -1);  slice_scatter_210 = slice_scatter_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1140: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_38, 1, 1, 9223372036854775807)
        slice_1138: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_38, 1, 1, 9223372036854775807)
        slice_1139: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1138, 3, 0, 256);  slice_1138 = None
        slice_1133: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1079, 2, -257, -1)
        slice_1134: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1133, 3, 257, 9223372036854775807);  slice_1133 = None
        copy_116: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_1139, slice_1134);  slice_1139 = slice_1134 = None
        slice_scatter_212: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1140, copy_116, 3, 0, 256);  slice_1140 = copy_116 = None
        slice_scatter_213: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_38, slice_scatter_212, 1, 1, 9223372036854775807);  select_scatter_38 = slice_scatter_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_258: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_213, 1, 0)
        slice_1150: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_258, 1, 1, 256)
        select_257: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_213, 1, 0)
        slice_1148: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_257, 1, 1, 256);  select_257 = None
        slice_1149: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1148, 2, 1, 256);  slice_1148 = None
        select_253: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_1079, 1, 0);  view_1079 = None
        slice_1143: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_253, 1, 0, 255);  select_253 = None
        slice_1144: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1143, 2, -255, 9223372036854775807);  slice_1143 = None
        copy_117: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_1149, slice_1144);  slice_1149 = slice_1144 = None
        slice_scatter_214: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1150, copy_117, 2, 1, 256);  slice_1150 = copy_117 = None
        slice_scatter_215: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_258, slice_scatter_214, 1, 1, 256);  select_258 = slice_scatter_214 = None
        select_scatter_39: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_213, slice_scatter_215, 1, 0);  slice_scatter_213 = slice_scatter_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1087: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_39, [1, 1, 4096, 513])
        permute_880: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1087, [0, 2, 1, 3]);  view_1087 = None
        slice_1163: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_880, 1, 0, 256)
        view_1086: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_39, [1, 1, 4096, 513])
        permute_879: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1086, [0, 2, 1, 3]);  view_1086 = None
        slice_1161: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_879, 1, 0, 256);  permute_879 = None
        slice_1162: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1161, 3, 0, 257);  slice_1161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_38: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_202: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_38, -2);  iota_38 = None
        iota_39: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_203: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
        sub_75: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_202, unsqueeze_203);  unsqueeze_202 = unsqueeze_203 = None
        le_19: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_75, 0);  sub_75 = None
        full_default_75: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_76: "f16[256, 257]" = torch.ops.aten.where.self(le_19, full_default_75, full_default_76);  le_19 = full_default_75 = full_default_76 = None
        rev_38: "f16[256, 257]" = torch.ops.prims.rev.default(where_76, [0]);  where_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_204: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_38, 0);  rev_38 = None
        unsqueeze_205: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_38: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_205, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_329: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_38, torch.bool);  expand_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_77: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1084: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_39, [1, 1, 4096, 513]);  select_scatter_39 = None
        permute_876: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1084, [0, 2, 1, 3]);  view_1084 = None
        slice_1156: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_876, 1, 0, 256);  permute_876 = None
        slice_1157: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1156, 3, 0, 257);  slice_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_77: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_329, full_default_77, slice_1157);  convert_element_type_329 = full_default_77 = slice_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_118: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1162, where_77);  slice_1162 = where_77 = None
        slice_scatter_216: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1163, copy_118, 3, 0, 257);  slice_1163 = copy_118 = None
        slice_scatter_217: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_880, slice_scatter_216, 1, 0, 256);  permute_880 = slice_scatter_216 = None
        permute_881: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_217, [0, 2, 1, 3]);  slice_scatter_217 = None
        view_1088: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_881, [1, 16, 256, 513]);  permute_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1095: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_1088, [1, 1, 4096, 513])
        permute_889: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1095, [0, 2, 1, 3]);  view_1095 = None
        slice_1176: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_889, 1, -256, 9223372036854775807)
        view_1094: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_1088, [1, 1, 4096, 513])
        permute_888: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1094, [0, 2, 1, 3]);  view_1094 = None
        slice_1174: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_888, 1, -256, 9223372036854775807);  permute_888 = None
        slice_1175: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1174, 3, -257, 9223372036854775807);  slice_1174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_39: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_205, [1, 3]);  unsqueeze_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_39: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_39, [1, 256, 1, 257]);  rev_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_330: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_39, torch.bool);  expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_78: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1092: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_1088, [1, 1, 4096, 513]);  view_1088 = None
        permute_885: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1092, [0, 2, 1, 3]);  view_1092 = None
        slice_1169: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_885, 1, -256, 9223372036854775807);  permute_885 = None
        slice_1170: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1169, 3, -257, 9223372036854775807);  slice_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_78: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_330, full_default_78, slice_1170);  convert_element_type_330 = full_default_78 = slice_1170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_119: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1175, where_78);  slice_1175 = where_78 = None
        slice_scatter_218: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1176, copy_119, 3, -257, 9223372036854775807);  slice_1176 = copy_119 = None
        slice_scatter_219: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_889, slice_scatter_218, 1, -256, 9223372036854775807);  permute_889 = slice_scatter_218 = None
        permute_890: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_219, [0, 2, 1, 3]);  slice_scatter_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_893: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_890, [0, 2, 1, 3]);  permute_890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_101: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_892, permute_893);  permute_892 = permute_893 = None
        permute_894: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_101, [0, 2, 1, 3]);  add_101 = None
        permute_895: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_894, [0, 2, 1, 3]);  permute_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_114: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_895, memory_format = torch.contiguous_format);  permute_895 = None
        convert_element_type_331: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_114, torch.float32);  clone_114 = None
        amax_9: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_331, [-1], True)
        sub_76: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_331, amax_9);  convert_element_type_331 = amax_9 = None
        exp_9: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_10: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_97: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_79: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_207, full_default_79, div_97);  unsqueeze_207 = full_default_79 = div_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_332: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_79, torch.float16);  where_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_897: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_332, [0, 2, 1, 3]);  convert_element_type_332 = None
        view_1104: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_897, [12, 16, 256, 513]);  permute_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_39: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_1104, [0, 257], 0.0);  view_1104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1106: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_39, [12, 16, -1]);  constant_pad_nd_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1179: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_1106, 2, 0, -256);  view_1106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1107: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_1179, [12, 16, 256, 769]);  slice_1179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1180: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_1107, 3, 0, -1);  view_1107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_208: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1180, 4);  slice_1180 = None
        view_1108: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_208, [192, 256, 768]);  unsqueeze_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_1012: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_819, [4096, 768]);  permute_819 = None
        permute_822: "f16[768, 768]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_56: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg152_1, view_1012, permute_822);  arg152_1 = view_1012 = permute_822 = None
        view_1013: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_56, [4096, 1, 768]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1103: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_1013, [4096, 1, 12, 64]);  view_1013 = None
        permute_896: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1103, [1, 0, 2, 3]);  view_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_898: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_896, [0, 2, 1, 3]);  permute_896 = None
        view_1105: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_898, [12, 4096, 64]);  permute_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_38: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_1105, [0, 0, 256, 256], -1.0);  view_1105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_89: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_38, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_209: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_89, 4);  as_strided_89 = None
        clone_116: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_209, memory_format = torch.contiguous_format);  unsqueeze_209 = None
        view_1109: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_116, [192, 768, 64]);  clone_116 = None
        bmm_19: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_1108, view_1109);  view_1108 = view_1109 = None
        view_1110: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_19, [12, 16, 256, 1, 64]);  bmm_19 = None
        permute_903: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_1110, [0, 1, 2, 4, 3]);  view_1110 = None
        view_1111: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_903, [12, 16, 256, 64]);  permute_903 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1112: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_1111, [1, 12, 4096, 64]);  view_1111 = None
        permute_904: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1112, [0, 2, 1, 3]);  view_1112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_905: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_904, [1, 0, 2, 3]);  permute_904 = None
        clone_117: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_905, memory_format = torch.contiguous_format);  permute_905 = None
        view_1113: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_117, [4096, 1, 768]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_906: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_1113, [1, 0, 2]);  view_1113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_1114: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_906, [4096, 768]);  permute_906 = None
        permute_907: "f16[768, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_57: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg154_1, view_1114, permute_907);  arg154_1 = view_1114 = permute_907 = None
        view_1115: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_57, [1, 4096, 768]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_103: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_1115, convert_element_type_314);  view_1115 = convert_element_type_314 = None
        convert_element_type_338: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_103, torch.float32);  add_103 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_338, [2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 4096, 1]" = var_mean_18[0]
        getitem_37: "f32[1, 4096, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_78: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_338, getitem_37);  convert_element_type_338 = getitem_37 = None
        add_104: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_73: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_18);  sub_78 = rsqrt_18 = None
        mul_74: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_73, arg155_1);  mul_73 = arg155_1 = None
        add_105: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_74, arg156_1);  mul_74 = arg156_1 = None
        convert_element_type_339: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_105, torch.float16);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1116: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_339, [4096, 768])
        permute_908: "f16[768, 3072]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_58: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg158_1, view_1116, permute_908);  arg158_1 = view_1116 = permute_908 = None
        view_1117: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_58, [1, 4096, 3072]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_343: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_1117, torch.float32);  view_1117 = None
        mul_75: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_343, 0.5)
        mul_76: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_343, 0.7071067811865476);  convert_element_type_343 = None
        erf_9: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_106: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_77: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_106);  mul_75 = add_106 = None
        convert_element_type_344: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_77, torch.float16);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1118: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_344, [4096, 3072]);  convert_element_type_344 = None
        permute_909: "f16[3072, 768]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_59: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg160_1, view_1118, permute_909);  arg160_1 = view_1118 = permute_909 = None
        view_1119: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_59, [1, 4096, 768]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_107: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_1119, convert_element_type_339);  view_1119 = convert_element_type_339 = None
        convert_element_type_348: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32);  add_107 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_348, [2], correction = 0, keepdim = True)
        getitem_38: "f32[1, 4096, 1]" = var_mean_19[0]
        getitem_39: "f32[1, 4096, 1]" = var_mean_19[1];  var_mean_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_97: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_98: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_227: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_228: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        full_default_87: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_90: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1185: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_90, 1, 0, -1)
        slice_1183: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_90, 1, 0, -1)
        slice_1184: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1183, 3, 256, 9223372036854775807);  slice_1183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_79: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_348, getitem_39);  convert_element_type_348 = getitem_39 = None
        add_108: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_78: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_19);  sub_79 = rsqrt_19 = None
        mul_79: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg161_1);  mul_78 = arg161_1 = None
        add_109: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_79, arg162_1);  mul_79 = arg162_1 = None
        convert_element_type_349: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_109, torch.float16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_910: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_349, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_1120: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_910, [4096, 768])
        permute_911: "f16[768, 768]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_60: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg164_1, view_1120, permute_911);  arg164_1 = view_1120 = permute_911 = None
        view_1121: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_60, [4096, 1, 768]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_100: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_1121, 8.0);  view_1121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1160: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_100, [4096, 1, 12, 64]);  div_100 = None
        permute_935: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1160, [1, 0, 2, 3]);  view_1160 = None
        permute_936: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_935, [0, 2, 1, 3]);  permute_935 = None
        view_1161: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_936, [12, 4096, 64]);  permute_936 = None
        view_1162: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_1161, [12, 8, 512, 64]);  view_1161 = None
        as_strided_95: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_1162, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_1162 = None
        unsqueeze_214: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_95, 4);  as_strided_95 = None
        clone_120: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_214, memory_format = torch.contiguous_format);  unsqueeze_214 = None
        view_1163: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_120, [180, 512, 64]);  clone_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_1122: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_910, [4096, 768])
        permute_912: "f16[768, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_61: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg166_1, view_1122, permute_912);  arg166_1 = view_1122 = permute_912 = None
        view_1123: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_61, [4096, 1, 768]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1131: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_1123, [4096, 1, 12, 64]);  view_1123 = None
        permute_915: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1131, [1, 0, 2, 3]);  view_1131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_920: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_915, [0, 2, 1, 3]);  permute_915 = None
        view_1137: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_920, [12, 4096, 64]);  permute_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1146: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_1137, [12, 8, 512, 64]);  view_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_91: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_1146, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_1146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_212: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_91, 4);  as_strided_91 = None
        permute_930: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_212, [0, 1, 4, 2, 3]);  unsqueeze_212 = None
        permute_939: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_930, [0, 1, 4, 3, 2]);  permute_930 = None
        clone_121: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_939, memory_format = torch.contiguous_format);  permute_939 = None
        view_1164: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_121, [180, 64, 512]);  clone_121 = None
        bmm_20: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_1163, view_1164);  view_1163 = view_1164 = None
        view_1165: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_20, [12, 15, 512, 1, 512]);  bmm_20 = None
        permute_940: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_1165, [0, 1, 2, 4, 3]);  view_1165 = None
        view_1166: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_940, [12, 15, 512, 512]);  permute_940 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_40: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1166, [0, 0, 0, 1], 0.0);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1167: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_40, [12, 15, 512, 513]);  constant_pad_nd_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1181: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1167, 2, 0, 256)
        slice_1182: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1181, 3, 0, 257);  slice_1181 = None
        copy_120: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_1184, slice_1182);  slice_1184 = slice_1182 = None
        slice_scatter_220: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1185, copy_120, 3, 256, 9223372036854775807);  slice_1185 = copy_120 = None
        slice_scatter_221: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_90, slice_scatter_220, 1, 0, -1);  full_90 = slice_scatter_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_264: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_221, 1, -1)
        select_263: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_221, 1, -1)
        slice_1191: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_263, 2, 256, 9223372036854775807);  select_263 = None
        select_260: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_1167, 1, -1)
        slice_1188: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_260, 1, 256, 9223372036854775807);  select_260 = None
        slice_1189: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1188, 2, 0, 257);  slice_1188 = None
        copy_121: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_1191, slice_1189);  slice_1191 = slice_1189 = None
        slice_scatter_222: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_264, copy_121, 2, 256, 9223372036854775807);  select_264 = copy_121 = None
        select_scatter_40: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_221, slice_scatter_222, 1, -1);  slice_scatter_221 = slice_scatter_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1200: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_40, 1, 1, 9223372036854775807)
        slice_1198: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_40, 1, 1, 9223372036854775807)
        slice_1199: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1198, 3, 0, 256);  slice_1198 = None
        slice_1193: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1167, 2, -257, -1)
        slice_1194: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1193, 3, 257, 9223372036854775807);  slice_1193 = None
        copy_122: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_1199, slice_1194);  slice_1199 = slice_1194 = None
        slice_scatter_223: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1200, copy_122, 3, 0, 256);  slice_1200 = copy_122 = None
        slice_scatter_224: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_40, slice_scatter_223, 1, 1, 9223372036854775807);  select_scatter_40 = slice_scatter_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_271: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_224, 1, 0)
        slice_1210: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_271, 1, 1, 256)
        select_270: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_224, 1, 0)
        slice_1208: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_270, 1, 1, 256);  select_270 = None
        slice_1209: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1208, 2, 1, 256);  slice_1208 = None
        select_266: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_1167, 1, 0);  view_1167 = None
        slice_1203: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_266, 1, 0, 255);  select_266 = None
        slice_1204: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1203, 2, -255, 9223372036854775807);  slice_1203 = None
        copy_123: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_1209, slice_1204);  slice_1209 = slice_1204 = None
        slice_scatter_225: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1210, copy_123, 2, 1, 256);  slice_1210 = copy_123 = None
        slice_scatter_226: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_271, slice_scatter_225, 1, 1, 256);  select_271 = slice_scatter_225 = None
        select_scatter_41: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_224, slice_scatter_226, 1, 0);  slice_scatter_224 = slice_scatter_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1175: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_41, [1, 12, 4096, 513])
        permute_948: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1175, [0, 2, 1, 3]);  view_1175 = None
        slice_1223: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_948, 1, 0, 256)
        view_1174: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_41, [1, 12, 4096, 513])
        permute_947: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1174, [0, 2, 1, 3]);  view_1174 = None
        slice_1221: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_947, 1, 0, 256);  permute_947 = None
        slice_1222: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1221, 3, 0, 257);  slice_1221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_40: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_215: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_40, -2);  iota_40 = None
        iota_41: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_216: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
        sub_81: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_215, unsqueeze_216);  unsqueeze_215 = unsqueeze_216 = None
        le_20: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_81, 0);  sub_81 = None
        full_default_80: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_81: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_80: "f16[256, 257]" = torch.ops.aten.where.self(le_20, full_default_80, full_default_81);  le_20 = full_default_80 = full_default_81 = None
        rev_40: "f16[256, 257]" = torch.ops.prims.rev.default(where_80, [0]);  where_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_217: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_40, 0);  rev_40 = None
        unsqueeze_218: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_40: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_218, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_361: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_40, torch.bool);  expand_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_92: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_945: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_92, [0, 2, 1, 3]);  full_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1172: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_41, [1, 12, 4096, 513]);  select_scatter_41 = None
        permute_944: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1172, [0, 2, 1, 3]);  view_1172 = None
        slice_1216: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_944, 1, 0, 256);  permute_944 = None
        slice_1217: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1216, 3, 0, 257);  slice_1216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_81: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_361, permute_945, slice_1217);  convert_element_type_361 = permute_945 = slice_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_124: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1222, where_81);  slice_1222 = where_81 = None
        slice_scatter_227: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1223, copy_124, 3, 0, 257);  slice_1223 = copy_124 = None
        slice_scatter_228: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_948, slice_scatter_227, 1, 0, 256);  permute_948 = slice_scatter_227 = None
        permute_949: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_228, [0, 2, 1, 3]);  slice_scatter_228 = None
        view_1176: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_949, [12, 16, 256, 513]);  permute_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1183: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_1176, [1, 12, 4096, 513])
        permute_957: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1183, [0, 2, 1, 3]);  view_1183 = None
        slice_1236: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_957, 1, -256, 9223372036854775807)
        view_1182: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_1176, [1, 12, 4096, 513])
        permute_956: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1182, [0, 2, 1, 3]);  view_1182 = None
        slice_1234: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_956, 1, -256, 9223372036854775807);  permute_956 = None
        slice_1235: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1234, 3, -257, 9223372036854775807);  slice_1234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_41: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_218, [1, 3]);  unsqueeze_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_41: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_41, [1, 256, 12, 257]);  rev_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_362: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_41, torch.bool);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_93: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_954: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_93, [0, 2, 1, 3]);  full_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1180: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_1176, [1, 12, 4096, 513]);  view_1176 = None
        permute_953: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1180, [0, 2, 1, 3]);  view_1180 = None
        slice_1229: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_953, 1, -256, 9223372036854775807);  permute_953 = None
        slice_1230: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1229, 3, -257, 9223372036854775807);  slice_1229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_82: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_362, permute_954, slice_1230);  convert_element_type_362 = permute_954 = slice_1230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_125: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1235, where_82);  slice_1235 = where_82 = None
        slice_scatter_229: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1236, copy_125, 3, -257, 9223372036854775807);  slice_1236 = copy_125 = None
        slice_scatter_230: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_957, slice_scatter_229, 1, -256, 9223372036854775807);  permute_957 = slice_scatter_229 = None
        permute_958: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_230, [0, 2, 1, 3]);  slice_scatter_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_983: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_958, [0, 2, 1, 3]);  permute_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_95: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1243: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_95, 1, 0, -1)
        slice_1241: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_95, 1, 0, -1)
        slice_1242: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1241, 3, 256, 9223372036854775807);  slice_1241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_94: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_960: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_94, [0, 2, 1, 3]);  full_94 = None
        view_1186: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_960, [1, 4096, 1]);  permute_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1188: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_1186, [1, 8, 512, 1]);  view_1186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_96: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_1188, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_1188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_221: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_96, 4);  as_strided_96 = None
        permute_962: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_221, [0, 1, 2, 4, 3]);  unsqueeze_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_82: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_10: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_219: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_10, 2);  ne_10 = None
        unsqueeze_220: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_363: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_220, torch.float16)
        where_83: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_220, full_default_82, convert_element_type_363);  unsqueeze_220 = full_default_82 = convert_element_type_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_961: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_83, [0, 2, 1, 3]);  where_83 = None
        view_1187: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_961, [1, 4096, 1]);  permute_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1189: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_1187, [1, 8, 512, 1]);  view_1187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_97: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_1189, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_1189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_222: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_97, 4);  as_strided_97 = None
        permute_963: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_222, [0, 1, 4, 2, 3]);  unsqueeze_222 = None
        mul_80: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_962, permute_963);  permute_962 = permute_963 = None
        view_1190: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_80, [1, 15, 512, 512]);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_41: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1190, [0, 0, 0, 1], 0.0);  view_1190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1191: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_41, [1, 15, 512, 513]);  constant_pad_nd_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1239: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1191, 2, 0, 256)
        slice_1240: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1239, 3, 0, 257);  slice_1239 = None
        copy_126: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_1242, slice_1240);  slice_1242 = slice_1240 = None
        slice_scatter_231: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1243, copy_126, 3, 256, 9223372036854775807);  slice_1243 = copy_126 = None
        slice_scatter_232: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_95, slice_scatter_231, 1, 0, -1);  full_95 = slice_scatter_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_277: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_232, 1, -1)
        select_276: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_232, 1, -1)
        slice_1249: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_276, 2, 256, 9223372036854775807);  select_276 = None
        select_273: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_1191, 1, -1)
        slice_1246: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_273, 1, 256, 9223372036854775807);  select_273 = None
        slice_1247: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1246, 2, 0, 257);  slice_1246 = None
        copy_127: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_1249, slice_1247);  slice_1249 = slice_1247 = None
        slice_scatter_233: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_277, copy_127, 2, 256, 9223372036854775807);  select_277 = copy_127 = None
        select_scatter_42: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_232, slice_scatter_233, 1, -1);  slice_scatter_232 = slice_scatter_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1258: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_42, 1, 1, 9223372036854775807)
        slice_1256: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_42, 1, 1, 9223372036854775807)
        slice_1257: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1256, 3, 0, 256);  slice_1256 = None
        slice_1251: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1191, 2, -257, -1)
        slice_1252: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1251, 3, 257, 9223372036854775807);  slice_1251 = None
        copy_128: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_1257, slice_1252);  slice_1257 = slice_1252 = None
        slice_scatter_234: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1258, copy_128, 3, 0, 256);  slice_1258 = copy_128 = None
        slice_scatter_235: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_42, slice_scatter_234, 1, 1, 9223372036854775807);  select_scatter_42 = slice_scatter_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_284: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_235, 1, 0)
        slice_1268: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_284, 1, 1, 256)
        select_283: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_235, 1, 0)
        slice_1266: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_283, 1, 1, 256);  select_283 = None
        slice_1267: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1266, 2, 1, 256);  slice_1266 = None
        select_279: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_1191, 1, 0);  view_1191 = None
        slice_1261: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_279, 1, 0, 255);  select_279 = None
        slice_1262: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1261, 2, -255, 9223372036854775807);  slice_1261 = None
        copy_129: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_1267, slice_1262);  slice_1267 = slice_1262 = None
        slice_scatter_236: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1268, copy_129, 2, 1, 256);  slice_1268 = copy_129 = None
        slice_scatter_237: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_284, slice_scatter_236, 1, 1, 256);  select_284 = slice_scatter_236 = None
        select_scatter_43: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_235, slice_scatter_237, 1, 0);  slice_scatter_235 = slice_scatter_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1199: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_43, [1, 1, 4096, 513])
        permute_971: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1199, [0, 2, 1, 3]);  view_1199 = None
        slice_1281: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_971, 1, 0, 256)
        view_1198: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_43, [1, 1, 4096, 513])
        permute_970: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1198, [0, 2, 1, 3]);  view_1198 = None
        slice_1279: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_970, 1, 0, 256);  permute_970 = None
        slice_1280: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1279, 3, 0, 257);  slice_1279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_42: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_223: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_42, -2);  iota_42 = None
        iota_43: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_224: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
        sub_83: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_223, unsqueeze_224);  unsqueeze_223 = unsqueeze_224 = None
        le_21: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_83, 0);  sub_83 = None
        full_default_83: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_84: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_84: "f16[256, 257]" = torch.ops.aten.where.self(le_21, full_default_83, full_default_84);  le_21 = full_default_83 = full_default_84 = None
        rev_42: "f16[256, 257]" = torch.ops.prims.rev.default(where_84, [0]);  where_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_225: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_42, 0);  rev_42 = None
        unsqueeze_226: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_42: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_226, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_364: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_42, torch.bool);  expand_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_85: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1196: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_43, [1, 1, 4096, 513]);  select_scatter_43 = None
        permute_967: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1196, [0, 2, 1, 3]);  view_1196 = None
        slice_1274: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_967, 1, 0, 256);  permute_967 = None
        slice_1275: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1274, 3, 0, 257);  slice_1274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_85: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_364, full_default_85, slice_1275);  convert_element_type_364 = full_default_85 = slice_1275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_130: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1280, where_85);  slice_1280 = where_85 = None
        slice_scatter_238: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1281, copy_130, 3, 0, 257);  slice_1281 = copy_130 = None
        slice_scatter_239: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_971, slice_scatter_238, 1, 0, 256);  permute_971 = slice_scatter_238 = None
        permute_972: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_239, [0, 2, 1, 3]);  slice_scatter_239 = None
        view_1200: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_972, [1, 16, 256, 513]);  permute_972 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1207: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_1200, [1, 1, 4096, 513])
        permute_980: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1207, [0, 2, 1, 3]);  view_1207 = None
        slice_1294: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_980, 1, -256, 9223372036854775807)
        view_1206: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_1200, [1, 1, 4096, 513])
        permute_979: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1206, [0, 2, 1, 3]);  view_1206 = None
        slice_1292: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_979, 1, -256, 9223372036854775807);  permute_979 = None
        slice_1293: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1292, 3, -257, 9223372036854775807);  slice_1292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_43: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_226, [1, 3]);  unsqueeze_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_43: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_43, [1, 256, 1, 257]);  rev_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_365: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_43, torch.bool);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_86: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1204: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_1200, [1, 1, 4096, 513]);  view_1200 = None
        permute_976: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1204, [0, 2, 1, 3]);  view_1204 = None
        slice_1287: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_976, 1, -256, 9223372036854775807);  permute_976 = None
        slice_1288: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1287, 3, -257, 9223372036854775807);  slice_1287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_86: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_365, full_default_86, slice_1288);  convert_element_type_365 = full_default_86 = slice_1288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_131: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1293, where_86);  slice_1293 = where_86 = None
        slice_scatter_240: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1294, copy_131, 3, -257, 9223372036854775807);  slice_1294 = copy_131 = None
        slice_scatter_241: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_980, slice_scatter_240, 1, -256, 9223372036854775807);  permute_980 = slice_scatter_240 = None
        permute_981: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_241, [0, 2, 1, 3]);  slice_scatter_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_984: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_981, [0, 2, 1, 3]);  permute_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_112: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_983, permute_984);  permute_983 = permute_984 = None
        permute_985: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_112, [0, 2, 1, 3]);  add_112 = None
        permute_986: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_985, [0, 2, 1, 3]);  permute_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_126: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_986, memory_format = torch.contiguous_format);  permute_986 = None
        convert_element_type_366: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_126, torch.float32);  clone_126 = None
        amax_10: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_366, [-1], True)
        sub_84: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_366, amax_10);  convert_element_type_366 = amax_10 = None
        exp_10: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
        sum_11: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_107: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_87: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_228, full_default_87, div_107);  unsqueeze_228 = full_default_87 = div_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_367: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_87, torch.float16);  where_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_988: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_367, [0, 2, 1, 3]);  convert_element_type_367 = None
        view_1216: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_988, [12, 16, 256, 513]);  permute_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_43: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_1216, [0, 257], 0.0);  view_1216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1218: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_43, [12, 16, -1]);  constant_pad_nd_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1297: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_1218, 2, 0, -256);  view_1218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1219: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_1297, [12, 16, 256, 769]);  slice_1297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1298: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_1219, 3, 0, -1);  view_1219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_229: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1298, 4);  slice_1298 = None
        view_1220: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_229, [192, 256, 768]);  unsqueeze_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_1124: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_910, [4096, 768]);  permute_910 = None
        permute_913: "f16[768, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_62: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg168_1, view_1124, permute_913);  arg168_1 = view_1124 = permute_913 = None
        view_1125: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_62, [4096, 1, 768]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1215: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_1125, [4096, 1, 12, 64]);  view_1125 = None
        permute_987: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1215, [1, 0, 2, 3]);  view_1215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_989: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_987, [0, 2, 1, 3]);  permute_987 = None
        view_1217: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_989, [12, 4096, 64]);  permute_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_42: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_1217, [0, 0, 256, 256], -1.0);  view_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_98: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_42, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_230: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_98, 4);  as_strided_98 = None
        clone_128: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_230, memory_format = torch.contiguous_format);  unsqueeze_230 = None
        view_1221: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_128, [192, 768, 64]);  clone_128 = None
        bmm_21: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_1220, view_1221);  view_1220 = view_1221 = None
        view_1222: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_21, [12, 16, 256, 1, 64]);  bmm_21 = None
        permute_994: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_1222, [0, 1, 2, 4, 3]);  view_1222 = None
        view_1223: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_994, [12, 16, 256, 64]);  permute_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1224: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_1223, [1, 12, 4096, 64]);  view_1223 = None
        permute_995: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1224, [0, 2, 1, 3]);  view_1224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_996: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_995, [1, 0, 2, 3]);  permute_995 = None
        clone_129: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_996, memory_format = torch.contiguous_format);  permute_996 = None
        view_1225: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_129, [4096, 1, 768]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_997: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_1225, [1, 0, 2]);  view_1225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_1226: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_997, [4096, 768]);  permute_997 = None
        permute_998: "f16[768, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_63: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg170_1, view_1226, permute_998);  arg170_1 = view_1226 = permute_998 = None
        view_1227: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_63, [1, 4096, 768]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_114: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_1227, convert_element_type_349);  view_1227 = convert_element_type_349 = None
        convert_element_type_373: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32);  add_114 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_373, [2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 4096, 1]" = var_mean_20[0]
        getitem_41: "f32[1, 4096, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_86: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_373, getitem_41);  convert_element_type_373 = getitem_41 = None
        add_115: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_81: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_20);  sub_86 = rsqrt_20 = None
        mul_82: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_81, arg171_1);  mul_81 = arg171_1 = None
        add_116: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_82, arg172_1);  mul_82 = arg172_1 = None
        convert_element_type_374: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_116, torch.float16);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1228: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_374, [4096, 768])
        permute_999: "f16[768, 3072]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_64: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg174_1, view_1228, permute_999);  arg174_1 = view_1228 = permute_999 = None
        view_1229: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_64, [1, 4096, 3072]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_378: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_1229, torch.float32);  view_1229 = None
        mul_83: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_378, 0.5)
        mul_84: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_378, 0.7071067811865476);  convert_element_type_378 = None
        erf_10: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_117: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_85: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_83, add_117);  mul_83 = add_117 = None
        convert_element_type_379: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_85, torch.float16);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1230: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_379, [4096, 3072]);  convert_element_type_379 = None
        permute_1000: "f16[3072, 768]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_65: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg176_1, view_1230, permute_1000);  arg176_1 = view_1230 = permute_1000 = None
        view_1231: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_65, [1, 4096, 768]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_118: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_1231, convert_element_type_374);  view_1231 = convert_element_type_374 = None
        convert_element_type_383: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_118, torch.float32);  add_118 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_383, [2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 4096, 1]" = var_mean_21[0]
        getitem_43: "f32[1, 4096, 1]" = var_mean_21[1];  var_mean_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_106: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_107: "f16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_248: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2);  arg8_1 = None
        unsqueeze_249: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        full_default_95: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_99: "f16[12, 16, 256, 513]" = torch.ops.aten.full.default([12, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1303: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_99, 1, 0, -1)
        slice_1301: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_99, 1, 0, -1)
        slice_1302: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1301, 3, 256, 9223372036854775807);  slice_1301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_87: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_383, getitem_43);  convert_element_type_383 = getitem_43 = None
        add_119: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_86: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_21);  sub_87 = rsqrt_21 = None
        mul_87: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_86, arg177_1);  mul_86 = arg177_1 = None
        add_120: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_87, arg178_1);  mul_87 = arg178_1 = None
        convert_element_type_384: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_120, torch.float16);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1001: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_384, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        view_1232: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_1001, [4096, 768])
        permute_1002: "f16[768, 768]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_66: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg180_1, view_1232, permute_1002);  arg180_1 = view_1232 = permute_1002 = None
        view_1233: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_66, [4096, 1, 768]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_110: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(view_1233, 8.0);  view_1233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1272: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_110, [4096, 1, 12, 64]);  div_110 = None
        permute_1026: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1272, [1, 0, 2, 3]);  view_1272 = None
        permute_1027: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_1026, [0, 2, 1, 3]);  permute_1026 = None
        view_1273: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_1027, [12, 4096, 64]);  permute_1027 = None
        view_1274: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_1273, [12, 8, 512, 64]);  view_1273 = None
        as_strided_104: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_1274, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_1274 = None
        unsqueeze_235: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_104, 4);  as_strided_104 = None
        clone_132: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_235, memory_format = torch.contiguous_format);  unsqueeze_235 = None
        view_1275: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_132, [180, 512, 64]);  clone_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        view_1234: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_1001, [4096, 768])
        permute_1003: "f16[768, 768]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_67: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg182_1, view_1234, permute_1003);  arg182_1 = view_1234 = permute_1003 = None
        view_1235: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_67, [4096, 1, 768]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1243: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_1235, [4096, 1, 12, 64]);  view_1235 = None
        permute_1006: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1243, [1, 0, 2, 3]);  view_1243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1011: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_1006, [0, 2, 1, 3]);  permute_1006 = None
        view_1249: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_1011, [12, 4096, 64]);  permute_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1258: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(view_1249, [12, 8, 512, 64]);  view_1249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_100: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(view_1258, [12, 15, 512, 64], [64, 196608, 768, 1]);  view_1258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_233: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_100, 4);  as_strided_100 = None
        permute_1021: "f16[12, 15, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_233, [0, 1, 4, 2, 3]);  unsqueeze_233 = None
        permute_1030: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.permute.default(permute_1021, [0, 1, 4, 3, 2]);  permute_1021 = None
        clone_133: "f16[12, 15, 64, 512, 1]" = torch.ops.aten.clone.default(permute_1030, memory_format = torch.contiguous_format);  permute_1030 = None
        view_1276: "f16[180, 64, 512]" = torch.ops.aten.reshape.default(clone_133, [180, 64, 512]);  clone_133 = None
        bmm_22: "f16[180, 512, 512]" = torch.ops.aten.bmm.default(view_1275, view_1276);  view_1275 = view_1276 = None
        view_1277: "f16[12, 15, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_22, [12, 15, 512, 1, 512]);  bmm_22 = None
        permute_1031: "f16[12, 15, 512, 512, 1]" = torch.ops.aten.permute.default(view_1277, [0, 1, 2, 4, 3]);  view_1277 = None
        view_1278: "f16[12, 15, 512, 512]" = torch.ops.aten.reshape.default(permute_1031, [12, 15, 512, 512]);  permute_1031 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_44: "f16[12, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1278, [0, 0, 0, 1], 0.0);  view_1278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1279: "f16[12, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_44, [12, 15, 512, 513]);  constant_pad_nd_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1299: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1279, 2, 0, 256)
        slice_1300: "f16[12, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1299, 3, 0, 257);  slice_1299 = None
        copy_132: "f16[12, 15, 256, 257]" = torch.ops.aten.copy.default(slice_1302, slice_1300);  slice_1302 = slice_1300 = None
        slice_scatter_242: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1303, copy_132, 3, 256, 9223372036854775807);  slice_1303 = copy_132 = None
        slice_scatter_243: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_99, slice_scatter_242, 1, 0, -1);  full_99 = slice_scatter_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_290: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_243, 1, -1)
        select_289: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_243, 1, -1)
        slice_1309: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_289, 2, 256, 9223372036854775807);  select_289 = None
        select_286: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_1279, 1, -1)
        slice_1306: "f16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_286, 1, 256, 9223372036854775807);  select_286 = None
        slice_1307: "f16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1306, 2, 0, 257);  slice_1306 = None
        copy_133: "f16[12, 256, 257]" = torch.ops.aten.copy.default(slice_1309, slice_1307);  slice_1309 = slice_1307 = None
        slice_scatter_244: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_290, copy_133, 2, 256, 9223372036854775807);  select_290 = copy_133 = None
        select_scatter_44: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_243, slice_scatter_244, 1, -1);  slice_scatter_243 = slice_scatter_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1318: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_44, 1, 1, 9223372036854775807)
        slice_1316: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_44, 1, 1, 9223372036854775807)
        slice_1317: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1316, 3, 0, 256);  slice_1316 = None
        slice_1311: "f16[12, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1279, 2, -257, -1)
        slice_1312: "f16[12, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1311, 3, 257, 9223372036854775807);  slice_1311 = None
        copy_134: "f16[12, 15, 256, 256]" = torch.ops.aten.copy.default(slice_1317, slice_1312);  slice_1317 = slice_1312 = None
        slice_scatter_245: "f16[12, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1318, copy_134, 3, 0, 256);  slice_1318 = copy_134 = None
        slice_scatter_246: "f16[12, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_44, slice_scatter_245, 1, 1, 9223372036854775807);  select_scatter_44 = slice_scatter_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_297: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_246, 1, 0)
        slice_1328: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_297, 1, 1, 256)
        select_296: "f16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_246, 1, 0)
        slice_1326: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_296, 1, 1, 256);  select_296 = None
        slice_1327: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1326, 2, 1, 256);  slice_1326 = None
        select_292: "f16[12, 512, 513]" = torch.ops.aten.select.int(view_1279, 1, 0);  view_1279 = None
        slice_1321: "f16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_292, 1, 0, 255);  select_292 = None
        slice_1322: "f16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1321, 2, -255, 9223372036854775807);  slice_1321 = None
        copy_135: "f16[12, 255, 255]" = torch.ops.aten.copy.default(slice_1327, slice_1322);  slice_1327 = slice_1322 = None
        slice_scatter_247: "f16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1328, copy_135, 2, 1, 256);  slice_1328 = copy_135 = None
        slice_scatter_248: "f16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_297, slice_scatter_247, 1, 1, 256);  select_297 = slice_scatter_247 = None
        select_scatter_45: "f16[12, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_246, slice_scatter_248, 1, 0);  slice_scatter_246 = slice_scatter_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1287: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_45, [1, 12, 4096, 513])
        permute_1039: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1287, [0, 2, 1, 3]);  view_1287 = None
        slice_1341: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1039, 1, 0, 256)
        view_1286: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_45, [1, 12, 4096, 513])
        permute_1038: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1286, [0, 2, 1, 3]);  view_1286 = None
        slice_1339: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1038, 1, 0, 256);  permute_1038 = None
        slice_1340: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1339, 3, 0, 257);  slice_1339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_44: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_236: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_44, -2);  iota_44 = None
        iota_45: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_237: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
        sub_89: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_236, unsqueeze_237);  unsqueeze_236 = unsqueeze_237 = None
        le_22: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_89, 0);  sub_89 = None
        full_default_88: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_89: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_88: "f16[256, 257]" = torch.ops.aten.where.self(le_22, full_default_88, full_default_89);  le_22 = full_default_88 = full_default_89 = None
        rev_44: "f16[256, 257]" = torch.ops.prims.rev.default(where_88, [0]);  where_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_238: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_44, 0);  rev_44 = None
        unsqueeze_239: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_44: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_239, [1, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_396: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_44, torch.bool);  expand_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_101: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_1036: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_101, [0, 2, 1, 3]);  full_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1284: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_45, [1, 12, 4096, 513]);  select_scatter_45 = None
        permute_1035: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1284, [0, 2, 1, 3]);  view_1284 = None
        slice_1334: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1035, 1, 0, 256);  permute_1035 = None
        slice_1335: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1334, 3, 0, 257);  slice_1334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_89: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_396, permute_1036, slice_1335);  convert_element_type_396 = permute_1036 = slice_1335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_136: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1340, where_89);  slice_1340 = where_89 = None
        slice_scatter_249: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1341, copy_136, 3, 0, 257);  slice_1341 = copy_136 = None
        slice_scatter_250: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1039, slice_scatter_249, 1, 0, 256);  permute_1039 = slice_scatter_249 = None
        permute_1040: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_250, [0, 2, 1, 3]);  slice_scatter_250 = None
        view_1288: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_1040, [12, 16, 256, 513]);  permute_1040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1295: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_1288, [1, 12, 4096, 513])
        permute_1048: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1295, [0, 2, 1, 3]);  view_1295 = None
        slice_1354: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1048, 1, -256, 9223372036854775807)
        view_1294: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_1288, [1, 12, 4096, 513])
        permute_1047: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1294, [0, 2, 1, 3]);  view_1294 = None
        slice_1352: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1047, 1, -256, 9223372036854775807);  permute_1047 = None
        slice_1353: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1352, 3, -257, 9223372036854775807);  slice_1352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_45: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_239, [1, 3]);  unsqueeze_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_45: "f16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_45, [1, 256, 12, 257]);  rev_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_397: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_45, torch.bool);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_102: "f16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_1045: "f16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_102, [0, 2, 1, 3]);  full_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1292: "f16[1, 12, 4096, 513]" = torch.ops.aten.reshape.default(view_1288, [1, 12, 4096, 513]);  view_1288 = None
        permute_1044: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(view_1292, [0, 2, 1, 3]);  view_1292 = None
        slice_1347: "f16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1044, 1, -256, 9223372036854775807);  permute_1044 = None
        slice_1348: "f16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1347, 3, -257, 9223372036854775807);  slice_1347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_90: "f16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_397, permute_1045, slice_1348);  convert_element_type_397 = permute_1045 = slice_1348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_137: "f16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1353, where_90);  slice_1353 = where_90 = None
        slice_scatter_251: "f16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1354, copy_137, 3, -257, 9223372036854775807);  slice_1354 = copy_137 = None
        slice_scatter_252: "f16[1, 4096, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1048, slice_scatter_251, 1, -256, 9223372036854775807);  permute_1048 = slice_scatter_251 = None
        permute_1049: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_252, [0, 2, 1, 3]);  slice_scatter_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1074: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_1049, [0, 2, 1, 3]);  permute_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_104: "f16[1, 16, 256, 513]" = torch.ops.aten.full.default([1, 16, 256, 513], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1361: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1)
        slice_1359: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1)
        slice_1360: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1359, 3, 256, 9223372036854775807);  slice_1359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_103: "f16[1, 4096, 1, 1]" = torch.ops.aten.full.default([1, 4096, 1, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1051: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(full_103, [0, 2, 1, 3]);  full_103 = None
        view_1298: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_1051, [1, 4096, 1]);  permute_1051 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1300: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_1298, [1, 8, 512, 1]);  view_1298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_105: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_1300, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_1300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_242: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_105, 4);  as_strided_105 = None
        permute_1053: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_242, [0, 1, 2, 4, 3]);  unsqueeze_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_90: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_11: "b8[1, 4096]" = torch.ops.aten.ne.Scalar(arg7_1, 0);  arg7_1 = None
        unsqueeze_240: "b8[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(ne_11, 2);  ne_11 = None
        unsqueeze_241: "b8[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_398: "f16[1, 4096, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_241, torch.float16)
        where_91: "f16[1, 4096, 1, 1]" = torch.ops.aten.where.self(unsqueeze_241, full_default_90, convert_element_type_398);  unsqueeze_241 = full_default_90 = convert_element_type_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1052: "f16[1, 1, 4096, 1]" = torch.ops.aten.permute.default(where_91, [0, 2, 1, 3]);  where_91 = None
        view_1299: "f16[1, 4096, 1]" = torch.ops.aten.reshape.default(permute_1052, [1, 4096, 1]);  permute_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1301: "f16[1, 8, 512, 1]" = torch.ops.aten.reshape.default(view_1299, [1, 8, 512, 1]);  view_1299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_106: "f16[1, 15, 512, 1]" = torch.ops.aten.as_strided.default(view_1301, [1, 15, 512, 1], [4096, 256, 1, 1]);  view_1301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_243: "f16[1, 15, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_106, 4);  as_strided_106 = None
        permute_1054: "f16[1, 15, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_243, [0, 1, 4, 2, 3]);  unsqueeze_243 = None
        mul_88: "f16[1, 15, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_1053, permute_1054);  permute_1053 = permute_1054 = None
        view_1302: "f16[1, 15, 512, 512]" = torch.ops.aten.reshape.default(mul_88, [1, 15, 512, 512]);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_45: "f16[1, 15, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1302, [0, 0, 0, 1], 0.0);  view_1302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1303: "f16[1, 15, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_45, [1, 15, 512, 513]);  constant_pad_nd_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1357: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1303, 2, 0, 256)
        slice_1358: "f16[1, 15, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1357, 3, 0, 257);  slice_1357 = None
        copy_138: "f16[1, 15, 256, 257]" = torch.ops.aten.copy.default(slice_1360, slice_1358);  slice_1360 = slice_1358 = None
        slice_scatter_253: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1361, copy_138, 3, 256, 9223372036854775807);  slice_1361 = copy_138 = None
        slice_scatter_254: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(full_104, slice_scatter_253, 1, 0, -1);  full_104 = slice_scatter_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_303: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_254, 1, -1)
        select_302: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_254, 1, -1)
        slice_1367: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_302, 2, 256, 9223372036854775807);  select_302 = None
        select_299: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_1303, 1, -1)
        slice_1364: "f16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_299, 1, 256, 9223372036854775807);  select_299 = None
        slice_1365: "f16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1364, 2, 0, 257);  slice_1364 = None
        copy_139: "f16[1, 256, 257]" = torch.ops.aten.copy.default(slice_1367, slice_1365);  slice_1367 = slice_1365 = None
        slice_scatter_255: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_303, copy_139, 2, 256, 9223372036854775807);  select_303 = copy_139 = None
        select_scatter_46: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_254, slice_scatter_255, 1, -1);  slice_scatter_254 = slice_scatter_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1376: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_46, 1, 1, 9223372036854775807)
        slice_1374: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_46, 1, 1, 9223372036854775807)
        slice_1375: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1374, 3, 0, 256);  slice_1374 = None
        slice_1369: "f16[1, 15, 256, 513]" = torch.ops.aten.slice.Tensor(view_1303, 2, -257, -1)
        slice_1370: "f16[1, 15, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1369, 3, 257, 9223372036854775807);  slice_1369 = None
        copy_140: "f16[1, 15, 256, 256]" = torch.ops.aten.copy.default(slice_1375, slice_1370);  slice_1375 = slice_1370 = None
        slice_scatter_256: "f16[1, 15, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1376, copy_140, 3, 0, 256);  slice_1376 = copy_140 = None
        slice_scatter_257: "f16[1, 16, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_46, slice_scatter_256, 1, 1, 9223372036854775807);  select_scatter_46 = slice_scatter_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_310: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_257, 1, 0)
        slice_1386: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_310, 1, 1, 256)
        select_309: "f16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_257, 1, 0)
        slice_1384: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_309, 1, 1, 256);  select_309 = None
        slice_1385: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1384, 2, 1, 256);  slice_1384 = None
        select_305: "f16[1, 512, 513]" = torch.ops.aten.select.int(view_1303, 1, 0);  view_1303 = None
        slice_1379: "f16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_305, 1, 0, 255);  select_305 = None
        slice_1380: "f16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1379, 2, -255, 9223372036854775807);  slice_1379 = None
        copy_141: "f16[1, 255, 255]" = torch.ops.aten.copy.default(slice_1385, slice_1380);  slice_1385 = slice_1380 = None
        slice_scatter_258: "f16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1386, copy_141, 2, 1, 256);  slice_1386 = copy_141 = None
        slice_scatter_259: "f16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_310, slice_scatter_258, 1, 1, 256);  select_310 = slice_scatter_258 = None
        select_scatter_47: "f16[1, 16, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_257, slice_scatter_259, 1, 0);  slice_scatter_257 = slice_scatter_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1311: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_47, [1, 1, 4096, 513])
        permute_1062: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1311, [0, 2, 1, 3]);  view_1311 = None
        slice_1399: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_1062, 1, 0, 256)
        view_1310: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_47, [1, 1, 4096, 513])
        permute_1061: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1310, [0, 2, 1, 3]);  view_1310 = None
        slice_1397: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_1061, 1, 0, 256);  permute_1061 = None
        slice_1398: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1397, 3, 0, 257);  slice_1397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_46: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_244: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_46, -2);  iota_46 = None
        iota_47: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_245: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
        sub_91: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_244, unsqueeze_245);  unsqueeze_244 = unsqueeze_245 = None
        le_23: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_91, 0);  sub_91 = None
        full_default_91: "f16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_92: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_92: "f16[256, 257]" = torch.ops.aten.where.self(le_23, full_default_91, full_default_92);  le_23 = full_default_91 = full_default_92 = None
        rev_46: "f16[256, 257]" = torch.ops.prims.rev.default(where_92, [0]);  where_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_246: "f16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_46, 0);  rev_46 = None
        unsqueeze_247: "f16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_46: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_247, [1, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_399: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_46, torch.bool);  expand_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_93: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1308: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(select_scatter_47, [1, 1, 4096, 513]);  select_scatter_47 = None
        permute_1058: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1308, [0, 2, 1, 3]);  view_1308 = None
        slice_1392: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_1058, 1, 0, 256);  permute_1058 = None
        slice_1393: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1392, 3, 0, 257);  slice_1392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_93: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_399, full_default_93, slice_1393);  convert_element_type_399 = full_default_93 = slice_1393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_142: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1398, where_93);  slice_1398 = where_93 = None
        slice_scatter_260: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1399, copy_142, 3, 0, 257);  slice_1399 = copy_142 = None
        slice_scatter_261: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_1062, slice_scatter_260, 1, 0, 256);  permute_1062 = slice_scatter_260 = None
        permute_1063: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_261, [0, 2, 1, 3]);  slice_scatter_261 = None
        view_1312: "f16[1, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_1063, [1, 16, 256, 513]);  permute_1063 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1319: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_1312, [1, 1, 4096, 513])
        permute_1071: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1319, [0, 2, 1, 3]);  view_1319 = None
        slice_1412: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_1071, 1, -256, 9223372036854775807)
        view_1318: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_1312, [1, 1, 4096, 513])
        permute_1070: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1318, [0, 2, 1, 3]);  view_1318 = None
        slice_1410: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_1070, 1, -256, 9223372036854775807);  permute_1070 = None
        slice_1411: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1410, 3, -257, 9223372036854775807);  slice_1410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_47: "f16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_247, [1, 3]);  unsqueeze_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_47: "f16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_47, [1, 256, 1, 257]);  rev_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_400: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_47, torch.bool);  expand_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_94: "f16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1316: "f16[1, 1, 4096, 513]" = torch.ops.aten.reshape.default(view_1312, [1, 1, 4096, 513]);  view_1312 = None
        permute_1067: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(view_1316, [0, 2, 1, 3]);  view_1316 = None
        slice_1405: "f16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_1067, 1, -256, 9223372036854775807);  permute_1067 = None
        slice_1406: "f16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1405, 3, -257, 9223372036854775807);  slice_1405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_94: "f16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_400, full_default_94, slice_1406);  convert_element_type_400 = full_default_94 = slice_1406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_143: "f16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1411, where_94);  slice_1411 = where_94 = None
        slice_scatter_262: "f16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1412, copy_143, 3, -257, 9223372036854775807);  slice_1412 = copy_143 = None
        slice_scatter_263: "f16[1, 4096, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_1071, slice_scatter_262, 1, -256, 9223372036854775807);  permute_1071 = slice_scatter_262 = None
        permute_1072: "f16[1, 1, 4096, 513]" = torch.ops.aten.permute.default(slice_scatter_263, [0, 2, 1, 3]);  slice_scatter_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1075: "f16[1, 4096, 1, 513]" = torch.ops.aten.permute.default(permute_1072, [0, 2, 1, 3]);  permute_1072 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_123: "f16[1, 4096, 12, 513]" = torch.ops.aten.add.Tensor(permute_1074, permute_1075);  permute_1074 = permute_1075 = None
        permute_1076: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(add_123, [0, 2, 1, 3]);  add_123 = None
        permute_1077: "f16[1, 4096, 12, 513]" = torch.ops.aten.permute.default(permute_1076, [0, 2, 1, 3]);  permute_1076 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_138: "f16[1, 4096, 12, 513]" = torch.ops.aten.clone.default(permute_1077, memory_format = torch.contiguous_format);  permute_1077 = None
        convert_element_type_401: "f32[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(clone_138, torch.float32);  clone_138 = None
        amax_11: "f32[1, 4096, 12, 1]" = torch.ops.aten.amax.default(convert_element_type_401, [-1], True)
        sub_92: "f32[1, 4096, 12, 513]" = torch.ops.aten.sub.Tensor(convert_element_type_401, amax_11);  convert_element_type_401 = amax_11 = None
        exp_11: "f32[1, 4096, 12, 513]" = torch.ops.aten.exp.default(sub_92);  sub_92 = None
        sum_12: "f32[1, 4096, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_117: "f32[1, 4096, 12, 513]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_95: "f32[1, 4096, 12, 513]" = torch.ops.aten.where.self(unsqueeze_249, full_default_95, div_117);  unsqueeze_249 = full_default_95 = div_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_402: "f16[1, 4096, 12, 513]" = torch.ops.prims.convert_element_type.default(where_95, torch.float16);  where_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_1079: "f16[1, 12, 4096, 513]" = torch.ops.aten.permute.default(convert_element_type_402, [0, 2, 1, 3]);  convert_element_type_402 = None
        view_1328: "f16[12, 16, 256, 513]" = torch.ops.aten.reshape.default(permute_1079, [12, 16, 256, 513]);  permute_1079 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_47: "f16[12, 16, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_1328, [0, 257], 0.0);  view_1328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1330: "f16[12, 16, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_47, [12, 16, -1]);  constant_pad_nd_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1415: "f16[12, 16, 196864]" = torch.ops.aten.slice.Tensor(view_1330, 2, 0, -256);  view_1330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1331: "f16[12, 16, 256, 769]" = torch.ops.aten.reshape.default(slice_1415, [12, 16, 256, 769]);  slice_1415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1416: "f16[12, 16, 256, 768]" = torch.ops.aten.slice.Tensor(view_1331, 3, 0, -1);  view_1331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_250: "f16[12, 16, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1416, 4);  slice_1416 = None
        view_1332: "f16[192, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_250, [192, 256, 768]);  unsqueeze_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        view_1236: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_1001, [4096, 768]);  permute_1001 = None
        permute_1004: "f16[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_68: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg184_1, view_1236, permute_1004);  arg184_1 = view_1236 = permute_1004 = None
        view_1237: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_68, [4096, 1, 768]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1327: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(view_1237, [4096, 1, 12, 64]);  view_1237 = None
        permute_1078: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1327, [1, 0, 2, 3]);  view_1327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1080: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_1078, [0, 2, 1, 3]);  permute_1078 = None
        view_1329: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_1080, [12, 4096, 64]);  permute_1080 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_46: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(view_1329, [0, 0, 256, 256], -1.0);  view_1329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_107: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_46, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_251: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_107, 4);  as_strided_107 = None
        clone_140: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_251, memory_format = torch.contiguous_format);  unsqueeze_251 = None
        view_1333: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_140, [192, 768, 64]);  clone_140 = None
        bmm_23: "f16[192, 256, 64]" = torch.ops.aten.bmm.default(view_1332, view_1333);  view_1332 = view_1333 = None
        view_1334: "f16[12, 16, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_23, [12, 16, 256, 1, 64]);  bmm_23 = None
        permute_1085: "f16[12, 16, 256, 64, 1]" = torch.ops.aten.permute.default(view_1334, [0, 1, 2, 4, 3]);  view_1334 = None
        view_1335: "f16[12, 16, 256, 64]" = torch.ops.aten.reshape.default(permute_1085, [12, 16, 256, 64]);  permute_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1336: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(view_1335, [1, 12, 4096, 64]);  view_1335 = None
        permute_1086: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(view_1336, [0, 2, 1, 3]);  view_1336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_1087: "f16[4096, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1086, [1, 0, 2, 3]);  permute_1086 = None
        clone_141: "f16[4096, 1, 12, 64]" = torch.ops.aten.clone.default(permute_1087, memory_format = torch.contiguous_format);  permute_1087 = None
        view_1337: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(clone_141, [4096, 1, 768]);  clone_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1088: "f16[1, 4096, 768]" = torch.ops.aten.permute.default(view_1337, [1, 0, 2]);  view_1337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        view_1338: "f16[4096, 768]" = torch.ops.aten.reshape.default(permute_1088, [4096, 768]);  permute_1088 = None
        permute_1089: "f16[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_69: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg186_1, view_1338, permute_1089);  arg186_1 = view_1338 = permute_1089 = None
        view_1339: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_69, [1, 4096, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_125: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_1339, convert_element_type_384);  view_1339 = convert_element_type_384 = None
        convert_element_type_408: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_125, torch.float32);  add_125 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_408, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 4096, 1]" = var_mean_22[0]
        getitem_45: "f32[1, 4096, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_94: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_408, getitem_45);  convert_element_type_408 = getitem_45 = None
        add_126: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_89: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_22);  sub_94 = rsqrt_22 = None
        mul_90: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_89, arg187_1);  mul_89 = arg187_1 = None
        add_127: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_90, arg188_1);  mul_90 = arg188_1 = None
        convert_element_type_409: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_127, torch.float16);  add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1340: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_409, [4096, 768])
        permute_1090: "f16[768, 3072]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_70: "f16[4096, 3072]" = torch.ops.aten.addmm.default(arg190_1, view_1340, permute_1090);  arg190_1 = view_1340 = permute_1090 = None
        view_1341: "f16[1, 4096, 3072]" = torch.ops.aten.reshape.default(addmm_70, [1, 4096, 3072]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_413: "f32[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(view_1341, torch.float32);  view_1341 = None
        mul_91: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_413, 0.5)
        mul_92: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_413, 0.7071067811865476);  convert_element_type_413 = None
        erf_11: "f32[1, 4096, 3072]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_128: "f32[1, 4096, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_93: "f32[1, 4096, 3072]" = torch.ops.aten.mul.Tensor(mul_91, add_128);  mul_91 = add_128 = None
        convert_element_type_414: "f16[1, 4096, 3072]" = torch.ops.prims.convert_element_type.default(mul_93, torch.float16);  mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1342: "f16[4096, 3072]" = torch.ops.aten.reshape.default(convert_element_type_414, [4096, 3072]);  convert_element_type_414 = None
        permute_1091: "f16[3072, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_71: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg192_1, view_1342, permute_1091);  arg192_1 = view_1342 = permute_1091 = None
        view_1343: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm_71, [1, 4096, 768]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_129: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_1343, convert_element_type_409);  view_1343 = convert_element_type_409 = None
        convert_element_type_418: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_129, torch.float32);  add_129 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_418, [2], correction = 0, keepdim = True)
        getitem_46: "f32[1, 4096, 1]" = var_mean_23[0]
        getitem_47: "f32[1, 4096, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_95: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_418, getitem_47);  convert_element_type_418 = getitem_47 = None
        add_130: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_94: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_23);  sub_95 = rsqrt_23 = None
        mul_95: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_94, arg193_1);  mul_94 = arg193_1 = None
        add_131: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_95, arg194_1);  mul_95 = arg194_1 = None
        convert_element_type_419: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_131, torch.float16);  add_131 = None
        return (convert_element_type_419,)

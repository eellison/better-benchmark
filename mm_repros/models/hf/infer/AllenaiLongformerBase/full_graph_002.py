class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8, 1024, 768][786432, 768, 1]cuda:0", arg1_1: "bf16[768, 768][768, 1]cuda:0", arg2_1: "bf16[768][1]cuda:0", arg3_1: "bf16[768, 768][768, 1]cuda:0", arg4_1: "bf16[768][1]cuda:0", arg5_1: "bf16[768, 768][768, 1]cuda:0", arg6_1: "bf16[768][1]cuda:0", arg7_1: "bf16[8, 1024][1024, 1]cuda:0", arg8_1: "b8[8, 1024][1024, 1]cuda:0", arg9_1: "bf16[768, 768][768, 1]cuda:0", arg10_1: "bf16[768][1]cuda:0", arg11_1: "bf16[768][1]cuda:0", arg12_1: "bf16[768][1]cuda:0", arg13_1: "bf16[3072, 768][768, 1]cuda:0", arg14_1: "bf16[3072][1]cuda:0", arg15_1: "bf16[768, 3072][3072, 1]cuda:0", arg16_1: "bf16[768][1]cuda:0", arg17_1: "bf16[768][1]cuda:0", arg18_1: "bf16[768][1]cuda:0", arg19_1: "bf16[768, 768][768, 1]cuda:0", arg20_1: "bf16[768][1]cuda:0", arg21_1: "bf16[768, 768][768, 1]cuda:0", arg22_1: "bf16[768][1]cuda:0", arg23_1: "bf16[768, 768][768, 1]cuda:0", arg24_1: "bf16[768][1]cuda:0", arg25_1: "bf16[768, 768][768, 1]cuda:0", arg26_1: "bf16[768][1]cuda:0", arg27_1: "bf16[768][1]cuda:0", arg28_1: "bf16[768][1]cuda:0", arg29_1: "bf16[3072, 768][768, 1]cuda:0", arg30_1: "bf16[3072][1]cuda:0", arg31_1: "bf16[768, 3072][3072, 1]cuda:0", arg32_1: "bf16[768][1]cuda:0", arg33_1: "bf16[768][1]cuda:0", arg34_1: "bf16[768][1]cuda:0", arg35_1: "bf16[768, 768][768, 1]cuda:0", arg36_1: "bf16[768][1]cuda:0", arg37_1: "bf16[768, 768][768, 1]cuda:0", arg38_1: "bf16[768][1]cuda:0", arg39_1: "bf16[768, 768][768, 1]cuda:0", arg40_1: "bf16[768][1]cuda:0", arg41_1: "bf16[768, 768][768, 1]cuda:0", arg42_1: "bf16[768][1]cuda:0", arg43_1: "bf16[768][1]cuda:0", arg44_1: "bf16[768][1]cuda:0", arg45_1: "bf16[3072, 768][768, 1]cuda:0", arg46_1: "bf16[3072][1]cuda:0", arg47_1: "bf16[768, 3072][3072, 1]cuda:0", arg48_1: "bf16[768][1]cuda:0", arg49_1: "bf16[768][1]cuda:0", arg50_1: "bf16[768][1]cuda:0", arg51_1: "bf16[768, 768][768, 1]cuda:0", arg52_1: "bf16[768][1]cuda:0", arg53_1: "bf16[768, 768][768, 1]cuda:0", arg54_1: "bf16[768][1]cuda:0", arg55_1: "bf16[768, 768][768, 1]cuda:0", arg56_1: "bf16[768][1]cuda:0", arg57_1: "bf16[768, 768][768, 1]cuda:0", arg58_1: "bf16[768][1]cuda:0", arg59_1: "bf16[768][1]cuda:0", arg60_1: "bf16[768][1]cuda:0", arg61_1: "bf16[3072, 768][768, 1]cuda:0", arg62_1: "bf16[3072][1]cuda:0", arg63_1: "bf16[768, 3072][3072, 1]cuda:0", arg64_1: "bf16[768][1]cuda:0", arg65_1: "bf16[768][1]cuda:0", arg66_1: "bf16[768][1]cuda:0", arg67_1: "bf16[768, 768][768, 1]cuda:0", arg68_1: "bf16[768][1]cuda:0", arg69_1: "bf16[768, 768][768, 1]cuda:0", arg70_1: "bf16[768][1]cuda:0", arg71_1: "bf16[768, 768][768, 1]cuda:0", arg72_1: "bf16[768][1]cuda:0", arg73_1: "bf16[768, 768][768, 1]cuda:0", arg74_1: "bf16[768][1]cuda:0", arg75_1: "bf16[768][1]cuda:0", arg76_1: "bf16[768][1]cuda:0", arg77_1: "bf16[3072, 768][768, 1]cuda:0", arg78_1: "bf16[3072][1]cuda:0", arg79_1: "bf16[768, 3072][3072, 1]cuda:0", arg80_1: "bf16[768][1]cuda:0", arg81_1: "bf16[768][1]cuda:0", arg82_1: "bf16[768][1]cuda:0", arg83_1: "bf16[768, 768][768, 1]cuda:0", arg84_1: "bf16[768][1]cuda:0", arg85_1: "bf16[768, 768][768, 1]cuda:0", arg86_1: "bf16[768][1]cuda:0", arg87_1: "bf16[768, 768][768, 1]cuda:0", arg88_1: "bf16[768][1]cuda:0", arg89_1: "bf16[768, 768][768, 1]cuda:0", arg90_1: "bf16[768][1]cuda:0", arg91_1: "bf16[768][1]cuda:0", arg92_1: "bf16[768][1]cuda:0", arg93_1: "bf16[3072, 768][768, 1]cuda:0", arg94_1: "bf16[3072][1]cuda:0", arg95_1: "bf16[768, 3072][3072, 1]cuda:0", arg96_1: "bf16[768][1]cuda:0", arg97_1: "bf16[768][1]cuda:0", arg98_1: "bf16[768][1]cuda:0", arg99_1: "bf16[768, 768][768, 1]cuda:0", arg100_1: "bf16[768][1]cuda:0", arg101_1: "bf16[768, 768][768, 1]cuda:0", arg102_1: "bf16[768][1]cuda:0", arg103_1: "bf16[768, 768][768, 1]cuda:0", arg104_1: "bf16[768][1]cuda:0", arg105_1: "bf16[768, 768][768, 1]cuda:0", arg106_1: "bf16[768][1]cuda:0", arg107_1: "bf16[768][1]cuda:0", arg108_1: "bf16[768][1]cuda:0", arg109_1: "bf16[3072, 768][768, 1]cuda:0", arg110_1: "bf16[3072][1]cuda:0", arg111_1: "bf16[768, 3072][3072, 1]cuda:0", arg112_1: "bf16[768][1]cuda:0", arg113_1: "bf16[768][1]cuda:0", arg114_1: "bf16[768][1]cuda:0", arg115_1: "bf16[768, 768][768, 1]cuda:0", arg116_1: "bf16[768][1]cuda:0", arg117_1: "bf16[768, 768][768, 1]cuda:0", arg118_1: "bf16[768][1]cuda:0", arg119_1: "bf16[768, 768][768, 1]cuda:0", arg120_1: "bf16[768][1]cuda:0", arg121_1: "bf16[768, 768][768, 1]cuda:0", arg122_1: "bf16[768][1]cuda:0", arg123_1: "bf16[768][1]cuda:0", arg124_1: "bf16[768][1]cuda:0", arg125_1: "bf16[3072, 768][768, 1]cuda:0", arg126_1: "bf16[3072][1]cuda:0", arg127_1: "bf16[768, 3072][3072, 1]cuda:0", arg128_1: "bf16[768][1]cuda:0", arg129_1: "bf16[768][1]cuda:0", arg130_1: "bf16[768][1]cuda:0", arg131_1: "bf16[768, 768][768, 1]cuda:0", arg132_1: "bf16[768][1]cuda:0", arg133_1: "bf16[768, 768][768, 1]cuda:0", arg134_1: "bf16[768][1]cuda:0", arg135_1: "bf16[768, 768][768, 1]cuda:0", arg136_1: "bf16[768][1]cuda:0", arg137_1: "bf16[768, 768][768, 1]cuda:0", arg138_1: "bf16[768][1]cuda:0", arg139_1: "bf16[768][1]cuda:0", arg140_1: "bf16[768][1]cuda:0", arg141_1: "bf16[3072, 768][768, 1]cuda:0", arg142_1: "bf16[3072][1]cuda:0", arg143_1: "bf16[768, 3072][3072, 1]cuda:0", arg144_1: "bf16[768][1]cuda:0", arg145_1: "bf16[768][1]cuda:0", arg146_1: "bf16[768][1]cuda:0", arg147_1: "bf16[768, 768][768, 1]cuda:0", arg148_1: "bf16[768][1]cuda:0", arg149_1: "bf16[768, 768][768, 1]cuda:0", arg150_1: "bf16[768][1]cuda:0", arg151_1: "bf16[768, 768][768, 1]cuda:0", arg152_1: "bf16[768][1]cuda:0", arg153_1: "bf16[768, 768][768, 1]cuda:0", arg154_1: "bf16[768][1]cuda:0", arg155_1: "bf16[768][1]cuda:0", arg156_1: "bf16[768][1]cuda:0", arg157_1: "bf16[3072, 768][768, 1]cuda:0", arg158_1: "bf16[3072][1]cuda:0", arg159_1: "bf16[768, 3072][3072, 1]cuda:0", arg160_1: "bf16[768][1]cuda:0", arg161_1: "bf16[768][1]cuda:0", arg162_1: "bf16[768][1]cuda:0", arg163_1: "bf16[768, 768][768, 1]cuda:0", arg164_1: "bf16[768][1]cuda:0", arg165_1: "bf16[768, 768][768, 1]cuda:0", arg166_1: "bf16[768][1]cuda:0", arg167_1: "bf16[768, 768][768, 1]cuda:0", arg168_1: "bf16[768][1]cuda:0", arg169_1: "bf16[768, 768][768, 1]cuda:0", arg170_1: "bf16[768][1]cuda:0", arg171_1: "bf16[768][1]cuda:0", arg172_1: "bf16[768][1]cuda:0", arg173_1: "bf16[3072, 768][768, 1]cuda:0", arg174_1: "bf16[3072][1]cuda:0", arg175_1: "bf16[768, 3072][3072, 1]cuda:0", arg176_1: "bf16[768][1]cuda:0", arg177_1: "bf16[768][1]cuda:0", arg178_1: "bf16[768][1]cuda:0", arg179_1: "bf16[768, 768][768, 1]cuda:0", arg180_1: "bf16[768][1]cuda:0", arg181_1: "bf16[768, 768][768, 1]cuda:0", arg182_1: "bf16[768][1]cuda:0", arg183_1: "bf16[768, 768][768, 1]cuda:0", arg184_1: "bf16[768][1]cuda:0", arg185_1: "bf16[768, 768][768, 1]cuda:0", arg186_1: "bf16[768][1]cuda:0", arg187_1: "bf16[768][1]cuda:0", arg188_1: "bf16[768][1]cuda:0", arg189_1: "bf16[3072, 768][768, 1]cuda:0", arg190_1: "bf16[3072][1]cuda:0", arg191_1: "bf16[768, 3072][3072, 1]cuda:0", arg192_1: "bf16[768][1]cuda:0", arg193_1: "bf16[768][1]cuda:0", arg194_1: "bf16[768][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_7: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_8: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_17: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_18: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        full_default_3: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_5: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_3: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_4: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_3, 3, 256, 9223372036854775807);  slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [8192, 768]);  clone = None
        permute_1: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view, permute_1);  view = permute_1 = None
        view_1: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [1024, 8, 768]);  mm = None
        add: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add, 8.0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_29: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div, [1024, 8, 12, 64]);  div = None
        permute_25: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [1, 0, 2, 3]);  view_29 = None
        permute_26: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_25, [0, 2, 1, 3]);  permute_25 = None
        view_30: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_26, [96, 1024, 64]);  permute_26 = None
        view_31: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [96, 2, 512, 64]);  view_30 = None
        as_strided_5: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_31, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_31 = None
        unsqueeze_4: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_5, 4);  as_strided_5 = None
        clone_3: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_4, memory_format = torch.contiguous_format);  unsqueeze_4 = None
        view_32: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [288, 512, 64]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_1: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view_2: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [8192, 768]);  clone_1 = None
        permute_2: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm_1: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2, permute_2);  view_2 = permute_2 = None
        view_3: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [1024, 8, 768]);  mm_1 = None
        add_1: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_3, arg4_1);  view_3 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_8: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_1, [1024, 8, 12, 64]);  add_1 = None
        permute_5: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [1, 0, 2, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_10: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        view_12: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_10, [96, 1024, 64]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_19: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_12, [96, 2, 512, 64]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_1: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_19, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_2: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        permute_20: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 4, 2, 3]);  unsqueeze_2 = None
        permute_29: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_20, [0, 1, 4, 3, 2]);  permute_20 = None
        clone_4: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_33: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [288, 64, 512]);  clone_4 = None
        bmm: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_32, view_33);  view_32 = view_33 = None
        view_34: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [96, 3, 512, 1, 512]);  bmm = None
        permute_30: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 1, 2, 4, 3]);  view_34 = None
        view_35: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_30, [96, 3, 512, 512]);  permute_30 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_35, [0, 0, 0, 1], 0.0);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_36: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd, [96, 3, 512, 513]);  constant_pad_nd = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_36, 2, 0, 256)
        slice_2: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 257);  slice_1 = None
        copy: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_4, slice_2);  slice_4 = slice_2 = None
        slice_scatter: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_5, copy, 3, 256, 9223372036854775807);  slice_5 = copy = None
        slice_scatter_1: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full, slice_scatter, 1, 0, -1);  full = slice_scatter = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_4: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        select_3: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        slice_11: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_3, 2, 256, 9223372036854775807);  select_3 = None
        select: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_36, 1, -1)
        slice_8: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807);  select = None
        slice_9: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_8, 2, 0, 257);  slice_8 = None
        copy_1: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_11, slice_9);  slice_11 = slice_9 = None
        slice_scatter_2: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_4, copy_1, 2, 256, 9223372036854775807);  select_4 = copy_1 = None
        select_scatter: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_1, slice_scatter_2, 1, -1);  slice_scatter_1 = slice_scatter_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_20: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_18: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_19: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_18, 3, 0, 256);  slice_18 = None
        slice_13: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_36, 2, -257, -1)
        slice_14: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_13, 3, 257, 9223372036854775807);  slice_13 = None
        copy_2: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_19, slice_14);  slice_19 = slice_14 = None
        slice_scatter_3: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_20, copy_2, 3, 0, 256);  slice_20 = copy_2 = None
        slice_scatter_4: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_3, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_11: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_30: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_11, 1, 1, 256)
        select_10: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_28: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_10, 1, 1, 256);  select_10 = None
        slice_29: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_28, 2, 1, 256);  slice_28 = None
        select_6: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_36, 1, 0);  view_36 = None
        slice_23: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_6, 1, 0, 255);  select_6 = None
        slice_24: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_23, 2, -255, 9223372036854775807);  slice_23 = None
        copy_3: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_29, slice_24);  slice_29 = slice_24 = None
        slice_scatter_5: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_30, copy_3, 2, 1, 256);  slice_30 = copy_3 = None
        slice_scatter_6: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_11, slice_scatter_5, 1, 1, 256);  select_11 = slice_scatter_5 = None
        select_scatter_1: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_4, slice_scatter_6, 1, 0);  slice_scatter_4 = slice_scatter_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_44: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_1, [8, 12, 1024, 513])
        permute_38: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        slice_43: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_38, 1, 0, 256)
        view_43: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_1, [8, 12, 1024, 513])
        permute_37: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        slice_41: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_37, 1, 0, 256);  permute_37 = None
        slice_42: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_41, 3, 0, 257);  slice_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_5: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub_1: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_5, unsqueeze_6);  unsqueeze_5 = unsqueeze_6 = None
        le: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_1, 0);  sub_1 = None
        full_1: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le, full_1, scalar_tensor);  le = full_1 = scalar_tensor = None
        rev: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where, [0]);  where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_7: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
        unsqueeze_8: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_8, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_8: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_2: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_35: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_2, [0, 2, 1, 3]);  full_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_41: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_1, [8, 12, 1024, 513]);  select_scatter_1 = None
        permute_34: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        slice_36: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_34, 1, 0, 256);  permute_34 = None
        slice_37: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_36, 3, 0, 257);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_1: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_8, permute_35, slice_37);  convert_element_type_8 = permute_35 = slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_4: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_42, where_1);  slice_42 = where_1 = None
        slice_scatter_7: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_43, copy_4, 3, 0, 257);  slice_43 = copy_4 = None
        slice_scatter_8: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_38, slice_scatter_7, 1, 0, 256);  permute_38 = slice_scatter_7 = None
        permute_39: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3]);  slice_scatter_8 = None
        view_45: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [96, 4, 256, 513]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_52: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [8, 12, 1024, 513])
        permute_47: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        slice_56: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_47, 1, -256, 9223372036854775807)
        view_51: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [8, 12, 1024, 513])
        permute_46: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        slice_54: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_46, 1, -256, 9223372036854775807);  permute_46 = None
        slice_55: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_54, 3, -257, 9223372036854775807);  slice_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_1: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_8, [1, 3]);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_1: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_1, [8, 256, 12, 257]);  rev_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_9: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_3: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_44: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_3, [0, 2, 1, 3]);  full_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_49: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [8, 12, 1024, 513]);  view_45 = None
        permute_43: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        slice_49: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_43, 1, -256, 9223372036854775807);  permute_43 = None
        slice_50: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_49, 3, -257, 9223372036854775807);  slice_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_2: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_9, permute_44, slice_50);  convert_element_type_9 = permute_44 = slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_5: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_55, where_2);  slice_55 = where_2 = None
        slice_scatter_9: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_56, copy_5, 3, -257, 9223372036854775807);  slice_56 = copy_5 = None
        slice_scatter_10: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_47, slice_scatter_9, 1, -256, 9223372036854775807);  permute_47 = slice_scatter_9 = None
        permute_48: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3]);  slice_scatter_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_73: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_48, [0, 2, 1, 3]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_5: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_63: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_61: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_62: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_61, 3, 256, 9223372036854775807);  slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_4: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_50: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
        view_55: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_50, [8, 1024, 1]);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_57: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_55, [8, 2, 512, 1]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_6: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_57, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_11: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_6, 4);  as_strided_6 = None
        permute_52: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_11, [0, 1, 2, 4, 3]);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_9: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne, 2);  ne = None
        unsqueeze_10: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_10: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_10, torch.bfloat16)
        where_3: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_10, full_default, convert_element_type_10);  unsqueeze_10 = full_default = convert_element_type_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_51: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
        view_56: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [8, 1024, 1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_58: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_56, [8, 2, 512, 1]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_7: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_58, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_12: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_7, 4);  as_strided_7 = None
        permute_53: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_12, [0, 1, 4, 2, 3]);  unsqueeze_12 = None
        mul: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_52, permute_53);  permute_52 = permute_53 = None
        view_59: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul, [8, 3, 512, 512]);  mul = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_59, [0, 0, 0, 1], 0.0);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_60: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_1, [8, 3, 512, 513]);  constant_pad_nd_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_59: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_60, 2, 0, 256)
        slice_60: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_59, 3, 0, 257);  slice_59 = None
        copy_6: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_62, slice_60);  slice_62 = slice_60 = None
        slice_scatter_11: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_63, copy_6, 3, 256, 9223372036854775807);  slice_63 = copy_6 = None
        slice_scatter_12: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_11, 1, 0, -1);  full_5 = slice_scatter_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_17: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        select_16: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        slice_69: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_16, 2, 256, 9223372036854775807);  select_16 = None
        select_13: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_60, 1, -1)
        slice_66: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_13, 1, 256, 9223372036854775807);  select_13 = None
        slice_67: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_66, 2, 0, 257);  slice_66 = None
        copy_7: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_69, slice_67);  slice_69 = slice_67 = None
        slice_scatter_13: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_17, copy_7, 2, 256, 9223372036854775807);  select_17 = copy_7 = None
        select_scatter_2: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_12, slice_scatter_13, 1, -1);  slice_scatter_12 = slice_scatter_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_78: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_76: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_77: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_76, 3, 0, 256);  slice_76 = None
        slice_71: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_60, 2, -257, -1)
        slice_72: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_71, 3, 257, 9223372036854775807);  slice_71 = None
        copy_8: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_77, slice_72);  slice_77 = slice_72 = None
        slice_scatter_14: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_78, copy_8, 3, 0, 256);  slice_78 = copy_8 = None
        slice_scatter_15: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_2, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter_2 = slice_scatter_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_24: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_88: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_24, 1, 1, 256)
        select_23: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_86: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_23, 1, 1, 256);  select_23 = None
        slice_87: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_86, 2, 1, 256);  slice_86 = None
        select_19: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_60, 1, 0);  view_60 = None
        slice_81: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_19, 1, 0, 255);  select_19 = None
        slice_82: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_81, 2, -255, 9223372036854775807);  slice_81 = None
        copy_9: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_87, slice_82);  slice_87 = slice_82 = None
        slice_scatter_16: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_88, copy_9, 2, 1, 256);  slice_88 = copy_9 = None
        slice_scatter_17: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_24, slice_scatter_16, 1, 1, 256);  select_24 = slice_scatter_16 = None
        select_scatter_3: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_17, 1, 0);  slice_scatter_15 = slice_scatter_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_68: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_3, [8, 1, 1024, 513])
        permute_61: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        slice_101: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_61, 1, 0, 256)
        view_67: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_3, [8, 1, 1024, 513])
        permute_60: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
        slice_99: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_60, 1, 0, 256);  permute_60 = None
        slice_100: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_99, 3, 0, 257);  slice_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_2: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_13: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_2, -2);  iota_2 = None
        iota_3: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_14: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
        sub_3: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_13, unsqueeze_14);  unsqueeze_13 = unsqueeze_14 = None
        le_1: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_3, 0);  sub_3 = None
        full_6: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_2: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_4: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_1, full_6, scalar_tensor_2);  le_1 = full_6 = scalar_tensor_2 = None
        rev_2: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_4, [0]);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_15: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_2, 0);  rev_2 = None
        unsqueeze_16: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_2: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_16, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_11: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_1: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_65: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_3, [8, 1, 1024, 513]);  select_scatter_3 = None
        permute_57: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        slice_94: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_57, 1, 0, 256);  permute_57 = None
        slice_95: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_94, 3, 0, 257);  slice_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_5: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_11, full_default_1, slice_95);  convert_element_type_11 = full_default_1 = slice_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_10: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_100, where_5);  slice_100 = where_5 = None
        slice_scatter_18: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_101, copy_10, 3, 0, 257);  slice_101 = copy_10 = None
        slice_scatter_19: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_61, slice_scatter_18, 1, 0, 256);  permute_61 = slice_scatter_18 = None
        permute_62: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_19, [0, 2, 1, 3]);  slice_scatter_19 = None
        view_69: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_62, [8, 4, 256, 513]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_76: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_69, [8, 1, 1024, 513])
        permute_70: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        slice_114: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_70, 1, -256, 9223372036854775807)
        view_75: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_69, [8, 1, 1024, 513])
        permute_69: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        slice_112: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_69, 1, -256, 9223372036854775807);  permute_69 = None
        slice_113: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_112, 3, -257, 9223372036854775807);  slice_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_3: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_16, [1, 3]);  unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_3: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_3, [8, 256, 1, 257]);  rev_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_12: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_2: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_73: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_69, [8, 1, 1024, 513]);  view_69 = None
        permute_66: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        slice_107: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_66, 1, -256, 9223372036854775807);  permute_66 = None
        slice_108: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_107, 3, -257, 9223372036854775807);  slice_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_6: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_12, full_default_2, slice_108);  convert_element_type_12 = full_default_2 = slice_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_11: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_113, where_6);  slice_113 = where_6 = None
        slice_scatter_20: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_114, copy_11, 3, -257, 9223372036854775807);  slice_114 = copy_11 = None
        slice_scatter_21: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_70, slice_scatter_20, 1, -256, 9223372036854775807);  permute_70 = slice_scatter_20 = None
        permute_71: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_21, [0, 2, 1, 3]);  slice_scatter_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_74: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_71, [0, 2, 1, 3]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_5: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_73, permute_74);  permute_73 = permute_74 = None
        permute_75: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_5, [0, 2, 1, 3]);  add_5 = None
        permute_76: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_75, [0, 2, 1, 3]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_13: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_76, torch.float32);  permute_76 = None
        clone_9: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_13, memory_format = torch.contiguous_format);  convert_element_type_13 = None
        amax: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_9, [-1], True)
        sub_4: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_9, amax);  clone_9 = amax = None
        exp: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_1: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_7: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_7: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, div_7);  unsqueeze_18 = full_default_3 = div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_14: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.bfloat16);  where_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_78: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_14, [0, 2, 1, 3]);  convert_element_type_14 = None
        clone_11: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_85: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [96, 4, 256, 513]);  clone_11 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_85, [0, 257], 0.0);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_87: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_3, [96, 4, -1]);  constant_pad_nd_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_117: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_87, 2, 0, -256);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_88: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_117, [96, 4, 256, 769]);  slice_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_118: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_88, 3, 0, -1);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_19: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_118, 4);  slice_118 = None
        view_89: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_19, [384, 256, 768]);  unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_2: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [8192, 768]);  clone_2 = None
        permute_3: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_2: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_4, permute_3);  view_4 = permute_3 = None
        view_5: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [1024, 8, 768]);  mm_2 = None
        add_2: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_5, arg6_1);  view_5 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_84: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_2, [1024, 8, 12, 64]);  add_2 = None
        permute_77: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_84, [1, 0, 2, 3]);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_79: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [0, 2, 1, 3]);  permute_77 = None
        view_86: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_79, [96, 1024, 64]);  permute_79 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_86, [0, 0, 256, 256], -1.0);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_8: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_2, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_20: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_8, 4);  as_strided_8 = None
        clone_12: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_20, memory_format = torch.contiguous_format);  unsqueeze_20 = None
        view_90: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [384, 768, 64]);  clone_12 = None
        bmm_1: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_89, view_90);  view_89 = view_90 = None
        view_91: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [96, 4, 256, 1, 64]);  bmm_1 = None
        permute_84: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_91, [0, 1, 2, 4, 3]);  view_91 = None
        view_92: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_84, [96, 4, 256, 64]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_93: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_92, [8, 12, 1024, 64]);  view_92 = None
        permute_85: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_86: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0, 2, 3]);  permute_85 = None
        clone_13: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
        view_94: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [1024, 8, 768]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_87: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_94, [1, 0, 2]);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_14: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
        view_95: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [8192, 768]);  clone_14 = None
        permute_88: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_3: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_95, permute_88);  view_95 = permute_88 = None
        view_96: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [8, 1024, 768]);  mm_3 = None
        add_7: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_96, arg10_1);  view_96 = arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_8: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_7, arg0_1);  add_7 = arg0_1 = None
        convert_element_type_19: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_19, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub_6: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_19, getitem_1);  convert_element_type_19 = getitem_1 = None
        add_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_1: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = rsqrt = None
        mul_2: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, arg11_1);  mul_1 = arg11_1 = None
        add_10: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, arg12_1);  mul_2 = arg12_1 = None
        convert_element_type_20: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_97: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_20, [8192, 768])
        permute_89: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg14_1, view_97, permute_89);  arg14_1 = view_97 = permute_89 = None
        view_98: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [8, 1024, 3072]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_24: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_98, torch.float32);  view_98 = None
        mul_3: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5)
        mul_4: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.7071067811865476);  convert_element_type_24 = None
        erf: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_11: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, add_11);  mul_3 = add_11 = None
        convert_element_type_25: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_99: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_25, [8192, 3072]);  convert_element_type_25 = None
        permute_90: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_1: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, view_99, permute_90);  arg16_1 = view_99 = permute_90 = None
        view_100: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [8, 1024, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_12: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_100, convert_element_type_20);  view_100 = convert_element_type_20 = None
        convert_element_type_29: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.float32);  add_12 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_29, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_16: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_17: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_38: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_39: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 3);  unsqueeze_38 = None
        full_default_7: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_9: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_123: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_9, 1, 0, -1)
        slice_121: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_9, 1, 0, -1)
        slice_122: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_121, 3, 256, 9223372036854775807);  slice_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_7: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_29, getitem_3);  convert_element_type_29 = getitem_3 = None
        add_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_6: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_1);  sub_7 = rsqrt_1 = None
        mul_7: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, arg17_1);  mul_6 = arg17_1 = None
        add_14: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_7, arg18_1);  mul_7 = arg18_1 = None
        convert_element_type_30: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_91: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_17: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format)
        view_101: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [8192, 768]);  clone_17 = None
        permute_92: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_4: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_101, permute_92);  view_101 = permute_92 = None
        view_102: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [1024, 8, 768]);  mm_4 = None
        add_15: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_102, arg20_1);  view_102 = arg20_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_10: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_15, 8.0);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_130: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_10, [1024, 8, 12, 64]);  div_10 = None
        permute_116: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_130, [1, 0, 2, 3]);  view_130 = None
        permute_117: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_116, [0, 2, 1, 3]);  permute_116 = None
        view_131: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_117, [96, 1024, 64]);  permute_117 = None
        view_132: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_131, [96, 2, 512, 64]);  view_131 = None
        as_strided_14: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_132, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_132 = None
        unsqueeze_25: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_14, 4);  as_strided_14 = None
        clone_20: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_25, memory_format = torch.contiguous_format);  unsqueeze_25 = None
        view_133: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [288, 512, 64]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_18: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format)
        view_103: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [8192, 768]);  clone_18 = None
        permute_93: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_5: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_103, permute_93);  view_103 = permute_93 = None
        view_104: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [1024, 8, 768]);  mm_5 = None
        add_16: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_104, arg22_1);  view_104 = arg22_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_109: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_16, [1024, 8, 12, 64]);  add_16 = None
        permute_96: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_109, [1, 0, 2, 3]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_101: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [0, 2, 1, 3]);  permute_96 = None
        view_113: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_101, [96, 1024, 64]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_120: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [96, 2, 512, 64]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_10: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_120, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_23: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_10, 4);  as_strided_10 = None
        permute_111: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_23, [0, 1, 4, 2, 3]);  unsqueeze_23 = None
        permute_120: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_111, [0, 1, 4, 3, 2]);  permute_111 = None
        clone_21: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        view_134: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [288, 64, 512]);  clone_21 = None
        bmm_2: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_133, view_134);  view_133 = view_134 = None
        view_135: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [96, 3, 512, 1, 512]);  bmm_2 = None
        permute_121: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_135, [0, 1, 2, 4, 3]);  view_135 = None
        view_136: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_121, [96, 3, 512, 512]);  permute_121 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_4: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_136, [0, 0, 0, 1], 0.0);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_137: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_4, [96, 3, 512, 513]);  constant_pad_nd_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_119: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_137, 2, 0, 256)
        slice_120: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_119, 3, 0, 257);  slice_119 = None
        copy_12: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_122, slice_120);  slice_122 = slice_120 = None
        slice_scatter_22: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_123, copy_12, 3, 256, 9223372036854775807);  slice_123 = copy_12 = None
        slice_scatter_23: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_22, 1, 0, -1);  full_9 = slice_scatter_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_30: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_23, 1, -1)
        select_29: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_23, 1, -1)
        slice_129: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_29, 2, 256, 9223372036854775807);  select_29 = None
        select_26: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_137, 1, -1)
        slice_126: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_26, 1, 256, 9223372036854775807);  select_26 = None
        slice_127: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_126, 2, 0, 257);  slice_126 = None
        copy_13: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_129, slice_127);  slice_129 = slice_127 = None
        slice_scatter_24: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_30, copy_13, 2, 256, 9223372036854775807);  select_30 = copy_13 = None
        select_scatter_4: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_23, slice_scatter_24, 1, -1);  slice_scatter_23 = slice_scatter_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_138: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_4, 1, 1, 9223372036854775807)
        slice_136: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_4, 1, 1, 9223372036854775807)
        slice_137: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_136, 3, 0, 256);  slice_136 = None
        slice_131: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_137, 2, -257, -1)
        slice_132: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_131, 3, 257, 9223372036854775807);  slice_131 = None
        copy_14: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_137, slice_132);  slice_137 = slice_132 = None
        slice_scatter_25: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_138, copy_14, 3, 0, 256);  slice_138 = copy_14 = None
        slice_scatter_26: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_4, slice_scatter_25, 1, 1, 9223372036854775807);  select_scatter_4 = slice_scatter_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_37: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_26, 1, 0)
        slice_148: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_37, 1, 1, 256)
        select_36: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_26, 1, 0)
        slice_146: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_36, 1, 1, 256);  select_36 = None
        slice_147: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_146, 2, 1, 256);  slice_146 = None
        select_32: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_137, 1, 0);  view_137 = None
        slice_141: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_32, 1, 0, 255);  select_32 = None
        slice_142: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_141, 2, -255, 9223372036854775807);  slice_141 = None
        copy_15: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_147, slice_142);  slice_147 = slice_142 = None
        slice_scatter_27: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_148, copy_15, 2, 1, 256);  slice_148 = copy_15 = None
        slice_scatter_28: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_37, slice_scatter_27, 1, 1, 256);  select_37 = slice_scatter_27 = None
        select_scatter_5: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_26, slice_scatter_28, 1, 0);  slice_scatter_26 = slice_scatter_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_145: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_5, [8, 12, 1024, 513])
        permute_129: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
        slice_161: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_129, 1, 0, 256)
        view_144: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_5, [8, 12, 1024, 513])
        permute_128: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        slice_159: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_128, 1, 0, 256);  permute_128 = None
        slice_160: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_159, 3, 0, 257);  slice_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_4: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_26: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, -2);  iota_4 = None
        iota_5: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_27: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
        sub_9: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_26, unsqueeze_27);  unsqueeze_26 = unsqueeze_27 = None
        le_2: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_9, 0);  sub_9 = None
        full_10: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_4: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_8: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_2, full_10, scalar_tensor_4);  le_2 = full_10 = scalar_tensor_4 = None
        rev_4: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_8, [0]);  where_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_28: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_4, 0);  rev_4 = None
        unsqueeze_29: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 2);  unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_4: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_29, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_39: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_4, torch.bool);  expand_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_11: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_126: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_11, [0, 2, 1, 3]);  full_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_142: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_5, [8, 12, 1024, 513]);  select_scatter_5 = None
        permute_125: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        slice_154: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_125, 1, 0, 256);  permute_125 = None
        slice_155: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_154, 3, 0, 257);  slice_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_9: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_39, permute_126, slice_155);  convert_element_type_39 = permute_126 = slice_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_16: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_160, where_9);  slice_160 = where_9 = None
        slice_scatter_29: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_161, copy_16, 3, 0, 257);  slice_161 = copy_16 = None
        slice_scatter_30: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_129, slice_scatter_29, 1, 0, 256);  permute_129 = slice_scatter_29 = None
        permute_130: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_30, [0, 2, 1, 3]);  slice_scatter_30 = None
        view_146: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_130, [96, 4, 256, 513]);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_153: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_146, [8, 12, 1024, 513])
        permute_138: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
        slice_174: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_138, 1, -256, 9223372036854775807)
        view_152: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_146, [8, 12, 1024, 513])
        permute_137: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        slice_172: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_137, 1, -256, 9223372036854775807);  permute_137 = None
        slice_173: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_172, 3, -257, 9223372036854775807);  slice_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_5: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_29, [1, 3]);  unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_5: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_5, [8, 256, 12, 257]);  rev_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_40: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_5, torch.bool);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_12: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_135: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_12, [0, 2, 1, 3]);  full_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_150: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_146, [8, 12, 1024, 513]);  view_146 = None
        permute_134: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        slice_167: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_134, 1, -256, 9223372036854775807);  permute_134 = None
        slice_168: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_167, 3, -257, 9223372036854775807);  slice_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_10: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_40, permute_135, slice_168);  convert_element_type_40 = permute_135 = slice_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_17: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_173, where_10);  slice_173 = where_10 = None
        slice_scatter_31: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_174, copy_17, 3, -257, 9223372036854775807);  slice_174 = copy_17 = None
        slice_scatter_32: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_138, slice_scatter_31, 1, -256, 9223372036854775807);  permute_138 = slice_scatter_31 = None
        permute_139: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_32, [0, 2, 1, 3]);  slice_scatter_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_164: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_139, [0, 2, 1, 3]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_14: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_181: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_14, 1, 0, -1)
        slice_179: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_14, 1, 0, -1)
        slice_180: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_179, 3, 256, 9223372036854775807);  slice_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_13: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_141: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_13, [0, 2, 1, 3]);  full_13 = None
        view_156: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_141, [8, 1024, 1]);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_158: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_156, [8, 2, 512, 1]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_15: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_158, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_32: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_15, 4);  as_strided_15 = None
        permute_143: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_32, [0, 1, 2, 4, 3]);  unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_1: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_30: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_1, 2);  ne_1 = None
        unsqueeze_31: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 3);  unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_41: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_31, torch.bfloat16)
        where_11: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_31, full_default_4, convert_element_type_41);  unsqueeze_31 = full_default_4 = convert_element_type_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_142: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_11, [0, 2, 1, 3]);  where_11 = None
        view_157: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_142, [8, 1024, 1]);  permute_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_159: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_157, [8, 2, 512, 1]);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_16: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_159, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_33: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_16, 4);  as_strided_16 = None
        permute_144: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_33, [0, 1, 4, 2, 3]);  unsqueeze_33 = None
        mul_8: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_143, permute_144);  permute_143 = permute_144 = None
        view_160: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_8, [8, 3, 512, 512]);  mul_8 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_5: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_160, [0, 0, 0, 1], 0.0);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_161: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_5, [8, 3, 512, 513]);  constant_pad_nd_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_177: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_161, 2, 0, 256)
        slice_178: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_177, 3, 0, 257);  slice_177 = None
        copy_18: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_180, slice_178);  slice_180 = slice_178 = None
        slice_scatter_33: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_181, copy_18, 3, 256, 9223372036854775807);  slice_181 = copy_18 = None
        slice_scatter_34: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_14, slice_scatter_33, 1, 0, -1);  full_14 = slice_scatter_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_43: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_34, 1, -1)
        select_42: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_34, 1, -1)
        slice_187: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_42, 2, 256, 9223372036854775807);  select_42 = None
        select_39: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_161, 1, -1)
        slice_184: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_39, 1, 256, 9223372036854775807);  select_39 = None
        slice_185: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_184, 2, 0, 257);  slice_184 = None
        copy_19: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_187, slice_185);  slice_187 = slice_185 = None
        slice_scatter_35: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_43, copy_19, 2, 256, 9223372036854775807);  select_43 = copy_19 = None
        select_scatter_6: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_34, slice_scatter_35, 1, -1);  slice_scatter_34 = slice_scatter_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_196: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_6, 1, 1, 9223372036854775807)
        slice_194: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_6, 1, 1, 9223372036854775807)
        slice_195: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_194, 3, 0, 256);  slice_194 = None
        slice_189: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_161, 2, -257, -1)
        slice_190: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_189, 3, 257, 9223372036854775807);  slice_189 = None
        copy_20: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_195, slice_190);  slice_195 = slice_190 = None
        slice_scatter_36: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_196, copy_20, 3, 0, 256);  slice_196 = copy_20 = None
        slice_scatter_37: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_6, slice_scatter_36, 1, 1, 9223372036854775807);  select_scatter_6 = slice_scatter_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_50: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_37, 1, 0)
        slice_206: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_50, 1, 1, 256)
        select_49: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_37, 1, 0)
        slice_204: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_49, 1, 1, 256);  select_49 = None
        slice_205: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_204, 2, 1, 256);  slice_204 = None
        select_45: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_161, 1, 0);  view_161 = None
        slice_199: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_45, 1, 0, 255);  select_45 = None
        slice_200: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_199, 2, -255, 9223372036854775807);  slice_199 = None
        copy_21: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_205, slice_200);  slice_205 = slice_200 = None
        slice_scatter_38: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_206, copy_21, 2, 1, 256);  slice_206 = copy_21 = None
        slice_scatter_39: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_50, slice_scatter_38, 1, 1, 256);  select_50 = slice_scatter_38 = None
        select_scatter_7: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_37, slice_scatter_39, 1, 0);  slice_scatter_37 = slice_scatter_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_169: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_7, [8, 1, 1024, 513])
        permute_152: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        slice_219: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_152, 1, 0, 256)
        view_168: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_7, [8, 1, 1024, 513])
        permute_151: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        slice_217: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_151, 1, 0, 256);  permute_151 = None
        slice_218: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_217, 3, 0, 257);  slice_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_6: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_34: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_6, -2);  iota_6 = None
        iota_7: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_35: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        sub_11: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_34, unsqueeze_35);  unsqueeze_34 = unsqueeze_35 = None
        le_3: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_11, 0);  sub_11 = None
        full_15: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_6: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_12: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_3, full_15, scalar_tensor_6);  le_3 = full_15 = scalar_tensor_6 = None
        rev_6: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_12, [0]);  where_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_36: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_6, 0);  rev_6 = None
        unsqueeze_37: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_6: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_37, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_42: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_6, torch.bool);  expand_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_5: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_166: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_7, [8, 1, 1024, 513]);  select_scatter_7 = None
        permute_148: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        slice_212: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_148, 1, 0, 256);  permute_148 = None
        slice_213: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_212, 3, 0, 257);  slice_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_13: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_42, full_default_5, slice_213);  convert_element_type_42 = full_default_5 = slice_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_22: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_218, where_13);  slice_218 = where_13 = None
        slice_scatter_40: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_219, copy_22, 3, 0, 257);  slice_219 = copy_22 = None
        slice_scatter_41: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_152, slice_scatter_40, 1, 0, 256);  permute_152 = slice_scatter_40 = None
        permute_153: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_41, [0, 2, 1, 3]);  slice_scatter_41 = None
        view_170: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_153, [8, 4, 256, 513]);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_177: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_170, [8, 1, 1024, 513])
        permute_161: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
        slice_232: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_161, 1, -256, 9223372036854775807)
        view_176: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_170, [8, 1, 1024, 513])
        permute_160: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
        slice_230: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_160, 1, -256, 9223372036854775807);  permute_160 = None
        slice_231: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_230, 3, -257, 9223372036854775807);  slice_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_7: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_37, [1, 3]);  unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_7: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_7, [8, 256, 1, 257]);  rev_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_43: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_7, torch.bool);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_6: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_174: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_170, [8, 1, 1024, 513]);  view_170 = None
        permute_157: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        slice_225: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_157, 1, -256, 9223372036854775807);  permute_157 = None
        slice_226: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_225, 3, -257, 9223372036854775807);  slice_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_14: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_43, full_default_6, slice_226);  convert_element_type_43 = full_default_6 = slice_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_23: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_231, where_14);  slice_231 = where_14 = None
        slice_scatter_42: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_232, copy_23, 3, -257, 9223372036854775807);  slice_232 = copy_23 = None
        slice_scatter_43: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_161, slice_scatter_42, 1, -256, 9223372036854775807);  permute_161 = slice_scatter_42 = None
        permute_162: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_43, [0, 2, 1, 3]);  slice_scatter_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_165: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_162, [0, 2, 1, 3]);  permute_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_20: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_164, permute_165);  permute_164 = permute_165 = None
        permute_166: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_20, [0, 2, 1, 3]);  add_20 = None
        permute_167: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_166, [0, 2, 1, 3]);  permute_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_44: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_167, torch.float32);  permute_167 = None
        clone_26: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_44, memory_format = torch.contiguous_format);  convert_element_type_44 = None
        amax_1: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_26, [-1], True)
        sub_12: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_26, amax_1);  clone_26 = amax_1 = None
        exp_1: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_2: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_17: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_15: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_39, full_default_7, div_17);  unsqueeze_39 = full_default_7 = div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_45: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.bfloat16);  where_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_169: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_45, [0, 2, 1, 3]);  convert_element_type_45 = None
        clone_28: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_186: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [96, 4, 256, 513]);  clone_28 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_7: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_186, [0, 257], 0.0);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_188: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_7, [96, 4, -1]);  constant_pad_nd_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_235: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_188, 2, 0, -256);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_189: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_235, [96, 4, 256, 769]);  slice_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_236: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_189, 3, 0, -1);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_40: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_236, 4);  slice_236 = None
        view_190: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_40, [384, 256, 768]);  unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_19: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        view_105: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [8192, 768]);  clone_19 = None
        permute_94: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_6: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_105, permute_94);  view_105 = permute_94 = None
        view_106: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [1024, 8, 768]);  mm_6 = None
        add_17: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_106, arg24_1);  view_106 = arg24_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_185: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_17, [1024, 8, 12, 64]);  add_17 = None
        permute_168: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_185, [1, 0, 2, 3]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_170: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_168, [0, 2, 1, 3]);  permute_168 = None
        view_187: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_170, [96, 1024, 64]);  permute_170 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_6: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_187, [0, 0, 256, 256], -1.0);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_17: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_6, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_41: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_17, 4);  as_strided_17 = None
        clone_29: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_41, memory_format = torch.contiguous_format);  unsqueeze_41 = None
        view_191: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [384, 768, 64]);  clone_29 = None
        bmm_3: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
        view_192: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [96, 4, 256, 1, 64]);  bmm_3 = None
        permute_175: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 1, 2, 4, 3]);  view_192 = None
        view_193: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_175, [96, 4, 256, 64]);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_194: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_193, [8, 12, 1024, 64]);  view_193 = None
        permute_176: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_177: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_176, [1, 0, 2, 3]);  permute_176 = None
        clone_30: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_195: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [1024, 8, 768]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_178: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_195, [1, 0, 2]);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_31: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
        view_196: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [8192, 768]);  clone_31 = None
        permute_179: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_7: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_196, permute_179);  view_196 = permute_179 = None
        view_197: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [8, 1024, 768]);  mm_7 = None
        add_22: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_197, arg26_1);  view_197 = arg26_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_22, convert_element_type_30);  add_22 = convert_element_type_30 = None
        convert_element_type_50: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_14: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_5);  convert_element_type_50 = getitem_5 = None
        add_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_9: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_2);  sub_14 = rsqrt_2 = None
        mul_10: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, arg27_1);  mul_9 = arg27_1 = None
        add_25: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, arg28_1);  mul_10 = arg28_1 = None
        convert_element_type_51: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_198: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_51, [8192, 768])
        permute_180: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_2: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg30_1, view_198, permute_180);  arg30_1 = view_198 = permute_180 = None
        view_199: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [8, 1024, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_55: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        mul_11: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.5)
        mul_12: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.7071067811865476);  convert_element_type_55 = None
        erf_1: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_26: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, add_26);  mul_11 = add_26 = None
        convert_element_type_56: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_200: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_56, [8192, 3072]);  convert_element_type_56 = None
        permute_181: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_3: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg32_1, view_200, permute_181);  arg32_1 = view_200 = permute_181 = None
        view_201: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [8, 1024, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_201, convert_element_type_51);  view_201 = convert_element_type_51 = None
        convert_element_type_60: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_60, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_25: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_26: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_59: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_60: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 3);  unsqueeze_59 = None
        full_default_11: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_18: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_241: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_18, 1, 0, -1)
        slice_239: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_18, 1, 0, -1)
        slice_240: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_239, 3, 256, 9223372036854775807);  slice_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_15: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_7);  convert_element_type_60 = getitem_7 = None
        add_28: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_14: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_3);  sub_15 = rsqrt_3 = None
        mul_15: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, arg33_1);  mul_14 = arg33_1 = None
        add_29: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, arg34_1);  mul_15 = arg34_1 = None
        convert_element_type_61: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_182: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_34: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format)
        view_202: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [8192, 768]);  clone_34 = None
        permute_183: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        mm_8: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_202, permute_183);  view_202 = permute_183 = None
        view_203: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [1024, 8, 768]);  mm_8 = None
        add_30: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_203, arg36_1);  view_203 = arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_20: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_30, 8.0);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_231: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_20, [1024, 8, 12, 64]);  div_20 = None
        permute_207: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_231, [1, 0, 2, 3]);  view_231 = None
        permute_208: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_207, [0, 2, 1, 3]);  permute_207 = None
        view_232: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_208, [96, 1024, 64]);  permute_208 = None
        view_233: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_232, [96, 2, 512, 64]);  view_232 = None
        as_strided_23: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_233, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_233 = None
        unsqueeze_46: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_23, 4);  as_strided_23 = None
        clone_37: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_46, memory_format = torch.contiguous_format);  unsqueeze_46 = None
        view_234: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [288, 512, 64]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_35: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format)
        view_204: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [8192, 768]);  clone_35 = None
        permute_184: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_9: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_204, permute_184);  view_204 = permute_184 = None
        view_205: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [1024, 8, 768]);  mm_9 = None
        add_31: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_205, arg38_1);  view_205 = arg38_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_210: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_31, [1024, 8, 12, 64]);  add_31 = None
        permute_187: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_210, [1, 0, 2, 3]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_192: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_187, [0, 2, 1, 3]);  permute_187 = None
        view_214: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_192, [96, 1024, 64]);  permute_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_221: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_214, [96, 2, 512, 64]);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_19: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_221, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_44: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_19, 4);  as_strided_19 = None
        permute_202: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_44, [0, 1, 4, 2, 3]);  unsqueeze_44 = None
        permute_211: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_202, [0, 1, 4, 3, 2]);  permute_202 = None
        clone_38: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
        view_235: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [288, 64, 512]);  clone_38 = None
        bmm_4: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_234, view_235);  view_234 = view_235 = None
        view_236: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [96, 3, 512, 1, 512]);  bmm_4 = None
        permute_212: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_236, [0, 1, 2, 4, 3]);  view_236 = None
        view_237: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_212, [96, 3, 512, 512]);  permute_212 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_8: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_237, [0, 0, 0, 1], 0.0);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_238: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_8, [96, 3, 512, 513]);  constant_pad_nd_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_237: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_238, 2, 0, 256)
        slice_238: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_237, 3, 0, 257);  slice_237 = None
        copy_24: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_240, slice_238);  slice_240 = slice_238 = None
        slice_scatter_44: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_241, copy_24, 3, 256, 9223372036854775807);  slice_241 = copy_24 = None
        slice_scatter_45: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_18, slice_scatter_44, 1, 0, -1);  full_18 = slice_scatter_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_56: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_45, 1, -1)
        select_55: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_45, 1, -1)
        slice_247: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_55, 2, 256, 9223372036854775807);  select_55 = None
        select_52: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_238, 1, -1)
        slice_244: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_52, 1, 256, 9223372036854775807);  select_52 = None
        slice_245: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_244, 2, 0, 257);  slice_244 = None
        copy_25: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_247, slice_245);  slice_247 = slice_245 = None
        slice_scatter_46: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_56, copy_25, 2, 256, 9223372036854775807);  select_56 = copy_25 = None
        select_scatter_8: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_45, slice_scatter_46, 1, -1);  slice_scatter_45 = slice_scatter_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_256: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_8, 1, 1, 9223372036854775807)
        slice_254: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_8, 1, 1, 9223372036854775807)
        slice_255: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_254, 3, 0, 256);  slice_254 = None
        slice_249: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_238, 2, -257, -1)
        slice_250: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_249, 3, 257, 9223372036854775807);  slice_249 = None
        copy_26: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_255, slice_250);  slice_255 = slice_250 = None
        slice_scatter_47: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_256, copy_26, 3, 0, 256);  slice_256 = copy_26 = None
        slice_scatter_48: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_8, slice_scatter_47, 1, 1, 9223372036854775807);  select_scatter_8 = slice_scatter_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_63: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_48, 1, 0)
        slice_266: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_63, 1, 1, 256)
        select_62: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_48, 1, 0)
        slice_264: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_62, 1, 1, 256);  select_62 = None
        slice_265: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_264, 2, 1, 256);  slice_264 = None
        select_58: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_238, 1, 0);  view_238 = None
        slice_259: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_58, 1, 0, 255);  select_58 = None
        slice_260: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_259, 2, -255, 9223372036854775807);  slice_259 = None
        copy_27: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_265, slice_260);  slice_265 = slice_260 = None
        slice_scatter_49: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_266, copy_27, 2, 1, 256);  slice_266 = copy_27 = None
        slice_scatter_50: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_63, slice_scatter_49, 1, 1, 256);  select_63 = slice_scatter_49 = None
        select_scatter_9: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_48, slice_scatter_50, 1, 0);  slice_scatter_48 = slice_scatter_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_246: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_9, [8, 12, 1024, 513])
        permute_220: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        slice_279: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_220, 1, 0, 256)
        view_245: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_9, [8, 12, 1024, 513])
        permute_219: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
        slice_277: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_219, 1, 0, 256);  permute_219 = None
        slice_278: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_277, 3, 0, 257);  slice_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_8: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_47: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_8, -2);  iota_8 = None
        iota_9: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_48: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
        sub_17: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_47, unsqueeze_48);  unsqueeze_47 = unsqueeze_48 = None
        le_4: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_17, 0);  sub_17 = None
        full_19: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_8: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_16: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_4, full_19, scalar_tensor_8);  le_4 = full_19 = scalar_tensor_8 = None
        rev_8: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_16, [0]);  where_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_49: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_8, 0);  rev_8 = None
        unsqueeze_50: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 2);  unsqueeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_8: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_50, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_70: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_8, torch.bool);  expand_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_20: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_217: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_20, [0, 2, 1, 3]);  full_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_243: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_9, [8, 12, 1024, 513]);  select_scatter_9 = None
        permute_216: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
        slice_272: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_216, 1, 0, 256);  permute_216 = None
        slice_273: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_272, 3, 0, 257);  slice_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_17: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_70, permute_217, slice_273);  convert_element_type_70 = permute_217 = slice_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_28: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_278, where_17);  slice_278 = where_17 = None
        slice_scatter_51: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_279, copy_28, 3, 0, 257);  slice_279 = copy_28 = None
        slice_scatter_52: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_220, slice_scatter_51, 1, 0, 256);  permute_220 = slice_scatter_51 = None
        permute_221: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_52, [0, 2, 1, 3]);  slice_scatter_52 = None
        view_247: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_221, [96, 4, 256, 513]);  permute_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_254: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_247, [8, 12, 1024, 513])
        permute_229: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
        slice_292: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_229, 1, -256, 9223372036854775807)
        view_253: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_247, [8, 12, 1024, 513])
        permute_228: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
        slice_290: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_228, 1, -256, 9223372036854775807);  permute_228 = None
        slice_291: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_290, 3, -257, 9223372036854775807);  slice_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_9: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_50, [1, 3]);  unsqueeze_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_9: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_9, [8, 256, 12, 257]);  rev_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_71: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_9, torch.bool);  expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_21: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_226: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_21, [0, 2, 1, 3]);  full_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_251: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_247, [8, 12, 1024, 513]);  view_247 = None
        permute_225: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
        slice_285: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_225, 1, -256, 9223372036854775807);  permute_225 = None
        slice_286: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_285, 3, -257, 9223372036854775807);  slice_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_18: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_71, permute_226, slice_286);  convert_element_type_71 = permute_226 = slice_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_29: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_291, where_18);  slice_291 = where_18 = None
        slice_scatter_53: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_292, copy_29, 3, -257, 9223372036854775807);  slice_292 = copy_29 = None
        slice_scatter_54: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_229, slice_scatter_53, 1, -256, 9223372036854775807);  permute_229 = slice_scatter_53 = None
        permute_230: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_54, [0, 2, 1, 3]);  slice_scatter_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_255: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_230, [0, 2, 1, 3]);  permute_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_23: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_299: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1)
        slice_297: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1)
        slice_298: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_297, 3, 256, 9223372036854775807);  slice_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_22: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_232: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_22, [0, 2, 1, 3]);  full_22 = None
        view_257: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_232, [8, 1024, 1]);  permute_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_259: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [8, 2, 512, 1]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_24: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_259, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_53: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_24, 4);  as_strided_24 = None
        permute_234: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_53, [0, 1, 2, 4, 3]);  unsqueeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_8: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_2: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_51: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_2, 2);  ne_2 = None
        unsqueeze_52: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 3);  unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_72: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_52, torch.bfloat16)
        where_19: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_52, full_default_8, convert_element_type_72);  unsqueeze_52 = full_default_8 = convert_element_type_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_233: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_19, [0, 2, 1, 3]);  where_19 = None
        view_258: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_233, [8, 1024, 1]);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_260: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_258, [8, 2, 512, 1]);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_25: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_260, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_54: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_25, 4);  as_strided_25 = None
        permute_235: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_54, [0, 1, 4, 2, 3]);  unsqueeze_54 = None
        mul_16: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_234, permute_235);  permute_234 = permute_235 = None
        view_261: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_16, [8, 3, 512, 512]);  mul_16 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_9: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_261, [0, 0, 0, 1], 0.0);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_262: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_9, [8, 3, 512, 513]);  constant_pad_nd_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_295: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_262, 2, 0, 256)
        slice_296: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_295, 3, 0, 257);  slice_295 = None
        copy_30: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_298, slice_296);  slice_298 = slice_296 = None
        slice_scatter_55: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_299, copy_30, 3, 256, 9223372036854775807);  slice_299 = copy_30 = None
        slice_scatter_56: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_23, slice_scatter_55, 1, 0, -1);  full_23 = slice_scatter_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_69: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_56, 1, -1)
        select_68: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_56, 1, -1)
        slice_305: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_68, 2, 256, 9223372036854775807);  select_68 = None
        select_65: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_262, 1, -1)
        slice_302: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_65, 1, 256, 9223372036854775807);  select_65 = None
        slice_303: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_302, 2, 0, 257);  slice_302 = None
        copy_31: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_305, slice_303);  slice_305 = slice_303 = None
        slice_scatter_57: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_69, copy_31, 2, 256, 9223372036854775807);  select_69 = copy_31 = None
        select_scatter_10: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_56, slice_scatter_57, 1, -1);  slice_scatter_56 = slice_scatter_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_314: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_10, 1, 1, 9223372036854775807)
        slice_312: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_10, 1, 1, 9223372036854775807)
        slice_313: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_312, 3, 0, 256);  slice_312 = None
        slice_307: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_262, 2, -257, -1)
        slice_308: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_307, 3, 257, 9223372036854775807);  slice_307 = None
        copy_32: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_313, slice_308);  slice_313 = slice_308 = None
        slice_scatter_58: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_314, copy_32, 3, 0, 256);  slice_314 = copy_32 = None
        slice_scatter_59: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_10, slice_scatter_58, 1, 1, 9223372036854775807);  select_scatter_10 = slice_scatter_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_76: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_59, 1, 0)
        slice_324: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_76, 1, 1, 256)
        select_75: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_59, 1, 0)
        slice_322: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_75, 1, 1, 256);  select_75 = None
        slice_323: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_322, 2, 1, 256);  slice_322 = None
        select_71: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_262, 1, 0);  view_262 = None
        slice_317: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_71, 1, 0, 255);  select_71 = None
        slice_318: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_317, 2, -255, 9223372036854775807);  slice_317 = None
        copy_33: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_323, slice_318);  slice_323 = slice_318 = None
        slice_scatter_60: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_324, copy_33, 2, 1, 256);  slice_324 = copy_33 = None
        slice_scatter_61: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_76, slice_scatter_60, 1, 1, 256);  select_76 = slice_scatter_60 = None
        select_scatter_11: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_59, slice_scatter_61, 1, 0);  slice_scatter_59 = slice_scatter_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_270: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_11, [8, 1, 1024, 513])
        permute_243: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        slice_337: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_243, 1, 0, 256)
        view_269: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_11, [8, 1, 1024, 513])
        permute_242: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
        slice_335: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_242, 1, 0, 256);  permute_242 = None
        slice_336: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_335, 3, 0, 257);  slice_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_10: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_55: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_10, -2);  iota_10 = None
        iota_11: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_56: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
        sub_19: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_55, unsqueeze_56);  unsqueeze_55 = unsqueeze_56 = None
        le_5: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_19, 0);  sub_19 = None
        full_24: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_10: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_20: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_5, full_24, scalar_tensor_10);  le_5 = full_24 = scalar_tensor_10 = None
        rev_10: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_20, [0]);  where_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_57: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_10, 0);  rev_10 = None
        unsqueeze_58: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_10: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_58, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_73: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_10, torch.bool);  expand_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_9: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_267: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_11, [8, 1, 1024, 513]);  select_scatter_11 = None
        permute_239: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
        slice_330: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_239, 1, 0, 256);  permute_239 = None
        slice_331: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_330, 3, 0, 257);  slice_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_21: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_73, full_default_9, slice_331);  convert_element_type_73 = full_default_9 = slice_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_34: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_336, where_21);  slice_336 = where_21 = None
        slice_scatter_62: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_337, copy_34, 3, 0, 257);  slice_337 = copy_34 = None
        slice_scatter_63: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_243, slice_scatter_62, 1, 0, 256);  permute_243 = slice_scatter_62 = None
        permute_244: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_63, [0, 2, 1, 3]);  slice_scatter_63 = None
        view_271: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_244, [8, 4, 256, 513]);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_278: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_271, [8, 1, 1024, 513])
        permute_252: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
        slice_350: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_252, 1, -256, 9223372036854775807)
        view_277: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_271, [8, 1, 1024, 513])
        permute_251: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
        slice_348: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_251, 1, -256, 9223372036854775807);  permute_251 = None
        slice_349: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_348, 3, -257, 9223372036854775807);  slice_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_11: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_58, [1, 3]);  unsqueeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_11: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_11, [8, 256, 1, 257]);  rev_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_74: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_11, torch.bool);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_10: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_275: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_271, [8, 1, 1024, 513]);  view_271 = None
        permute_248: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None
        slice_343: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_248, 1, -256, 9223372036854775807);  permute_248 = None
        slice_344: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_343, 3, -257, 9223372036854775807);  slice_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_22: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_74, full_default_10, slice_344);  convert_element_type_74 = full_default_10 = slice_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_35: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_349, where_22);  slice_349 = where_22 = None
        slice_scatter_64: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_350, copy_35, 3, -257, 9223372036854775807);  slice_350 = copy_35 = None
        slice_scatter_65: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_252, slice_scatter_64, 1, -256, 9223372036854775807);  permute_252 = slice_scatter_64 = None
        permute_253: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_65, [0, 2, 1, 3]);  slice_scatter_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_256: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_253, [0, 2, 1, 3]);  permute_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_35: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_255, permute_256);  permute_255 = permute_256 = None
        permute_257: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_35, [0, 2, 1, 3]);  add_35 = None
        permute_258: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_257, [0, 2, 1, 3]);  permute_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_75: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_258, torch.float32);  permute_258 = None
        clone_43: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_75, memory_format = torch.contiguous_format);  convert_element_type_75 = None
        amax_2: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_43, [-1], True)
        sub_20: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_43, amax_2);  clone_43 = amax_2 = None
        exp_2: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_3: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_27: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_23: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_60, full_default_11, div_27);  unsqueeze_60 = full_default_11 = div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_76: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.bfloat16);  where_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_260: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_76, [0, 2, 1, 3]);  convert_element_type_76 = None
        clone_45: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_287: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [96, 4, 256, 513]);  clone_45 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_11: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_287, [0, 257], 0.0);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_289: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_11, [96, 4, -1]);  constant_pad_nd_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_353: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_289, 2, 0, -256);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_290: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_353, [96, 4, 256, 769]);  slice_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_354: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_290, 3, 0, -1);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_61: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_354, 4);  slice_354 = None
        view_291: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_61, [384, 256, 768]);  unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_36: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_206: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [8192, 768]);  clone_36 = None
        permute_185: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_10: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_206, permute_185);  view_206 = permute_185 = None
        view_207: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [1024, 8, 768]);  mm_10 = None
        add_32: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_207, arg40_1);  view_207 = arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_286: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_32, [1024, 8, 12, 64]);  add_32 = None
        permute_259: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_286, [1, 0, 2, 3]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_261: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_259, [0, 2, 1, 3]);  permute_259 = None
        view_288: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_261, [96, 1024, 64]);  permute_261 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_10: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_288, [0, 0, 256, 256], -1.0);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_26: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_10, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_62: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_26, 4);  as_strided_26 = None
        clone_46: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_62, memory_format = torch.contiguous_format);  unsqueeze_62 = None
        view_292: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [384, 768, 64]);  clone_46 = None
        bmm_5: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_291, view_292);  view_291 = view_292 = None
        view_293: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [96, 4, 256, 1, 64]);  bmm_5 = None
        permute_266: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_293, [0, 1, 2, 4, 3]);  view_293 = None
        view_294: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_266, [96, 4, 256, 64]);  permute_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_295: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_294, [8, 12, 1024, 64]);  view_294 = None
        permute_267: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_268: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_267, [1, 0, 2, 3]);  permute_267 = None
        clone_47: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
        view_296: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [1024, 8, 768]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_269: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_296, [1, 0, 2]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_48: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_297: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [8192, 768]);  clone_48 = None
        permute_270: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_11: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_297, permute_270);  view_297 = permute_270 = None
        view_298: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [8, 1024, 768]);  mm_11 = None
        add_37: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_298, arg42_1);  view_298 = arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_38: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_37, convert_element_type_61);  add_37 = convert_element_type_61 = None
        convert_element_type_81: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_81, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_22: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_81, getitem_9);  convert_element_type_81 = getitem_9 = None
        add_39: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_17: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_4);  sub_22 = rsqrt_4 = None
        mul_18: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, arg43_1);  mul_17 = arg43_1 = None
        add_40: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, arg44_1);  mul_18 = arg44_1 = None
        convert_element_type_82: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_299: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_82, [8192, 768])
        permute_271: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_4: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg46_1, view_299, permute_271);  arg46_1 = view_299 = permute_271 = None
        view_300: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [8, 1024, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_86: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        mul_19: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.5)
        mul_20: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.7071067811865476);  convert_element_type_86 = None
        erf_2: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_41: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, add_41);  mul_19 = add_41 = None
        convert_element_type_87: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_301: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_87, [8192, 3072]);  convert_element_type_87 = None
        permute_272: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_5: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg48_1, view_301, permute_272);  arg48_1 = view_301 = permute_272 = None
        view_302: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [8, 1024, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_42: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_302, convert_element_type_82);  view_302 = convert_element_type_82 = None
        convert_element_type_91: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_42, torch.float32);  add_42 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_91, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_34: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_35: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_80: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_81: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 3);  unsqueeze_80 = None
        full_default_15: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_27: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_359: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1)
        slice_357: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1)
        slice_358: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_357, 3, 256, 9223372036854775807);  slice_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_23: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_91, getitem_11);  convert_element_type_91 = getitem_11 = None
        add_43: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_22: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_5);  sub_23 = rsqrt_5 = None
        mul_23: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, arg49_1);  mul_22 = arg49_1 = None
        add_44: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, arg50_1);  mul_23 = arg50_1 = None
        convert_element_type_92: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_273: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_92, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_51: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format)
        view_303: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [8192, 768]);  clone_51 = None
        permute_274: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_12: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_303, permute_274);  view_303 = permute_274 = None
        view_304: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [1024, 8, 768]);  mm_12 = None
        add_45: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_304, arg52_1);  view_304 = arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_30: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_45, 8.0);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_332: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_30, [1024, 8, 12, 64]);  div_30 = None
        permute_298: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [1, 0, 2, 3]);  view_332 = None
        permute_299: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_298, [0, 2, 1, 3]);  permute_298 = None
        view_333: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_299, [96, 1024, 64]);  permute_299 = None
        view_334: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_333, [96, 2, 512, 64]);  view_333 = None
        as_strided_32: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_334, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_334 = None
        unsqueeze_67: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_32, 4);  as_strided_32 = None
        clone_54: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_67, memory_format = torch.contiguous_format);  unsqueeze_67 = None
        view_335: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [288, 512, 64]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_52: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format)
        view_305: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [8192, 768]);  clone_52 = None
        permute_275: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_13: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_305, permute_275);  view_305 = permute_275 = None
        view_306: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [1024, 8, 768]);  mm_13 = None
        add_46: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_306, arg54_1);  view_306 = arg54_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_311: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_46, [1024, 8, 12, 64]);  add_46 = None
        permute_278: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_311, [1, 0, 2, 3]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_283: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_278, [0, 2, 1, 3]);  permute_278 = None
        view_315: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_283, [96, 1024, 64]);  permute_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_322: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_315, [96, 2, 512, 64]);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_28: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_322, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_65: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_28, 4);  as_strided_28 = None
        permute_293: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_65, [0, 1, 4, 2, 3]);  unsqueeze_65 = None
        permute_302: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_293, [0, 1, 4, 3, 2]);  permute_293 = None
        clone_55: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_336: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_55, [288, 64, 512]);  clone_55 = None
        bmm_6: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_335, view_336);  view_335 = view_336 = None
        view_337: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [96, 3, 512, 1, 512]);  bmm_6 = None
        permute_303: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_337, [0, 1, 2, 4, 3]);  view_337 = None
        view_338: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_303, [96, 3, 512, 512]);  permute_303 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_12: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_338, [0, 0, 0, 1], 0.0);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_339: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_12, [96, 3, 512, 513]);  constant_pad_nd_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_355: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_339, 2, 0, 256)
        slice_356: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_355, 3, 0, 257);  slice_355 = None
        copy_36: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_358, slice_356);  slice_358 = slice_356 = None
        slice_scatter_66: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_359, copy_36, 3, 256, 9223372036854775807);  slice_359 = copy_36 = None
        slice_scatter_67: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_27, slice_scatter_66, 1, 0, -1);  full_27 = slice_scatter_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_82: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_67, 1, -1)
        select_81: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_67, 1, -1)
        slice_365: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_81, 2, 256, 9223372036854775807);  select_81 = None
        select_78: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_339, 1, -1)
        slice_362: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_78, 1, 256, 9223372036854775807);  select_78 = None
        slice_363: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_362, 2, 0, 257);  slice_362 = None
        copy_37: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_365, slice_363);  slice_365 = slice_363 = None
        slice_scatter_68: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_82, copy_37, 2, 256, 9223372036854775807);  select_82 = copy_37 = None
        select_scatter_12: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_67, slice_scatter_68, 1, -1);  slice_scatter_67 = slice_scatter_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_374: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_12, 1, 1, 9223372036854775807)
        slice_372: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_12, 1, 1, 9223372036854775807)
        slice_373: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_372, 3, 0, 256);  slice_372 = None
        slice_367: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_339, 2, -257, -1)
        slice_368: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_367, 3, 257, 9223372036854775807);  slice_367 = None
        copy_38: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_373, slice_368);  slice_373 = slice_368 = None
        slice_scatter_69: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_374, copy_38, 3, 0, 256);  slice_374 = copy_38 = None
        slice_scatter_70: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_12, slice_scatter_69, 1, 1, 9223372036854775807);  select_scatter_12 = slice_scatter_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_89: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_70, 1, 0)
        slice_384: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_89, 1, 1, 256)
        select_88: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_70, 1, 0)
        slice_382: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_88, 1, 1, 256);  select_88 = None
        slice_383: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_382, 2, 1, 256);  slice_382 = None
        select_84: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_339, 1, 0);  view_339 = None
        slice_377: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_84, 1, 0, 255);  select_84 = None
        slice_378: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_377, 2, -255, 9223372036854775807);  slice_377 = None
        copy_39: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_383, slice_378);  slice_383 = slice_378 = None
        slice_scatter_71: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_384, copy_39, 2, 1, 256);  slice_384 = copy_39 = None
        slice_scatter_72: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_89, slice_scatter_71, 1, 1, 256);  select_89 = slice_scatter_71 = None
        select_scatter_13: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_70, slice_scatter_72, 1, 0);  slice_scatter_70 = slice_scatter_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_347: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_13, [8, 12, 1024, 513])
        permute_311: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
        slice_397: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_311, 1, 0, 256)
        view_346: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_13, [8, 12, 1024, 513])
        permute_310: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        slice_395: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_310, 1, 0, 256);  permute_310 = None
        slice_396: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_395, 3, 0, 257);  slice_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_12: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_68: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_12, -2);  iota_12 = None
        iota_13: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_69: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
        sub_25: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_68, unsqueeze_69);  unsqueeze_68 = unsqueeze_69 = None
        le_6: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_25, 0);  sub_25 = None
        full_28: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_12: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_24: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_6, full_28, scalar_tensor_12);  le_6 = full_28 = scalar_tensor_12 = None
        rev_12: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_24, [0]);  where_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_70: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_12, 0);  rev_12 = None
        unsqueeze_71: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, 2);  unsqueeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_12: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_71, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_101: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_12, torch.bool);  expand_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_29: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_308: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_29, [0, 2, 1, 3]);  full_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_344: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_13, [8, 12, 1024, 513]);  select_scatter_13 = None
        permute_307: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        slice_390: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_307, 1, 0, 256);  permute_307 = None
        slice_391: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_390, 3, 0, 257);  slice_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_25: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_101, permute_308, slice_391);  convert_element_type_101 = permute_308 = slice_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_40: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_396, where_25);  slice_396 = where_25 = None
        slice_scatter_73: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_397, copy_40, 3, 0, 257);  slice_397 = copy_40 = None
        slice_scatter_74: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_311, slice_scatter_73, 1, 0, 256);  permute_311 = slice_scatter_73 = None
        permute_312: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_74, [0, 2, 1, 3]);  slice_scatter_74 = None
        view_348: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_312, [96, 4, 256, 513]);  permute_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_355: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_348, [8, 12, 1024, 513])
        permute_320: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_355, [0, 2, 1, 3]);  view_355 = None
        slice_410: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_320, 1, -256, 9223372036854775807)
        view_354: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_348, [8, 12, 1024, 513])
        permute_319: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        slice_408: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_319, 1, -256, 9223372036854775807);  permute_319 = None
        slice_409: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_408, 3, -257, 9223372036854775807);  slice_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_13: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_71, [1, 3]);  unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_13: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_13, [8, 256, 12, 257]);  rev_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_102: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_13, torch.bool);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_30: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_317: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_30, [0, 2, 1, 3]);  full_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_352: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_348, [8, 12, 1024, 513]);  view_348 = None
        permute_316: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        slice_403: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_316, 1, -256, 9223372036854775807);  permute_316 = None
        slice_404: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_403, 3, -257, 9223372036854775807);  slice_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_26: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_102, permute_317, slice_404);  convert_element_type_102 = permute_317 = slice_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_41: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_409, where_26);  slice_409 = where_26 = None
        slice_scatter_75: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_410, copy_41, 3, -257, 9223372036854775807);  slice_410 = copy_41 = None
        slice_scatter_76: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_320, slice_scatter_75, 1, -256, 9223372036854775807);  permute_320 = slice_scatter_75 = None
        permute_321: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_76, [0, 2, 1, 3]);  slice_scatter_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_346: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_321, [0, 2, 1, 3]);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_32: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_417: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_32, 1, 0, -1)
        slice_415: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_32, 1, 0, -1)
        slice_416: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_415, 3, 256, 9223372036854775807);  slice_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_31: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_323: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_31, [0, 2, 1, 3]);  full_31 = None
        view_358: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_323, [8, 1024, 1]);  permute_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_360: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_358, [8, 2, 512, 1]);  view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_33: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_360, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_74: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_33, 4);  as_strided_33 = None
        permute_325: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_74, [0, 1, 2, 4, 3]);  unsqueeze_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_12: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_3: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_72: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_3, 2);  ne_3 = None
        unsqueeze_73: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_103: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_73, torch.bfloat16)
        where_27: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_73, full_default_12, convert_element_type_103);  unsqueeze_73 = full_default_12 = convert_element_type_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_324: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_27, [0, 2, 1, 3]);  where_27 = None
        view_359: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_324, [8, 1024, 1]);  permute_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_361: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_359, [8, 2, 512, 1]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_34: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_361, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_75: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_34, 4);  as_strided_34 = None
        permute_326: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 2, 3]);  unsqueeze_75 = None
        mul_24: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_325, permute_326);  permute_325 = permute_326 = None
        view_362: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_24, [8, 3, 512, 512]);  mul_24 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_13: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_362, [0, 0, 0, 1], 0.0);  view_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_363: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_13, [8, 3, 512, 513]);  constant_pad_nd_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_413: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_363, 2, 0, 256)
        slice_414: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_413, 3, 0, 257);  slice_413 = None
        copy_42: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_416, slice_414);  slice_416 = slice_414 = None
        slice_scatter_77: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_417, copy_42, 3, 256, 9223372036854775807);  slice_417 = copy_42 = None
        slice_scatter_78: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_32, slice_scatter_77, 1, 0, -1);  full_32 = slice_scatter_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_95: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_78, 1, -1)
        select_94: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_78, 1, -1)
        slice_423: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_94, 2, 256, 9223372036854775807);  select_94 = None
        select_91: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_363, 1, -1)
        slice_420: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_91, 1, 256, 9223372036854775807);  select_91 = None
        slice_421: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_420, 2, 0, 257);  slice_420 = None
        copy_43: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_423, slice_421);  slice_423 = slice_421 = None
        slice_scatter_79: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_95, copy_43, 2, 256, 9223372036854775807);  select_95 = copy_43 = None
        select_scatter_14: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_78, slice_scatter_79, 1, -1);  slice_scatter_78 = slice_scatter_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_432: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_14, 1, 1, 9223372036854775807)
        slice_430: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_14, 1, 1, 9223372036854775807)
        slice_431: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_430, 3, 0, 256);  slice_430 = None
        slice_425: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_363, 2, -257, -1)
        slice_426: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_425, 3, 257, 9223372036854775807);  slice_425 = None
        copy_44: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_431, slice_426);  slice_431 = slice_426 = None
        slice_scatter_80: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_432, copy_44, 3, 0, 256);  slice_432 = copy_44 = None
        slice_scatter_81: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_14, slice_scatter_80, 1, 1, 9223372036854775807);  select_scatter_14 = slice_scatter_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_102: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_81, 1, 0)
        slice_442: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_102, 1, 1, 256)
        select_101: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_81, 1, 0)
        slice_440: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_101, 1, 1, 256);  select_101 = None
        slice_441: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_440, 2, 1, 256);  slice_440 = None
        select_97: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_363, 1, 0);  view_363 = None
        slice_435: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_97, 1, 0, 255);  select_97 = None
        slice_436: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_435, 2, -255, 9223372036854775807);  slice_435 = None
        copy_45: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_441, slice_436);  slice_441 = slice_436 = None
        slice_scatter_82: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_442, copy_45, 2, 1, 256);  slice_442 = copy_45 = None
        slice_scatter_83: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_102, slice_scatter_82, 1, 1, 256);  select_102 = slice_scatter_82 = None
        select_scatter_15: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_81, slice_scatter_83, 1, 0);  slice_scatter_81 = slice_scatter_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_371: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_15, [8, 1, 1024, 513])
        permute_334: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
        slice_455: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_334, 1, 0, 256)
        view_370: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_15, [8, 1, 1024, 513])
        permute_333: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
        slice_453: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_333, 1, 0, 256);  permute_333 = None
        slice_454: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_453, 3, 0, 257);  slice_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_14: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_76: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_14, -2);  iota_14 = None
        iota_15: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_77: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
        sub_27: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_76, unsqueeze_77);  unsqueeze_76 = unsqueeze_77 = None
        le_7: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_27, 0);  sub_27 = None
        full_33: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_14: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_28: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_7, full_33, scalar_tensor_14);  le_7 = full_33 = scalar_tensor_14 = None
        rev_14: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_28, [0]);  where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_78: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_14, 0);  rev_14 = None
        unsqueeze_79: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 2);  unsqueeze_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_14: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_79, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_104: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_14, torch.bool);  expand_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_13: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_368: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_15, [8, 1, 1024, 513]);  select_scatter_15 = None
        permute_330: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
        slice_448: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_330, 1, 0, 256);  permute_330 = None
        slice_449: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_448, 3, 0, 257);  slice_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_29: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_104, full_default_13, slice_449);  convert_element_type_104 = full_default_13 = slice_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_46: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_454, where_29);  slice_454 = where_29 = None
        slice_scatter_84: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_455, copy_46, 3, 0, 257);  slice_455 = copy_46 = None
        slice_scatter_85: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_334, slice_scatter_84, 1, 0, 256);  permute_334 = slice_scatter_84 = None
        permute_335: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_85, [0, 2, 1, 3]);  slice_scatter_85 = None
        view_372: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_335, [8, 4, 256, 513]);  permute_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_379: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_372, [8, 1, 1024, 513])
        permute_343: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        slice_468: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_343, 1, -256, 9223372036854775807)
        view_378: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_372, [8, 1, 1024, 513])
        permute_342: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        slice_466: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_342, 1, -256, 9223372036854775807);  permute_342 = None
        slice_467: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_466, 3, -257, 9223372036854775807);  slice_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_15: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_79, [1, 3]);  unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_15: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_15, [8, 256, 1, 257]);  rev_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_105: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_15, torch.bool);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_14: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_376: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_372, [8, 1, 1024, 513]);  view_372 = None
        permute_339: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        slice_461: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_339, 1, -256, 9223372036854775807);  permute_339 = None
        slice_462: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_461, 3, -257, 9223372036854775807);  slice_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_30: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_105, full_default_14, slice_462);  convert_element_type_105 = full_default_14 = slice_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_47: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_467, where_30);  slice_467 = where_30 = None
        slice_scatter_86: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_468, copy_47, 3, -257, 9223372036854775807);  slice_468 = copy_47 = None
        slice_scatter_87: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_343, slice_scatter_86, 1, -256, 9223372036854775807);  permute_343 = slice_scatter_86 = None
        permute_344: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_87, [0, 2, 1, 3]);  slice_scatter_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_347: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_344, [0, 2, 1, 3]);  permute_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_50: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_346, permute_347);  permute_346 = permute_347 = None
        permute_348: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_50, [0, 2, 1, 3]);  add_50 = None
        permute_349: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_348, [0, 2, 1, 3]);  permute_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_106: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_349, torch.float32);  permute_349 = None
        clone_60: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_106, memory_format = torch.contiguous_format);  convert_element_type_106 = None
        amax_3: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_60, [-1], True)
        sub_28: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_60, amax_3);  clone_60 = amax_3 = None
        exp_3: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_4: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_37: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_31: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_81, full_default_15, div_37);  unsqueeze_81 = full_default_15 = div_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_107: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_31, torch.bfloat16);  where_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_351: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_107, [0, 2, 1, 3]);  convert_element_type_107 = None
        clone_62: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_351, memory_format = torch.contiguous_format);  permute_351 = None
        view_388: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_62, [96, 4, 256, 513]);  clone_62 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_15: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_388, [0, 257], 0.0);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_390: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_15, [96, 4, -1]);  constant_pad_nd_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_471: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_390, 2, 0, -256);  view_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_391: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_471, [96, 4, 256, 769]);  slice_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_472: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_391, 3, 0, -1);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_82: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_472, 4);  slice_472 = None
        view_392: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_82, [384, 256, 768]);  unsqueeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_53: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format);  permute_273 = None
        view_307: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [8192, 768]);  clone_53 = None
        permute_276: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_14: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_307, permute_276);  view_307 = permute_276 = None
        view_308: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [1024, 8, 768]);  mm_14 = None
        add_47: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_308, arg56_1);  view_308 = arg56_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_387: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_47, [1024, 8, 12, 64]);  add_47 = None
        permute_350: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_387, [1, 0, 2, 3]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_352: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_350, [0, 2, 1, 3]);  permute_350 = None
        view_389: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_352, [96, 1024, 64]);  permute_352 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_14: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_389, [0, 0, 256, 256], -1.0);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_35: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_14, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_83: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_35, 4);  as_strided_35 = None
        clone_63: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_83, memory_format = torch.contiguous_format);  unsqueeze_83 = None
        view_393: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [384, 768, 64]);  clone_63 = None
        bmm_7: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_392, view_393);  view_392 = view_393 = None
        view_394: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [96, 4, 256, 1, 64]);  bmm_7 = None
        permute_357: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_394, [0, 1, 2, 4, 3]);  view_394 = None
        view_395: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_357, [96, 4, 256, 64]);  permute_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_396: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_395, [8, 12, 1024, 64]);  view_395 = None
        permute_358: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_359: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_358, [1, 0, 2, 3]);  permute_358 = None
        clone_64: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_397: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [1024, 8, 768]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_360: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_397, [1, 0, 2]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_65: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
        view_398: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [8192, 768]);  clone_65 = None
        permute_361: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        mm_15: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_398, permute_361);  view_398 = permute_361 = None
        view_399: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [8, 1024, 768]);  mm_15 = None
        add_52: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_399, arg58_1);  view_399 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_53: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_52, convert_element_type_92);  add_52 = convert_element_type_92 = None
        convert_element_type_112: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.float32);  add_53 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_112, [2], correction = 0, keepdim = True)
        getitem_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_30: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_112, getitem_13);  convert_element_type_112 = getitem_13 = None
        add_54: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_25: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_6);  sub_30 = rsqrt_6 = None
        mul_26: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, arg59_1);  mul_25 = arg59_1 = None
        add_55: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, arg60_1);  mul_26 = arg60_1 = None
        convert_element_type_113: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_400: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_113, [8192, 768])
        permute_362: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_6: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg62_1, view_400, permute_362);  arg62_1 = view_400 = permute_362 = None
        view_401: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [8, 1024, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_117: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_401, torch.float32);  view_401 = None
        mul_27: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_117, 0.5)
        mul_28: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_117, 0.7071067811865476);  convert_element_type_117 = None
        erf_3: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_56: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_29: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, add_56);  mul_27 = add_56 = None
        convert_element_type_118: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_402: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_118, [8192, 3072]);  convert_element_type_118 = None
        permute_363: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_7: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg64_1, view_402, permute_363);  arg64_1 = view_402 = permute_363 = None
        view_403: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [8, 1024, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_403, convert_element_type_113);  view_403 = convert_element_type_113 = None
        convert_element_type_122: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_122, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_43: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_44: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_101: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_102: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 3);  unsqueeze_101 = None
        full_default_19: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_36: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_477: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_36, 1, 0, -1)
        slice_475: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_36, 1, 0, -1)
        slice_476: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_475, 3, 256, 9223372036854775807);  slice_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_31: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_122, getitem_15);  convert_element_type_122 = getitem_15 = None
        add_58: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_30: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_7);  sub_31 = rsqrt_7 = None
        mul_31: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, arg65_1);  mul_30 = arg65_1 = None
        add_59: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, arg66_1);  mul_31 = arg66_1 = None
        convert_element_type_123: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_364: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_123, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_68: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_364, memory_format = torch.contiguous_format)
        view_404: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [8192, 768]);  clone_68 = None
        permute_365: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_16: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_404, permute_365);  view_404 = permute_365 = None
        view_405: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [1024, 8, 768]);  mm_16 = None
        add_60: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_405, arg68_1);  view_405 = arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_40: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_60, 8.0);  add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_433: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_40, [1024, 8, 12, 64]);  div_40 = None
        permute_389: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_433, [1, 0, 2, 3]);  view_433 = None
        permute_390: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_389, [0, 2, 1, 3]);  permute_389 = None
        view_434: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_390, [96, 1024, 64]);  permute_390 = None
        view_435: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_434, [96, 2, 512, 64]);  view_434 = None
        as_strided_41: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_435, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_435 = None
        unsqueeze_88: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_41, 4);  as_strided_41 = None
        clone_71: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_88, memory_format = torch.contiguous_format);  unsqueeze_88 = None
        view_436: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [288, 512, 64]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_69: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_364, memory_format = torch.contiguous_format)
        view_406: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [8192, 768]);  clone_69 = None
        permute_366: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_17: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_406, permute_366);  view_406 = permute_366 = None
        view_407: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [1024, 8, 768]);  mm_17 = None
        add_61: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_407, arg70_1);  view_407 = arg70_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_412: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_61, [1024, 8, 12, 64]);  add_61 = None
        permute_369: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_412, [1, 0, 2, 3]);  view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_374: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_369, [0, 2, 1, 3]);  permute_369 = None
        view_416: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_374, [96, 1024, 64]);  permute_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_423: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_416, [96, 2, 512, 64]);  view_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_37: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_423, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_86: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_37, 4);  as_strided_37 = None
        permute_384: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_86, [0, 1, 4, 2, 3]);  unsqueeze_86 = None
        permute_393: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_384, [0, 1, 4, 3, 2]);  permute_384 = None
        clone_72: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
        view_437: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [288, 64, 512]);  clone_72 = None
        bmm_8: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_436, view_437);  view_436 = view_437 = None
        view_438: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [96, 3, 512, 1, 512]);  bmm_8 = None
        permute_394: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_438, [0, 1, 2, 4, 3]);  view_438 = None
        view_439: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_394, [96, 3, 512, 512]);  permute_394 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_16: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_439, [0, 0, 0, 1], 0.0);  view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_440: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_16, [96, 3, 512, 513]);  constant_pad_nd_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_473: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_440, 2, 0, 256)
        slice_474: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_473, 3, 0, 257);  slice_473 = None
        copy_48: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_476, slice_474);  slice_476 = slice_474 = None
        slice_scatter_88: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_477, copy_48, 3, 256, 9223372036854775807);  slice_477 = copy_48 = None
        slice_scatter_89: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_36, slice_scatter_88, 1, 0, -1);  full_36 = slice_scatter_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_108: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_89, 1, -1)
        select_107: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_89, 1, -1)
        slice_483: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_107, 2, 256, 9223372036854775807);  select_107 = None
        select_104: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_440, 1, -1)
        slice_480: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_104, 1, 256, 9223372036854775807);  select_104 = None
        slice_481: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_480, 2, 0, 257);  slice_480 = None
        copy_49: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_483, slice_481);  slice_483 = slice_481 = None
        slice_scatter_90: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_108, copy_49, 2, 256, 9223372036854775807);  select_108 = copy_49 = None
        select_scatter_16: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_89, slice_scatter_90, 1, -1);  slice_scatter_89 = slice_scatter_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_492: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_16, 1, 1, 9223372036854775807)
        slice_490: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_16, 1, 1, 9223372036854775807)
        slice_491: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_490, 3, 0, 256);  slice_490 = None
        slice_485: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_440, 2, -257, -1)
        slice_486: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_485, 3, 257, 9223372036854775807);  slice_485 = None
        copy_50: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_491, slice_486);  slice_491 = slice_486 = None
        slice_scatter_91: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_492, copy_50, 3, 0, 256);  slice_492 = copy_50 = None
        slice_scatter_92: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_16, slice_scatter_91, 1, 1, 9223372036854775807);  select_scatter_16 = slice_scatter_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_115: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_92, 1, 0)
        slice_502: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_115, 1, 1, 256)
        select_114: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_92, 1, 0)
        slice_500: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_114, 1, 1, 256);  select_114 = None
        slice_501: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_500, 2, 1, 256);  slice_500 = None
        select_110: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_440, 1, 0);  view_440 = None
        slice_495: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_110, 1, 0, 255);  select_110 = None
        slice_496: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_495, 2, -255, 9223372036854775807);  slice_495 = None
        copy_51: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_501, slice_496);  slice_501 = slice_496 = None
        slice_scatter_93: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_502, copy_51, 2, 1, 256);  slice_502 = copy_51 = None
        slice_scatter_94: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_115, slice_scatter_93, 1, 1, 256);  select_115 = slice_scatter_93 = None
        select_scatter_17: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_92, slice_scatter_94, 1, 0);  slice_scatter_92 = slice_scatter_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_448: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_17, [8, 12, 1024, 513])
        permute_402: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        slice_515: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_402, 1, 0, 256)
        view_447: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_17, [8, 12, 1024, 513])
        permute_401: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
        slice_513: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_401, 1, 0, 256);  permute_401 = None
        slice_514: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_513, 3, 0, 257);  slice_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_16: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_89: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_16, -2);  iota_16 = None
        iota_17: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_90: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
        sub_33: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_89, unsqueeze_90);  unsqueeze_89 = unsqueeze_90 = None
        le_8: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_33, 0);  sub_33 = None
        full_37: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_16: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_32: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_8, full_37, scalar_tensor_16);  le_8 = full_37 = scalar_tensor_16 = None
        rev_16: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_32, [0]);  where_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_91: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_16, 0);  rev_16 = None
        unsqueeze_92: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 2);  unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_16: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_92, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_132: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_16, torch.bool);  expand_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_38: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_399: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_38, [0, 2, 1, 3]);  full_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_445: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_17, [8, 12, 1024, 513]);  select_scatter_17 = None
        permute_398: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
        slice_508: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_398, 1, 0, 256);  permute_398 = None
        slice_509: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_508, 3, 0, 257);  slice_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_33: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_132, permute_399, slice_509);  convert_element_type_132 = permute_399 = slice_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_52: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_514, where_33);  slice_514 = where_33 = None
        slice_scatter_95: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_515, copy_52, 3, 0, 257);  slice_515 = copy_52 = None
        slice_scatter_96: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_402, slice_scatter_95, 1, 0, 256);  permute_402 = slice_scatter_95 = None
        permute_403: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_96, [0, 2, 1, 3]);  slice_scatter_96 = None
        view_449: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_403, [96, 4, 256, 513]);  permute_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_456: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_449, [8, 12, 1024, 513])
        permute_411: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        slice_528: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_411, 1, -256, 9223372036854775807)
        view_455: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_449, [8, 12, 1024, 513])
        permute_410: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
        slice_526: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_410, 1, -256, 9223372036854775807);  permute_410 = None
        slice_527: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_526, 3, -257, 9223372036854775807);  slice_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_17: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_92, [1, 3]);  unsqueeze_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_17: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_17, [8, 256, 12, 257]);  rev_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_133: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_17, torch.bool);  expand_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_39: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_408: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_39, [0, 2, 1, 3]);  full_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_453: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_449, [8, 12, 1024, 513]);  view_449 = None
        permute_407: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        slice_521: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_407, 1, -256, 9223372036854775807);  permute_407 = None
        slice_522: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_521, 3, -257, 9223372036854775807);  slice_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_34: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_133, permute_408, slice_522);  convert_element_type_133 = permute_408 = slice_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_53: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_527, where_34);  slice_527 = where_34 = None
        slice_scatter_97: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_528, copy_53, 3, -257, 9223372036854775807);  slice_528 = copy_53 = None
        slice_scatter_98: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_411, slice_scatter_97, 1, -256, 9223372036854775807);  permute_411 = slice_scatter_97 = None
        permute_412: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_98, [0, 2, 1, 3]);  slice_scatter_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_437: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_412, [0, 2, 1, 3]);  permute_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_41: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_535: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_41, 1, 0, -1)
        slice_533: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_41, 1, 0, -1)
        slice_534: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_533, 3, 256, 9223372036854775807);  slice_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_40: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_414: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_40, [0, 2, 1, 3]);  full_40 = None
        view_459: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_414, [8, 1024, 1]);  permute_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_461: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_459, [8, 2, 512, 1]);  view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_42: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_461, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_95: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_42, 4);  as_strided_42 = None
        permute_416: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_95, [0, 1, 2, 4, 3]);  unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_16: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_4: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_93: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_4, 2);  ne_4 = None
        unsqueeze_94: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_134: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_94, torch.bfloat16)
        where_35: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_94, full_default_16, convert_element_type_134);  unsqueeze_94 = full_default_16 = convert_element_type_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_415: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_35, [0, 2, 1, 3]);  where_35 = None
        view_460: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_415, [8, 1024, 1]);  permute_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_462: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_460, [8, 2, 512, 1]);  view_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_43: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_462, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_96: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_43, 4);  as_strided_43 = None
        permute_417: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_96, [0, 1, 4, 2, 3]);  unsqueeze_96 = None
        mul_32: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_416, permute_417);  permute_416 = permute_417 = None
        view_463: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_32, [8, 3, 512, 512]);  mul_32 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_17: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_463, [0, 0, 0, 1], 0.0);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_464: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_17, [8, 3, 512, 513]);  constant_pad_nd_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_531: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_464, 2, 0, 256)
        slice_532: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_531, 3, 0, 257);  slice_531 = None
        copy_54: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_534, slice_532);  slice_534 = slice_532 = None
        slice_scatter_99: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_535, copy_54, 3, 256, 9223372036854775807);  slice_535 = copy_54 = None
        slice_scatter_100: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_41, slice_scatter_99, 1, 0, -1);  full_41 = slice_scatter_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_121: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_100, 1, -1)
        select_120: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_100, 1, -1)
        slice_541: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_120, 2, 256, 9223372036854775807);  select_120 = None
        select_117: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_464, 1, -1)
        slice_538: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_117, 1, 256, 9223372036854775807);  select_117 = None
        slice_539: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_538, 2, 0, 257);  slice_538 = None
        copy_55: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_541, slice_539);  slice_541 = slice_539 = None
        slice_scatter_101: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_121, copy_55, 2, 256, 9223372036854775807);  select_121 = copy_55 = None
        select_scatter_18: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_100, slice_scatter_101, 1, -1);  slice_scatter_100 = slice_scatter_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_550: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_18, 1, 1, 9223372036854775807)
        slice_548: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_18, 1, 1, 9223372036854775807)
        slice_549: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_548, 3, 0, 256);  slice_548 = None
        slice_543: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_464, 2, -257, -1)
        slice_544: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_543, 3, 257, 9223372036854775807);  slice_543 = None
        copy_56: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_549, slice_544);  slice_549 = slice_544 = None
        slice_scatter_102: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_550, copy_56, 3, 0, 256);  slice_550 = copy_56 = None
        slice_scatter_103: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_18, slice_scatter_102, 1, 1, 9223372036854775807);  select_scatter_18 = slice_scatter_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_128: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_103, 1, 0)
        slice_560: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_128, 1, 1, 256)
        select_127: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_103, 1, 0)
        slice_558: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_127, 1, 1, 256);  select_127 = None
        slice_559: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_558, 2, 1, 256);  slice_558 = None
        select_123: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_464, 1, 0);  view_464 = None
        slice_553: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_123, 1, 0, 255);  select_123 = None
        slice_554: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_553, 2, -255, 9223372036854775807);  slice_553 = None
        copy_57: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_559, slice_554);  slice_559 = slice_554 = None
        slice_scatter_104: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_560, copy_57, 2, 1, 256);  slice_560 = copy_57 = None
        slice_scatter_105: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_128, slice_scatter_104, 1, 1, 256);  select_128 = slice_scatter_104 = None
        select_scatter_19: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_103, slice_scatter_105, 1, 0);  slice_scatter_103 = slice_scatter_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_472: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_19, [8, 1, 1024, 513])
        permute_425: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        slice_573: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_425, 1, 0, 256)
        view_471: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_19, [8, 1, 1024, 513])
        permute_424: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
        slice_571: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_424, 1, 0, 256);  permute_424 = None
        slice_572: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_571, 3, 0, 257);  slice_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_18: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_97: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_18, -2);  iota_18 = None
        iota_19: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_98: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
        sub_35: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_97, unsqueeze_98);  unsqueeze_97 = unsqueeze_98 = None
        le_9: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_35, 0);  sub_35 = None
        full_42: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_18: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_36: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_9, full_42, scalar_tensor_18);  le_9 = full_42 = scalar_tensor_18 = None
        rev_18: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_36, [0]);  where_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_99: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_18, 0);  rev_18 = None
        unsqueeze_100: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_18: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_100, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_135: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_18, torch.bool);  expand_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_17: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_469: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_19, [8, 1, 1024, 513]);  select_scatter_19 = None
        permute_421: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
        slice_566: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_421, 1, 0, 256);  permute_421 = None
        slice_567: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_566, 3, 0, 257);  slice_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_37: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_135, full_default_17, slice_567);  convert_element_type_135 = full_default_17 = slice_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_58: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_572, where_37);  slice_572 = where_37 = None
        slice_scatter_106: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_573, copy_58, 3, 0, 257);  slice_573 = copy_58 = None
        slice_scatter_107: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_425, slice_scatter_106, 1, 0, 256);  permute_425 = slice_scatter_106 = None
        permute_426: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_107, [0, 2, 1, 3]);  slice_scatter_107 = None
        view_473: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_426, [8, 4, 256, 513]);  permute_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_480: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_473, [8, 1, 1024, 513])
        permute_434: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None
        slice_586: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_434, 1, -256, 9223372036854775807)
        view_479: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_473, [8, 1, 1024, 513])
        permute_433: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
        slice_584: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_433, 1, -256, 9223372036854775807);  permute_433 = None
        slice_585: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_584, 3, -257, 9223372036854775807);  slice_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_19: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_100, [1, 3]);  unsqueeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_19: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_19, [8, 256, 1, 257]);  rev_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_136: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_19, torch.bool);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_18: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_477: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_473, [8, 1, 1024, 513]);  view_473 = None
        permute_430: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
        slice_579: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_430, 1, -256, 9223372036854775807);  permute_430 = None
        slice_580: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_579, 3, -257, 9223372036854775807);  slice_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_38: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_136, full_default_18, slice_580);  convert_element_type_136 = full_default_18 = slice_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_59: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_585, where_38);  slice_585 = where_38 = None
        slice_scatter_108: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_586, copy_59, 3, -257, 9223372036854775807);  slice_586 = copy_59 = None
        slice_scatter_109: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_434, slice_scatter_108, 1, -256, 9223372036854775807);  permute_434 = slice_scatter_108 = None
        permute_435: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_109, [0, 2, 1, 3]);  slice_scatter_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_438: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_435, [0, 2, 1, 3]);  permute_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_65: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_437, permute_438);  permute_437 = permute_438 = None
        permute_439: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_65, [0, 2, 1, 3]);  add_65 = None
        permute_440: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_439, [0, 2, 1, 3]);  permute_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_137: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_440, torch.float32);  permute_440 = None
        clone_77: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_137, memory_format = torch.contiguous_format);  convert_element_type_137 = None
        amax_4: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_77, [-1], True)
        sub_36: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_77, amax_4);  clone_77 = amax_4 = None
        exp_4: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_5: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_47: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_39: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_102, full_default_19, div_47);  unsqueeze_102 = full_default_19 = div_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_138: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_39, torch.bfloat16);  where_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_442: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_138, [0, 2, 1, 3]);  convert_element_type_138 = None
        clone_79: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_442, memory_format = torch.contiguous_format);  permute_442 = None
        view_489: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [96, 4, 256, 513]);  clone_79 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_19: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_489, [0, 257], 0.0);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_491: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_19, [96, 4, -1]);  constant_pad_nd_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_589: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_491, 2, 0, -256);  view_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_492: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_589, [96, 4, 256, 769]);  slice_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_590: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_492, 3, 0, -1);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_103: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_590, 4);  slice_590 = None
        view_493: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_103, [384, 256, 768]);  unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_70: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_364, memory_format = torch.contiguous_format);  permute_364 = None
        view_408: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [8192, 768]);  clone_70 = None
        permute_367: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_18: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_408, permute_367);  view_408 = permute_367 = None
        view_409: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [1024, 8, 768]);  mm_18 = None
        add_62: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_409, arg72_1);  view_409 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_488: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_62, [1024, 8, 12, 64]);  add_62 = None
        permute_441: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_488, [1, 0, 2, 3]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_443: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_441, [0, 2, 1, 3]);  permute_441 = None
        view_490: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_443, [96, 1024, 64]);  permute_443 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_18: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_490, [0, 0, 256, 256], -1.0);  view_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_44: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_18, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_104: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_44, 4);  as_strided_44 = None
        clone_80: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_104, memory_format = torch.contiguous_format);  unsqueeze_104 = None
        view_494: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [384, 768, 64]);  clone_80 = None
        bmm_9: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_493, view_494);  view_493 = view_494 = None
        view_495: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [96, 4, 256, 1, 64]);  bmm_9 = None
        permute_448: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_495, [0, 1, 2, 4, 3]);  view_495 = None
        view_496: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_448, [96, 4, 256, 64]);  permute_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_497: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_496, [8, 12, 1024, 64]);  view_496 = None
        permute_449: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_450: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_449, [1, 0, 2, 3]);  permute_449 = None
        clone_81: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None
        view_498: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [1024, 8, 768]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_451: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_498, [1, 0, 2]);  view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_82: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        view_499: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [8192, 768]);  clone_82 = None
        permute_452: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_19: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_499, permute_452);  view_499 = permute_452 = None
        view_500: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [8, 1024, 768]);  mm_19 = None
        add_67: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_500, arg74_1);  view_500 = arg74_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_68: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_67, convert_element_type_123);  add_67 = convert_element_type_123 = None
        convert_element_type_143: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.float32);  add_68 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_143, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_38: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_143, getitem_17);  convert_element_type_143 = getitem_17 = None
        add_69: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_33: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_8);  sub_38 = rsqrt_8 = None
        mul_34: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, arg75_1);  mul_33 = arg75_1 = None
        add_70: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, arg76_1);  mul_34 = arg76_1 = None
        convert_element_type_144: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_501: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_144, [8192, 768])
        permute_453: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_8: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg78_1, view_501, permute_453);  arg78_1 = view_501 = permute_453 = None
        view_502: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [8, 1024, 3072]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_148: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_502, torch.float32);  view_502 = None
        mul_35: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.5)
        mul_36: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.7071067811865476);  convert_element_type_148 = None
        erf_4: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_71: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_37: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, add_71);  mul_35 = add_71 = None
        convert_element_type_149: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_503: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_149, [8192, 3072]);  convert_element_type_149 = None
        permute_454: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_9: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg80_1, view_503, permute_454);  arg80_1 = view_503 = permute_454 = None
        view_504: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [8, 1024, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_72: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_504, convert_element_type_144);  view_504 = convert_element_type_144 = None
        convert_element_type_153: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.float32);  add_72 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_153, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_52: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_53: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_122: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_123: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 3);  unsqueeze_122 = None
        full_default_23: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_45: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_595: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1)
        slice_593: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1)
        slice_594: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_593, 3, 256, 9223372036854775807);  slice_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_39: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_153, getitem_19);  convert_element_type_153 = getitem_19 = None
        add_73: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_38: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_9);  sub_39 = rsqrt_9 = None
        mul_39: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, arg81_1);  mul_38 = arg81_1 = None
        add_74: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, arg82_1);  mul_39 = arg82_1 = None
        convert_element_type_154: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_455: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_154, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_85: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format)
        view_505: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [8192, 768]);  clone_85 = None
        permute_456: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_20: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_505, permute_456);  view_505 = permute_456 = None
        view_506: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [1024, 8, 768]);  mm_20 = None
        add_75: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_506, arg84_1);  view_506 = arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_50: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_75, 8.0);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_534: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_50, [1024, 8, 12, 64]);  div_50 = None
        permute_480: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_534, [1, 0, 2, 3]);  view_534 = None
        permute_481: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_480, [0, 2, 1, 3]);  permute_480 = None
        view_535: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_481, [96, 1024, 64]);  permute_481 = None
        view_536: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_535, [96, 2, 512, 64]);  view_535 = None
        as_strided_50: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_536, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_536 = None
        unsqueeze_109: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_50, 4);  as_strided_50 = None
        clone_88: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_109, memory_format = torch.contiguous_format);  unsqueeze_109 = None
        view_537: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [288, 512, 64]);  clone_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_86: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format)
        view_507: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [8192, 768]);  clone_86 = None
        permute_457: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_21: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_507, permute_457);  view_507 = permute_457 = None
        view_508: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [1024, 8, 768]);  mm_21 = None
        add_76: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_508, arg86_1);  view_508 = arg86_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_513: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_76, [1024, 8, 12, 64]);  add_76 = None
        permute_460: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_513, [1, 0, 2, 3]);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_465: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_460, [0, 2, 1, 3]);  permute_460 = None
        view_517: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_465, [96, 1024, 64]);  permute_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_524: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_517, [96, 2, 512, 64]);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_46: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_524, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_107: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_46, 4);  as_strided_46 = None
        permute_475: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_107, [0, 1, 4, 2, 3]);  unsqueeze_107 = None
        permute_484: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_475, [0, 1, 4, 3, 2]);  permute_475 = None
        clone_89: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        view_538: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [288, 64, 512]);  clone_89 = None
        bmm_10: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_537, view_538);  view_537 = view_538 = None
        view_539: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [96, 3, 512, 1, 512]);  bmm_10 = None
        permute_485: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_539, [0, 1, 2, 4, 3]);  view_539 = None
        view_540: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_485, [96, 3, 512, 512]);  permute_485 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_20: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_540, [0, 0, 0, 1], 0.0);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_541: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_20, [96, 3, 512, 513]);  constant_pad_nd_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_591: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_541, 2, 0, 256)
        slice_592: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_591, 3, 0, 257);  slice_591 = None
        copy_60: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_594, slice_592);  slice_594 = slice_592 = None
        slice_scatter_110: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_595, copy_60, 3, 256, 9223372036854775807);  slice_595 = copy_60 = None
        slice_scatter_111: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_45, slice_scatter_110, 1, 0, -1);  full_45 = slice_scatter_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_134: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_111, 1, -1)
        select_133: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_111, 1, -1)
        slice_601: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_133, 2, 256, 9223372036854775807);  select_133 = None
        select_130: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_541, 1, -1)
        slice_598: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_130, 1, 256, 9223372036854775807);  select_130 = None
        slice_599: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_598, 2, 0, 257);  slice_598 = None
        copy_61: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_601, slice_599);  slice_601 = slice_599 = None
        slice_scatter_112: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_134, copy_61, 2, 256, 9223372036854775807);  select_134 = copy_61 = None
        select_scatter_20: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_111, slice_scatter_112, 1, -1);  slice_scatter_111 = slice_scatter_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_610: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_20, 1, 1, 9223372036854775807)
        slice_608: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_20, 1, 1, 9223372036854775807)
        slice_609: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_608, 3, 0, 256);  slice_608 = None
        slice_603: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_541, 2, -257, -1)
        slice_604: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_603, 3, 257, 9223372036854775807);  slice_603 = None
        copy_62: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_609, slice_604);  slice_609 = slice_604 = None
        slice_scatter_113: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_610, copy_62, 3, 0, 256);  slice_610 = copy_62 = None
        slice_scatter_114: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_20, slice_scatter_113, 1, 1, 9223372036854775807);  select_scatter_20 = slice_scatter_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_141: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_114, 1, 0)
        slice_620: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_141, 1, 1, 256)
        select_140: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_114, 1, 0)
        slice_618: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_140, 1, 1, 256);  select_140 = None
        slice_619: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_618, 2, 1, 256);  slice_618 = None
        select_136: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_541, 1, 0);  view_541 = None
        slice_613: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_136, 1, 0, 255);  select_136 = None
        slice_614: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_613, 2, -255, 9223372036854775807);  slice_613 = None
        copy_63: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_619, slice_614);  slice_619 = slice_614 = None
        slice_scatter_115: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_620, copy_63, 2, 1, 256);  slice_620 = copy_63 = None
        slice_scatter_116: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_141, slice_scatter_115, 1, 1, 256);  select_141 = slice_scatter_115 = None
        select_scatter_21: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_114, slice_scatter_116, 1, 0);  slice_scatter_114 = slice_scatter_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_549: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_21, [8, 12, 1024, 513])
        permute_493: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_549, [0, 2, 1, 3]);  view_549 = None
        slice_633: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_493, 1, 0, 256)
        view_548: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_21, [8, 12, 1024, 513])
        permute_492: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None
        slice_631: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_492, 1, 0, 256);  permute_492 = None
        slice_632: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_631, 3, 0, 257);  slice_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_20: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_110: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_20, -2);  iota_20 = None
        iota_21: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_111: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
        sub_41: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_110, unsqueeze_111);  unsqueeze_110 = unsqueeze_111 = None
        le_10: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_41, 0);  sub_41 = None
        full_46: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_20: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_40: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_10, full_46, scalar_tensor_20);  le_10 = full_46 = scalar_tensor_20 = None
        rev_20: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_40, [0]);  where_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_112: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_20, 0);  rev_20 = None
        unsqueeze_113: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 2);  unsqueeze_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_20: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_113, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_163: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_20, torch.bool);  expand_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_47: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_490: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_47, [0, 2, 1, 3]);  full_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_546: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_21, [8, 12, 1024, 513]);  select_scatter_21 = None
        permute_489: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
        slice_626: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_489, 1, 0, 256);  permute_489 = None
        slice_627: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_626, 3, 0, 257);  slice_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_41: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_163, permute_490, slice_627);  convert_element_type_163 = permute_490 = slice_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_64: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_632, where_41);  slice_632 = where_41 = None
        slice_scatter_117: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_633, copy_64, 3, 0, 257);  slice_633 = copy_64 = None
        slice_scatter_118: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_493, slice_scatter_117, 1, 0, 256);  permute_493 = slice_scatter_117 = None
        permute_494: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_118, [0, 2, 1, 3]);  slice_scatter_118 = None
        view_550: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_494, [96, 4, 256, 513]);  permute_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_557: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_550, [8, 12, 1024, 513])
        permute_502: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None
        slice_646: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_502, 1, -256, 9223372036854775807)
        view_556: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_550, [8, 12, 1024, 513])
        permute_501: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
        slice_644: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_501, 1, -256, 9223372036854775807);  permute_501 = None
        slice_645: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_644, 3, -257, 9223372036854775807);  slice_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_21: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_113, [1, 3]);  unsqueeze_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_21: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_21, [8, 256, 12, 257]);  rev_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_164: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_21, torch.bool);  expand_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_48: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_499: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_48, [0, 2, 1, 3]);  full_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_554: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_550, [8, 12, 1024, 513]);  view_550 = None
        permute_498: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
        slice_639: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_498, 1, -256, 9223372036854775807);  permute_498 = None
        slice_640: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_639, 3, -257, 9223372036854775807);  slice_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_42: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_164, permute_499, slice_640);  convert_element_type_164 = permute_499 = slice_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_65: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_645, where_42);  slice_645 = where_42 = None
        slice_scatter_119: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_646, copy_65, 3, -257, 9223372036854775807);  slice_646 = copy_65 = None
        slice_scatter_120: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_502, slice_scatter_119, 1, -256, 9223372036854775807);  permute_502 = slice_scatter_119 = None
        permute_503: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_120, [0, 2, 1, 3]);  slice_scatter_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_528: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_503, [0, 2, 1, 3]);  permute_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_50: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_653: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_50, 1, 0, -1)
        slice_651: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_50, 1, 0, -1)
        slice_652: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_651, 3, 256, 9223372036854775807);  slice_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_49: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_505: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_49, [0, 2, 1, 3]);  full_49 = None
        view_560: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_505, [8, 1024, 1]);  permute_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_562: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_560, [8, 2, 512, 1]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_51: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_562, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_116: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_51, 4);  as_strided_51 = None
        permute_507: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_116, [0, 1, 2, 4, 3]);  unsqueeze_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_20: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_5: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_114: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_5, 2);  ne_5 = None
        unsqueeze_115: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 3);  unsqueeze_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_165: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_115, torch.bfloat16)
        where_43: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_115, full_default_20, convert_element_type_165);  unsqueeze_115 = full_default_20 = convert_element_type_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_506: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_43, [0, 2, 1, 3]);  where_43 = None
        view_561: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_506, [8, 1024, 1]);  permute_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_563: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_561, [8, 2, 512, 1]);  view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_52: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_563, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_117: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_52, 4);  as_strided_52 = None
        permute_508: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_117, [0, 1, 4, 2, 3]);  unsqueeze_117 = None
        mul_40: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_507, permute_508);  permute_507 = permute_508 = None
        view_564: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_40, [8, 3, 512, 512]);  mul_40 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_21: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_564, [0, 0, 0, 1], 0.0);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_565: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_21, [8, 3, 512, 513]);  constant_pad_nd_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_649: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_565, 2, 0, 256)
        slice_650: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_649, 3, 0, 257);  slice_649 = None
        copy_66: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_652, slice_650);  slice_652 = slice_650 = None
        slice_scatter_121: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_653, copy_66, 3, 256, 9223372036854775807);  slice_653 = copy_66 = None
        slice_scatter_122: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_50, slice_scatter_121, 1, 0, -1);  full_50 = slice_scatter_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_147: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_122, 1, -1)
        select_146: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_122, 1, -1)
        slice_659: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_146, 2, 256, 9223372036854775807);  select_146 = None
        select_143: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_565, 1, -1)
        slice_656: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_143, 1, 256, 9223372036854775807);  select_143 = None
        slice_657: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_656, 2, 0, 257);  slice_656 = None
        copy_67: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_659, slice_657);  slice_659 = slice_657 = None
        slice_scatter_123: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_147, copy_67, 2, 256, 9223372036854775807);  select_147 = copy_67 = None
        select_scatter_22: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_122, slice_scatter_123, 1, -1);  slice_scatter_122 = slice_scatter_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_668: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_22, 1, 1, 9223372036854775807)
        slice_666: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_22, 1, 1, 9223372036854775807)
        slice_667: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_666, 3, 0, 256);  slice_666 = None
        slice_661: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_565, 2, -257, -1)
        slice_662: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_661, 3, 257, 9223372036854775807);  slice_661 = None
        copy_68: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_667, slice_662);  slice_667 = slice_662 = None
        slice_scatter_124: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_668, copy_68, 3, 0, 256);  slice_668 = copy_68 = None
        slice_scatter_125: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_22, slice_scatter_124, 1, 1, 9223372036854775807);  select_scatter_22 = slice_scatter_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_154: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_125, 1, 0)
        slice_678: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_154, 1, 1, 256)
        select_153: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_125, 1, 0)
        slice_676: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_153, 1, 1, 256);  select_153 = None
        slice_677: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_676, 2, 1, 256);  slice_676 = None
        select_149: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_565, 1, 0);  view_565 = None
        slice_671: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_149, 1, 0, 255);  select_149 = None
        slice_672: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_671, 2, -255, 9223372036854775807);  slice_671 = None
        copy_69: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_677, slice_672);  slice_677 = slice_672 = None
        slice_scatter_126: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_678, copy_69, 2, 1, 256);  slice_678 = copy_69 = None
        slice_scatter_127: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_154, slice_scatter_126, 1, 1, 256);  select_154 = slice_scatter_126 = None
        select_scatter_23: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_125, slice_scatter_127, 1, 0);  slice_scatter_125 = slice_scatter_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_573: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_23, [8, 1, 1024, 513])
        permute_516: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
        slice_691: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_516, 1, 0, 256)
        view_572: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_23, [8, 1, 1024, 513])
        permute_515: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        slice_689: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_515, 1, 0, 256);  permute_515 = None
        slice_690: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_689, 3, 0, 257);  slice_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_22: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_118: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_22, -2);  iota_22 = None
        iota_23: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_119: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
        sub_43: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_118, unsqueeze_119);  unsqueeze_118 = unsqueeze_119 = None
        le_11: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_43, 0);  sub_43 = None
        full_51: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_22: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_44: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_11, full_51, scalar_tensor_22);  le_11 = full_51 = scalar_tensor_22 = None
        rev_22: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_44, [0]);  where_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_120: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_22, 0);  rev_22 = None
        unsqueeze_121: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_22: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_121, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_166: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_22, torch.bool);  expand_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_21: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_570: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_23, [8, 1, 1024, 513]);  select_scatter_23 = None
        permute_512: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
        slice_684: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_512, 1, 0, 256);  permute_512 = None
        slice_685: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_684, 3, 0, 257);  slice_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_45: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_166, full_default_21, slice_685);  convert_element_type_166 = full_default_21 = slice_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_70: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_690, where_45);  slice_690 = where_45 = None
        slice_scatter_128: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_691, copy_70, 3, 0, 257);  slice_691 = copy_70 = None
        slice_scatter_129: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_516, slice_scatter_128, 1, 0, 256);  permute_516 = slice_scatter_128 = None
        permute_517: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_129, [0, 2, 1, 3]);  slice_scatter_129 = None
        view_574: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_517, [8, 4, 256, 513]);  permute_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_581: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_574, [8, 1, 1024, 513])
        permute_525: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_581, [0, 2, 1, 3]);  view_581 = None
        slice_704: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_525, 1, -256, 9223372036854775807)
        view_580: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_574, [8, 1, 1024, 513])
        permute_524: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None
        slice_702: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_524, 1, -256, 9223372036854775807);  permute_524 = None
        slice_703: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_702, 3, -257, 9223372036854775807);  slice_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_23: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_121, [1, 3]);  unsqueeze_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_23: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_23, [8, 256, 1, 257]);  rev_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_167: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_23, torch.bool);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_22: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_578: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_574, [8, 1, 1024, 513]);  view_574 = None
        permute_521: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
        slice_697: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_521, 1, -256, 9223372036854775807);  permute_521 = None
        slice_698: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_697, 3, -257, 9223372036854775807);  slice_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_46: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_167, full_default_22, slice_698);  convert_element_type_167 = full_default_22 = slice_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_71: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_703, where_46);  slice_703 = where_46 = None
        slice_scatter_130: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_704, copy_71, 3, -257, 9223372036854775807);  slice_704 = copy_71 = None
        slice_scatter_131: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_525, slice_scatter_130, 1, -256, 9223372036854775807);  permute_525 = slice_scatter_130 = None
        permute_526: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_131, [0, 2, 1, 3]);  slice_scatter_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_529: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_526, [0, 2, 1, 3]);  permute_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_80: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_528, permute_529);  permute_528 = permute_529 = None
        permute_530: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_80, [0, 2, 1, 3]);  add_80 = None
        permute_531: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_530, [0, 2, 1, 3]);  permute_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_168: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_531, torch.float32);  permute_531 = None
        clone_94: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_168, memory_format = torch.contiguous_format);  convert_element_type_168 = None
        amax_5: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_94, [-1], True)
        sub_44: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_94, amax_5);  clone_94 = amax_5 = None
        exp_5: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
        sum_6: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_57: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_47: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_123, full_default_23, div_57);  unsqueeze_123 = full_default_23 = div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_169: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_47, torch.bfloat16);  where_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_533: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_169, [0, 2, 1, 3]);  convert_element_type_169 = None
        clone_96: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_533, memory_format = torch.contiguous_format);  permute_533 = None
        view_590: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [96, 4, 256, 513]);  clone_96 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_23: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_590, [0, 257], 0.0);  view_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_592: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_23, [96, 4, -1]);  constant_pad_nd_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_707: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_592, 2, 0, -256);  view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_593: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_707, [96, 4, 256, 769]);  slice_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_708: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_593, 3, 0, -1);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_124: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_708, 4);  slice_708 = None
        view_594: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_124, [384, 256, 768]);  unsqueeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_87: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
        view_509: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [8192, 768]);  clone_87 = None
        permute_458: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_22: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_509, permute_458);  view_509 = permute_458 = None
        view_510: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [1024, 8, 768]);  mm_22 = None
        add_77: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_510, arg88_1);  view_510 = arg88_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_589: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_77, [1024, 8, 12, 64]);  add_77 = None
        permute_532: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_589, [1, 0, 2, 3]);  view_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_534: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_532, [0, 2, 1, 3]);  permute_532 = None
        view_591: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_534, [96, 1024, 64]);  permute_534 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_22: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_591, [0, 0, 256, 256], -1.0);  view_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_53: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_22, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_125: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_53, 4);  as_strided_53 = None
        clone_97: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_125, memory_format = torch.contiguous_format);  unsqueeze_125 = None
        view_595: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [384, 768, 64]);  clone_97 = None
        bmm_11: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_594, view_595);  view_594 = view_595 = None
        view_596: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [96, 4, 256, 1, 64]);  bmm_11 = None
        permute_539: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_596, [0, 1, 2, 4, 3]);  view_596 = None
        view_597: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_539, [96, 4, 256, 64]);  permute_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_598: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_597, [8, 12, 1024, 64]);  view_597 = None
        permute_540: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_541: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_540, [1, 0, 2, 3]);  permute_540 = None
        clone_98: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_541, memory_format = torch.contiguous_format);  permute_541 = None
        view_599: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [1024, 8, 768]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_542: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_599, [1, 0, 2]);  view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_99: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_542, memory_format = torch.contiguous_format);  permute_542 = None
        view_600: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [8192, 768]);  clone_99 = None
        permute_543: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_23: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_600, permute_543);  view_600 = permute_543 = None
        view_601: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [8, 1024, 768]);  mm_23 = None
        add_82: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_601, arg90_1);  view_601 = arg90_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_82, convert_element_type_154);  add_82 = convert_element_type_154 = None
        convert_element_type_174: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.float32);  add_83 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_174, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_46: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_174, getitem_21);  convert_element_type_174 = getitem_21 = None
        add_84: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_41: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_10);  sub_46 = rsqrt_10 = None
        mul_42: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, arg91_1);  mul_41 = arg91_1 = None
        add_85: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_42, arg92_1);  mul_42 = arg92_1 = None
        convert_element_type_175: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_602: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_175, [8192, 768])
        permute_544: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_10: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg94_1, view_602, permute_544);  arg94_1 = view_602 = permute_544 = None
        view_603: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [8, 1024, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_179: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_603, torch.float32);  view_603 = None
        mul_43: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_179, 0.5)
        mul_44: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_179, 0.7071067811865476);  convert_element_type_179 = None
        erf_5: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
        add_86: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_45: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, add_86);  mul_43 = add_86 = None
        convert_element_type_180: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_604: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_180, [8192, 3072]);  convert_element_type_180 = None
        permute_545: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_11: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg96_1, view_604, permute_545);  arg96_1 = view_604 = permute_545 = None
        view_605: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [8, 1024, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_605, convert_element_type_175);  view_605 = convert_element_type_175 = None
        convert_element_type_184: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.float32);  add_87 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_184, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_61: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_62: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_143: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_144: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
        full_default_27: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_54: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_713: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_54, 1, 0, -1)
        slice_711: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_54, 1, 0, -1)
        slice_712: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_711, 3, 256, 9223372036854775807);  slice_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_47: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_184, getitem_23);  convert_element_type_184 = getitem_23 = None
        add_88: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_46: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_11);  sub_47 = rsqrt_11 = None
        mul_47: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, arg97_1);  mul_46 = arg97_1 = None
        add_89: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_47, arg98_1);  mul_47 = arg98_1 = None
        convert_element_type_185: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_546: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_102: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_546, memory_format = torch.contiguous_format)
        view_606: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [8192, 768]);  clone_102 = None
        permute_547: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_24: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_606, permute_547);  view_606 = permute_547 = None
        view_607: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [1024, 8, 768]);  mm_24 = None
        add_90: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_607, arg100_1);  view_607 = arg100_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_60: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_90, 8.0);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_635: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_60, [1024, 8, 12, 64]);  div_60 = None
        permute_571: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_635, [1, 0, 2, 3]);  view_635 = None
        permute_572: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_571, [0, 2, 1, 3]);  permute_571 = None
        view_636: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_572, [96, 1024, 64]);  permute_572 = None
        view_637: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_636, [96, 2, 512, 64]);  view_636 = None
        as_strided_59: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_637, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_637 = None
        unsqueeze_130: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_59, 4);  as_strided_59 = None
        clone_105: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_130, memory_format = torch.contiguous_format);  unsqueeze_130 = None
        view_638: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [288, 512, 64]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_103: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_546, memory_format = torch.contiguous_format)
        view_608: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [8192, 768]);  clone_103 = None
        permute_548: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_25: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_608, permute_548);  view_608 = permute_548 = None
        view_609: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [1024, 8, 768]);  mm_25 = None
        add_91: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_609, arg102_1);  view_609 = arg102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_614: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_91, [1024, 8, 12, 64]);  add_91 = None
        permute_551: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_614, [1, 0, 2, 3]);  view_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_556: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_551, [0, 2, 1, 3]);  permute_551 = None
        view_618: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_556, [96, 1024, 64]);  permute_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_625: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_618, [96, 2, 512, 64]);  view_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_55: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_625, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_128: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_55, 4);  as_strided_55 = None
        permute_566: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_128, [0, 1, 4, 2, 3]);  unsqueeze_128 = None
        permute_575: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_566, [0, 1, 4, 3, 2]);  permute_566 = None
        clone_106: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
        view_639: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [288, 64, 512]);  clone_106 = None
        bmm_12: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_638, view_639);  view_638 = view_639 = None
        view_640: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [96, 3, 512, 1, 512]);  bmm_12 = None
        permute_576: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_640, [0, 1, 2, 4, 3]);  view_640 = None
        view_641: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_576, [96, 3, 512, 512]);  permute_576 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_24: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_641, [0, 0, 0, 1], 0.0);  view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_642: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_24, [96, 3, 512, 513]);  constant_pad_nd_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_709: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_642, 2, 0, 256)
        slice_710: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_709, 3, 0, 257);  slice_709 = None
        copy_72: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_712, slice_710);  slice_712 = slice_710 = None
        slice_scatter_132: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_713, copy_72, 3, 256, 9223372036854775807);  slice_713 = copy_72 = None
        slice_scatter_133: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_54, slice_scatter_132, 1, 0, -1);  full_54 = slice_scatter_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_160: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_133, 1, -1)
        select_159: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_133, 1, -1)
        slice_719: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_159, 2, 256, 9223372036854775807);  select_159 = None
        select_156: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_642, 1, -1)
        slice_716: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_156, 1, 256, 9223372036854775807);  select_156 = None
        slice_717: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_716, 2, 0, 257);  slice_716 = None
        copy_73: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_719, slice_717);  slice_719 = slice_717 = None
        slice_scatter_134: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_160, copy_73, 2, 256, 9223372036854775807);  select_160 = copy_73 = None
        select_scatter_24: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_133, slice_scatter_134, 1, -1);  slice_scatter_133 = slice_scatter_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_728: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_24, 1, 1, 9223372036854775807)
        slice_726: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_24, 1, 1, 9223372036854775807)
        slice_727: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_726, 3, 0, 256);  slice_726 = None
        slice_721: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_642, 2, -257, -1)
        slice_722: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_721, 3, 257, 9223372036854775807);  slice_721 = None
        copy_74: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_727, slice_722);  slice_727 = slice_722 = None
        slice_scatter_135: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_728, copy_74, 3, 0, 256);  slice_728 = copy_74 = None
        slice_scatter_136: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_24, slice_scatter_135, 1, 1, 9223372036854775807);  select_scatter_24 = slice_scatter_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_167: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_136, 1, 0)
        slice_738: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_167, 1, 1, 256)
        select_166: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_136, 1, 0)
        slice_736: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_166, 1, 1, 256);  select_166 = None
        slice_737: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_736, 2, 1, 256);  slice_736 = None
        select_162: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_642, 1, 0);  view_642 = None
        slice_731: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_162, 1, 0, 255);  select_162 = None
        slice_732: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_731, 2, -255, 9223372036854775807);  slice_731 = None
        copy_75: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_737, slice_732);  slice_737 = slice_732 = None
        slice_scatter_137: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_738, copy_75, 2, 1, 256);  slice_738 = copy_75 = None
        slice_scatter_138: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_167, slice_scatter_137, 1, 1, 256);  select_167 = slice_scatter_137 = None
        select_scatter_25: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_136, slice_scatter_138, 1, 0);  slice_scatter_136 = slice_scatter_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_650: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_25, [8, 12, 1024, 513])
        permute_584: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_650, [0, 2, 1, 3]);  view_650 = None
        slice_751: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_584, 1, 0, 256)
        view_649: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_25, [8, 12, 1024, 513])
        permute_583: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None
        slice_749: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_583, 1, 0, 256);  permute_583 = None
        slice_750: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_749, 3, 0, 257);  slice_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_24: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_131: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_24, -2);  iota_24 = None
        iota_25: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_132: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
        sub_49: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_131, unsqueeze_132);  unsqueeze_131 = unsqueeze_132 = None
        le_12: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_49, 0);  sub_49 = None
        full_55: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_24: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_48: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_12, full_55, scalar_tensor_24);  le_12 = full_55 = scalar_tensor_24 = None
        rev_24: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_48, [0]);  where_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_133: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_24, 0);  rev_24 = None
        unsqueeze_134: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_24: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_134, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_194: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_24, torch.bool);  expand_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_56: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_581: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_56, [0, 2, 1, 3]);  full_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_647: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_25, [8, 12, 1024, 513]);  select_scatter_25 = None
        permute_580: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_647, [0, 2, 1, 3]);  view_647 = None
        slice_744: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_580, 1, 0, 256);  permute_580 = None
        slice_745: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_744, 3, 0, 257);  slice_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_49: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_194, permute_581, slice_745);  convert_element_type_194 = permute_581 = slice_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_76: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_750, where_49);  slice_750 = where_49 = None
        slice_scatter_139: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_751, copy_76, 3, 0, 257);  slice_751 = copy_76 = None
        slice_scatter_140: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_584, slice_scatter_139, 1, 0, 256);  permute_584 = slice_scatter_139 = None
        permute_585: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_140, [0, 2, 1, 3]);  slice_scatter_140 = None
        view_651: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_585, [96, 4, 256, 513]);  permute_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_658: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_651, [8, 12, 1024, 513])
        permute_593: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_658, [0, 2, 1, 3]);  view_658 = None
        slice_764: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_593, 1, -256, 9223372036854775807)
        view_657: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_651, [8, 12, 1024, 513])
        permute_592: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_657, [0, 2, 1, 3]);  view_657 = None
        slice_762: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_592, 1, -256, 9223372036854775807);  permute_592 = None
        slice_763: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_762, 3, -257, 9223372036854775807);  slice_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_25: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_134, [1, 3]);  unsqueeze_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_25: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_25, [8, 256, 12, 257]);  rev_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_195: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_25, torch.bool);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_57: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_590: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_57, [0, 2, 1, 3]);  full_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_655: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_651, [8, 12, 1024, 513]);  view_651 = None
        permute_589: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_655, [0, 2, 1, 3]);  view_655 = None
        slice_757: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_589, 1, -256, 9223372036854775807);  permute_589 = None
        slice_758: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_757, 3, -257, 9223372036854775807);  slice_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_50: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_195, permute_590, slice_758);  convert_element_type_195 = permute_590 = slice_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_77: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_763, where_50);  slice_763 = where_50 = None
        slice_scatter_141: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_764, copy_77, 3, -257, 9223372036854775807);  slice_764 = copy_77 = None
        slice_scatter_142: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_593, slice_scatter_141, 1, -256, 9223372036854775807);  permute_593 = slice_scatter_141 = None
        permute_594: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_142, [0, 2, 1, 3]);  slice_scatter_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_619: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_594, [0, 2, 1, 3]);  permute_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_59: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_771: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_59, 1, 0, -1)
        slice_769: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_59, 1, 0, -1)
        slice_770: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_769, 3, 256, 9223372036854775807);  slice_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_58: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_596: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_58, [0, 2, 1, 3]);  full_58 = None
        view_661: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_596, [8, 1024, 1]);  permute_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_663: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_661, [8, 2, 512, 1]);  view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_60: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_663, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_137: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_60, 4);  as_strided_60 = None
        permute_598: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_137, [0, 1, 2, 4, 3]);  unsqueeze_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_24: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_6: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_135: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_6, 2);  ne_6 = None
        unsqueeze_136: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_196: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_136, torch.bfloat16)
        where_51: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_136, full_default_24, convert_element_type_196);  unsqueeze_136 = full_default_24 = convert_element_type_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_597: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_51, [0, 2, 1, 3]);  where_51 = None
        view_662: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_597, [8, 1024, 1]);  permute_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_664: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_662, [8, 2, 512, 1]);  view_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_61: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_664, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_138: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_61, 4);  as_strided_61 = None
        permute_599: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_138, [0, 1, 4, 2, 3]);  unsqueeze_138 = None
        mul_48: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_598, permute_599);  permute_598 = permute_599 = None
        view_665: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_48, [8, 3, 512, 512]);  mul_48 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_25: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_665, [0, 0, 0, 1], 0.0);  view_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_666: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_25, [8, 3, 512, 513]);  constant_pad_nd_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_767: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_666, 2, 0, 256)
        slice_768: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_767, 3, 0, 257);  slice_767 = None
        copy_78: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_770, slice_768);  slice_770 = slice_768 = None
        slice_scatter_143: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_771, copy_78, 3, 256, 9223372036854775807);  slice_771 = copy_78 = None
        slice_scatter_144: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_59, slice_scatter_143, 1, 0, -1);  full_59 = slice_scatter_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_173: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_144, 1, -1)
        select_172: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_144, 1, -1)
        slice_777: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_172, 2, 256, 9223372036854775807);  select_172 = None
        select_169: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_666, 1, -1)
        slice_774: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_169, 1, 256, 9223372036854775807);  select_169 = None
        slice_775: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_774, 2, 0, 257);  slice_774 = None
        copy_79: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_777, slice_775);  slice_777 = slice_775 = None
        slice_scatter_145: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_173, copy_79, 2, 256, 9223372036854775807);  select_173 = copy_79 = None
        select_scatter_26: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_144, slice_scatter_145, 1, -1);  slice_scatter_144 = slice_scatter_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_786: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_26, 1, 1, 9223372036854775807)
        slice_784: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_26, 1, 1, 9223372036854775807)
        slice_785: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_784, 3, 0, 256);  slice_784 = None
        slice_779: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_666, 2, -257, -1)
        slice_780: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_779, 3, 257, 9223372036854775807);  slice_779 = None
        copy_80: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_785, slice_780);  slice_785 = slice_780 = None
        slice_scatter_146: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_786, copy_80, 3, 0, 256);  slice_786 = copy_80 = None
        slice_scatter_147: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_26, slice_scatter_146, 1, 1, 9223372036854775807);  select_scatter_26 = slice_scatter_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_180: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_147, 1, 0)
        slice_796: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_180, 1, 1, 256)
        select_179: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_147, 1, 0)
        slice_794: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_179, 1, 1, 256);  select_179 = None
        slice_795: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_794, 2, 1, 256);  slice_794 = None
        select_175: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_666, 1, 0);  view_666 = None
        slice_789: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_175, 1, 0, 255);  select_175 = None
        slice_790: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_789, 2, -255, 9223372036854775807);  slice_789 = None
        copy_81: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_795, slice_790);  slice_795 = slice_790 = None
        slice_scatter_148: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_796, copy_81, 2, 1, 256);  slice_796 = copy_81 = None
        slice_scatter_149: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_180, slice_scatter_148, 1, 1, 256);  select_180 = slice_scatter_148 = None
        select_scatter_27: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_147, slice_scatter_149, 1, 0);  slice_scatter_147 = slice_scatter_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_674: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_27, [8, 1, 1024, 513])
        permute_607: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_674, [0, 2, 1, 3]);  view_674 = None
        slice_809: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_607, 1, 0, 256)
        view_673: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_27, [8, 1, 1024, 513])
        permute_606: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
        slice_807: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_606, 1, 0, 256);  permute_606 = None
        slice_808: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_807, 3, 0, 257);  slice_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_26: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_139: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_26, -2);  iota_26 = None
        iota_27: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_140: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
        sub_51: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_139, unsqueeze_140);  unsqueeze_139 = unsqueeze_140 = None
        le_13: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_51, 0);  sub_51 = None
        full_60: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_26: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_52: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_13, full_60, scalar_tensor_26);  le_13 = full_60 = scalar_tensor_26 = None
        rev_26: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_52, [0]);  where_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_141: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_26, 0);  rev_26 = None
        unsqueeze_142: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_26: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_142, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_197: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_26, torch.bool);  expand_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_25: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_671: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_27, [8, 1, 1024, 513]);  select_scatter_27 = None
        permute_603: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None
        slice_802: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_603, 1, 0, 256);  permute_603 = None
        slice_803: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_802, 3, 0, 257);  slice_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_53: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_197, full_default_25, slice_803);  convert_element_type_197 = full_default_25 = slice_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_82: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_808, where_53);  slice_808 = where_53 = None
        slice_scatter_150: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_809, copy_82, 3, 0, 257);  slice_809 = copy_82 = None
        slice_scatter_151: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_607, slice_scatter_150, 1, 0, 256);  permute_607 = slice_scatter_150 = None
        permute_608: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_151, [0, 2, 1, 3]);  slice_scatter_151 = None
        view_675: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_608, [8, 4, 256, 513]);  permute_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_682: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_675, [8, 1, 1024, 513])
        permute_616: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
        slice_822: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_616, 1, -256, 9223372036854775807)
        view_681: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_675, [8, 1, 1024, 513])
        permute_615: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_681, [0, 2, 1, 3]);  view_681 = None
        slice_820: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_615, 1, -256, 9223372036854775807);  permute_615 = None
        slice_821: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_820, 3, -257, 9223372036854775807);  slice_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_27: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_142, [1, 3]);  unsqueeze_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_27: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_27, [8, 256, 1, 257]);  rev_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_198: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_27, torch.bool);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_26: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_679: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_675, [8, 1, 1024, 513]);  view_675 = None
        permute_612: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_679, [0, 2, 1, 3]);  view_679 = None
        slice_815: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_612, 1, -256, 9223372036854775807);  permute_612 = None
        slice_816: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_815, 3, -257, 9223372036854775807);  slice_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_54: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_198, full_default_26, slice_816);  convert_element_type_198 = full_default_26 = slice_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_83: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_821, where_54);  slice_821 = where_54 = None
        slice_scatter_152: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_822, copy_83, 3, -257, 9223372036854775807);  slice_822 = copy_83 = None
        slice_scatter_153: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_616, slice_scatter_152, 1, -256, 9223372036854775807);  permute_616 = slice_scatter_152 = None
        permute_617: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_153, [0, 2, 1, 3]);  slice_scatter_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_620: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_617, [0, 2, 1, 3]);  permute_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_95: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_619, permute_620);  permute_619 = permute_620 = None
        permute_621: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_95, [0, 2, 1, 3]);  add_95 = None
        permute_622: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_621, [0, 2, 1, 3]);  permute_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_199: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_622, torch.float32);  permute_622 = None
        clone_111: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_199, memory_format = torch.contiguous_format);  convert_element_type_199 = None
        amax_6: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_111, [-1], True)
        sub_52: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_111, amax_6);  clone_111 = amax_6 = None
        exp_6: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_7: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_67: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_55: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_144, full_default_27, div_67);  unsqueeze_144 = full_default_27 = div_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_200: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_55, torch.bfloat16);  where_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_624: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_200, [0, 2, 1, 3]);  convert_element_type_200 = None
        clone_113: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_624, memory_format = torch.contiguous_format);  permute_624 = None
        view_691: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [96, 4, 256, 513]);  clone_113 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_27: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_691, [0, 257], 0.0);  view_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_693: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_27, [96, 4, -1]);  constant_pad_nd_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_825: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_693, 2, 0, -256);  view_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_694: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_825, [96, 4, 256, 769]);  slice_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_826: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_694, 3, 0, -1);  view_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_145: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_826, 4);  slice_826 = None
        view_695: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_145, [384, 256, 768]);  unsqueeze_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_104: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_546, memory_format = torch.contiguous_format);  permute_546 = None
        view_610: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_104, [8192, 768]);  clone_104 = None
        permute_549: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_26: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_610, permute_549);  view_610 = permute_549 = None
        view_611: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [1024, 8, 768]);  mm_26 = None
        add_92: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_611, arg104_1);  view_611 = arg104_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_690: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_92, [1024, 8, 12, 64]);  add_92 = None
        permute_623: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_690, [1, 0, 2, 3]);  view_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_625: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_623, [0, 2, 1, 3]);  permute_623 = None
        view_692: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_625, [96, 1024, 64]);  permute_625 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_26: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_692, [0, 0, 256, 256], -1.0);  view_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_62: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_26, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_146: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_62, 4);  as_strided_62 = None
        clone_114: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_146, memory_format = torch.contiguous_format);  unsqueeze_146 = None
        view_696: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [384, 768, 64]);  clone_114 = None
        bmm_13: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_695, view_696);  view_695 = view_696 = None
        view_697: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [96, 4, 256, 1, 64]);  bmm_13 = None
        permute_630: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_697, [0, 1, 2, 4, 3]);  view_697 = None
        view_698: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_630, [96, 4, 256, 64]);  permute_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_699: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_698, [8, 12, 1024, 64]);  view_698 = None
        permute_631: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_699, [0, 2, 1, 3]);  view_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_632: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_631, [1, 0, 2, 3]);  permute_631 = None
        clone_115: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_632, memory_format = torch.contiguous_format);  permute_632 = None
        view_700: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [1024, 8, 768]);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_633: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_700, [1, 0, 2]);  view_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_116: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_633, memory_format = torch.contiguous_format);  permute_633 = None
        view_701: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [8192, 768]);  clone_116 = None
        permute_634: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_27: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_701, permute_634);  view_701 = permute_634 = None
        view_702: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [8, 1024, 768]);  mm_27 = None
        add_97: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_702, arg106_1);  view_702 = arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_98: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, convert_element_type_185);  add_97 = convert_element_type_185 = None
        convert_element_type_205: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_98, torch.float32);  add_98 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_205, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_54: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_205, getitem_25);  convert_element_type_205 = getitem_25 = None
        add_99: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_49: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_12);  sub_54 = rsqrt_12 = None
        mul_50: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, arg107_1);  mul_49 = arg107_1 = None
        add_100: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, arg108_1);  mul_50 = arg108_1 = None
        convert_element_type_206: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_703: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_206, [8192, 768])
        permute_635: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_12: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg110_1, view_703, permute_635);  arg110_1 = view_703 = permute_635 = None
        view_704: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [8, 1024, 3072]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_210: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_704, torch.float32);  view_704 = None
        mul_51: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.5)
        mul_52: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.7071067811865476);  convert_element_type_210 = None
        erf_6: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_101: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_53: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, add_101);  mul_51 = add_101 = None
        convert_element_type_211: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_705: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_211, [8192, 3072]);  convert_element_type_211 = None
        permute_636: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_13: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg112_1, view_705, permute_636);  arg112_1 = view_705 = permute_636 = None
        view_706: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [8, 1024, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_102: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_706, convert_element_type_206);  view_706 = convert_element_type_206 = None
        convert_element_type_215: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.float32);  add_102 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_215, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_70: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_71: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_164: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_165: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        full_default_31: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_63: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_831: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_63, 1, 0, -1)
        slice_829: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_63, 1, 0, -1)
        slice_830: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_829, 3, 256, 9223372036854775807);  slice_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_55: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_215, getitem_27);  convert_element_type_215 = getitem_27 = None
        add_103: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_54: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_13);  sub_55 = rsqrt_13 = None
        mul_55: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, arg113_1);  mul_54 = arg113_1 = None
        add_104: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, arg114_1);  mul_55 = arg114_1 = None
        convert_element_type_216: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_637: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_216, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_119: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_637, memory_format = torch.contiguous_format)
        view_707: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_119, [8192, 768]);  clone_119 = None
        permute_638: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_28: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_707, permute_638);  view_707 = permute_638 = None
        view_708: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [1024, 8, 768]);  mm_28 = None
        add_105: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_708, arg116_1);  view_708 = arg116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_70: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_105, 8.0);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_736: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_70, [1024, 8, 12, 64]);  div_70 = None
        permute_662: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_736, [1, 0, 2, 3]);  view_736 = None
        permute_663: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_662, [0, 2, 1, 3]);  permute_662 = None
        view_737: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_663, [96, 1024, 64]);  permute_663 = None
        view_738: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_737, [96, 2, 512, 64]);  view_737 = None
        as_strided_68: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_738, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_738 = None
        unsqueeze_151: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_68, 4);  as_strided_68 = None
        clone_122: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_151, memory_format = torch.contiguous_format);  unsqueeze_151 = None
        view_739: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [288, 512, 64]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_120: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_637, memory_format = torch.contiguous_format)
        view_709: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [8192, 768]);  clone_120 = None
        permute_639: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_29: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_709, permute_639);  view_709 = permute_639 = None
        view_710: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [1024, 8, 768]);  mm_29 = None
        add_106: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_710, arg118_1);  view_710 = arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_715: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_106, [1024, 8, 12, 64]);  add_106 = None
        permute_642: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_715, [1, 0, 2, 3]);  view_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_647: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_642, [0, 2, 1, 3]);  permute_642 = None
        view_719: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_647, [96, 1024, 64]);  permute_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_726: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_719, [96, 2, 512, 64]);  view_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_64: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_726, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_149: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_64, 4);  as_strided_64 = None
        permute_657: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_149, [0, 1, 4, 2, 3]);  unsqueeze_149 = None
        permute_666: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_657, [0, 1, 4, 3, 2]);  permute_657 = None
        clone_123: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_666, memory_format = torch.contiguous_format);  permute_666 = None
        view_740: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [288, 64, 512]);  clone_123 = None
        bmm_14: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_739, view_740);  view_739 = view_740 = None
        view_741: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [96, 3, 512, 1, 512]);  bmm_14 = None
        permute_667: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_741, [0, 1, 2, 4, 3]);  view_741 = None
        view_742: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_667, [96, 3, 512, 512]);  permute_667 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_28: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_742, [0, 0, 0, 1], 0.0);  view_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_743: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_28, [96, 3, 512, 513]);  constant_pad_nd_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_827: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_743, 2, 0, 256)
        slice_828: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_827, 3, 0, 257);  slice_827 = None
        copy_84: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_830, slice_828);  slice_830 = slice_828 = None
        slice_scatter_154: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_831, copy_84, 3, 256, 9223372036854775807);  slice_831 = copy_84 = None
        slice_scatter_155: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_63, slice_scatter_154, 1, 0, -1);  full_63 = slice_scatter_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_186: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_155, 1, -1)
        select_185: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_155, 1, -1)
        slice_837: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_185, 2, 256, 9223372036854775807);  select_185 = None
        select_182: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_743, 1, -1)
        slice_834: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_182, 1, 256, 9223372036854775807);  select_182 = None
        slice_835: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_834, 2, 0, 257);  slice_834 = None
        copy_85: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_837, slice_835);  slice_837 = slice_835 = None
        slice_scatter_156: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_186, copy_85, 2, 256, 9223372036854775807);  select_186 = copy_85 = None
        select_scatter_28: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_155, slice_scatter_156, 1, -1);  slice_scatter_155 = slice_scatter_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_846: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_28, 1, 1, 9223372036854775807)
        slice_844: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_28, 1, 1, 9223372036854775807)
        slice_845: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_844, 3, 0, 256);  slice_844 = None
        slice_839: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_743, 2, -257, -1)
        slice_840: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_839, 3, 257, 9223372036854775807);  slice_839 = None
        copy_86: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_845, slice_840);  slice_845 = slice_840 = None
        slice_scatter_157: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_846, copy_86, 3, 0, 256);  slice_846 = copy_86 = None
        slice_scatter_158: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_28, slice_scatter_157, 1, 1, 9223372036854775807);  select_scatter_28 = slice_scatter_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_193: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_158, 1, 0)
        slice_856: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_193, 1, 1, 256)
        select_192: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_158, 1, 0)
        slice_854: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_192, 1, 1, 256);  select_192 = None
        slice_855: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_854, 2, 1, 256);  slice_854 = None
        select_188: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_743, 1, 0);  view_743 = None
        slice_849: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_188, 1, 0, 255);  select_188 = None
        slice_850: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_849, 2, -255, 9223372036854775807);  slice_849 = None
        copy_87: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_855, slice_850);  slice_855 = slice_850 = None
        slice_scatter_159: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_856, copy_87, 2, 1, 256);  slice_856 = copy_87 = None
        slice_scatter_160: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_193, slice_scatter_159, 1, 1, 256);  select_193 = slice_scatter_159 = None
        select_scatter_29: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_158, slice_scatter_160, 1, 0);  slice_scatter_158 = slice_scatter_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_751: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_29, [8, 12, 1024, 513])
        permute_675: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_751, [0, 2, 1, 3]);  view_751 = None
        slice_869: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_675, 1, 0, 256)
        view_750: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_29, [8, 12, 1024, 513])
        permute_674: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_750, [0, 2, 1, 3]);  view_750 = None
        slice_867: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_674, 1, 0, 256);  permute_674 = None
        slice_868: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_867, 3, 0, 257);  slice_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_28: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_152: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_28, -2);  iota_28 = None
        iota_29: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_153: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
        sub_57: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_152, unsqueeze_153);  unsqueeze_152 = unsqueeze_153 = None
        le_14: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_57, 0);  sub_57 = None
        full_64: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_28: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_56: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_14, full_64, scalar_tensor_28);  le_14 = full_64 = scalar_tensor_28 = None
        rev_28: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_56, [0]);  where_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_154: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_28, 0);  rev_28 = None
        unsqueeze_155: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_28: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_155, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_225: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_28, torch.bool);  expand_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_65: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_672: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_65, [0, 2, 1, 3]);  full_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_748: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_29, [8, 12, 1024, 513]);  select_scatter_29 = None
        permute_671: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None
        slice_862: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_671, 1, 0, 256);  permute_671 = None
        slice_863: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_862, 3, 0, 257);  slice_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_57: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_225, permute_672, slice_863);  convert_element_type_225 = permute_672 = slice_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_88: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_868, where_57);  slice_868 = where_57 = None
        slice_scatter_161: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_869, copy_88, 3, 0, 257);  slice_869 = copy_88 = None
        slice_scatter_162: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_675, slice_scatter_161, 1, 0, 256);  permute_675 = slice_scatter_161 = None
        permute_676: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_162, [0, 2, 1, 3]);  slice_scatter_162 = None
        view_752: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_676, [96, 4, 256, 513]);  permute_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_759: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_752, [8, 12, 1024, 513])
        permute_684: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_759, [0, 2, 1, 3]);  view_759 = None
        slice_882: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_684, 1, -256, 9223372036854775807)
        view_758: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_752, [8, 12, 1024, 513])
        permute_683: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_758, [0, 2, 1, 3]);  view_758 = None
        slice_880: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_683, 1, -256, 9223372036854775807);  permute_683 = None
        slice_881: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_880, 3, -257, 9223372036854775807);  slice_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_29: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_155, [1, 3]);  unsqueeze_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_29: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_29, [8, 256, 12, 257]);  rev_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_226: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_29, torch.bool);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_66: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_681: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_66, [0, 2, 1, 3]);  full_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_756: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_752, [8, 12, 1024, 513]);  view_752 = None
        permute_680: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_756, [0, 2, 1, 3]);  view_756 = None
        slice_875: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_680, 1, -256, 9223372036854775807);  permute_680 = None
        slice_876: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_875, 3, -257, 9223372036854775807);  slice_875 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_58: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_226, permute_681, slice_876);  convert_element_type_226 = permute_681 = slice_876 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_89: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_881, where_58);  slice_881 = where_58 = None
        slice_scatter_163: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_882, copy_89, 3, -257, 9223372036854775807);  slice_882 = copy_89 = None
        slice_scatter_164: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_684, slice_scatter_163, 1, -256, 9223372036854775807);  permute_684 = slice_scatter_163 = None
        permute_685: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_164, [0, 2, 1, 3]);  slice_scatter_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_710: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_685, [0, 2, 1, 3]);  permute_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_68: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_889: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_68, 1, 0, -1)
        slice_887: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_68, 1, 0, -1)
        slice_888: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_887, 3, 256, 9223372036854775807);  slice_887 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_67: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_687: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_67, [0, 2, 1, 3]);  full_67 = None
        view_762: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_687, [8, 1024, 1]);  permute_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_764: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_762, [8, 2, 512, 1]);  view_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_69: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_764, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_158: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_69, 4);  as_strided_69 = None
        permute_689: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_158, [0, 1, 2, 4, 3]);  unsqueeze_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_28: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_7: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_156: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_7, 2);  ne_7 = None
        unsqueeze_157: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_227: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_157, torch.bfloat16)
        where_59: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_157, full_default_28, convert_element_type_227);  unsqueeze_157 = full_default_28 = convert_element_type_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_688: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_59, [0, 2, 1, 3]);  where_59 = None
        view_763: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_688, [8, 1024, 1]);  permute_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_765: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_763, [8, 2, 512, 1]);  view_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_70: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_765, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_159: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_70, 4);  as_strided_70 = None
        permute_690: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_159, [0, 1, 4, 2, 3]);  unsqueeze_159 = None
        mul_56: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_689, permute_690);  permute_689 = permute_690 = None
        view_766: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_56, [8, 3, 512, 512]);  mul_56 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_29: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_766, [0, 0, 0, 1], 0.0);  view_766 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_767: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_29, [8, 3, 512, 513]);  constant_pad_nd_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_885: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_767, 2, 0, 256)
        slice_886: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_885, 3, 0, 257);  slice_885 = None
        copy_90: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_888, slice_886);  slice_888 = slice_886 = None
        slice_scatter_165: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_889, copy_90, 3, 256, 9223372036854775807);  slice_889 = copy_90 = None
        slice_scatter_166: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_68, slice_scatter_165, 1, 0, -1);  full_68 = slice_scatter_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_199: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_166, 1, -1)
        select_198: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_166, 1, -1)
        slice_895: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_198, 2, 256, 9223372036854775807);  select_198 = None
        select_195: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_767, 1, -1)
        slice_892: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_195, 1, 256, 9223372036854775807);  select_195 = None
        slice_893: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_892, 2, 0, 257);  slice_892 = None
        copy_91: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_895, slice_893);  slice_895 = slice_893 = None
        slice_scatter_167: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_199, copy_91, 2, 256, 9223372036854775807);  select_199 = copy_91 = None
        select_scatter_30: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_166, slice_scatter_167, 1, -1);  slice_scatter_166 = slice_scatter_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_904: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_30, 1, 1, 9223372036854775807)
        slice_902: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_30, 1, 1, 9223372036854775807)
        slice_903: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_902, 3, 0, 256);  slice_902 = None
        slice_897: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_767, 2, -257, -1)
        slice_898: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_897, 3, 257, 9223372036854775807);  slice_897 = None
        copy_92: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_903, slice_898);  slice_903 = slice_898 = None
        slice_scatter_168: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_904, copy_92, 3, 0, 256);  slice_904 = copy_92 = None
        slice_scatter_169: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_30, slice_scatter_168, 1, 1, 9223372036854775807);  select_scatter_30 = slice_scatter_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_206: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_169, 1, 0)
        slice_914: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_206, 1, 1, 256)
        select_205: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_169, 1, 0)
        slice_912: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_205, 1, 1, 256);  select_205 = None
        slice_913: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_912, 2, 1, 256);  slice_912 = None
        select_201: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_767, 1, 0);  view_767 = None
        slice_907: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_201, 1, 0, 255);  select_201 = None
        slice_908: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_907, 2, -255, 9223372036854775807);  slice_907 = None
        copy_93: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_913, slice_908);  slice_913 = slice_908 = None
        slice_scatter_170: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_914, copy_93, 2, 1, 256);  slice_914 = copy_93 = None
        slice_scatter_171: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_206, slice_scatter_170, 1, 1, 256);  select_206 = slice_scatter_170 = None
        select_scatter_31: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_169, slice_scatter_171, 1, 0);  slice_scatter_169 = slice_scatter_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_775: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_31, [8, 1, 1024, 513])
        permute_698: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_775, [0, 2, 1, 3]);  view_775 = None
        slice_927: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_698, 1, 0, 256)
        view_774: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_31, [8, 1, 1024, 513])
        permute_697: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None
        slice_925: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_697, 1, 0, 256);  permute_697 = None
        slice_926: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_925, 3, 0, 257);  slice_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_30: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_160: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_30, -2);  iota_30 = None
        iota_31: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_161: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
        sub_59: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_160, unsqueeze_161);  unsqueeze_160 = unsqueeze_161 = None
        le_15: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_59, 0);  sub_59 = None
        full_69: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_30: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_60: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_15, full_69, scalar_tensor_30);  le_15 = full_69 = scalar_tensor_30 = None
        rev_30: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_60, [0]);  where_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_162: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_30, 0);  rev_30 = None
        unsqueeze_163: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 2);  unsqueeze_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_30: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_163, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_228: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_30, torch.bool);  expand_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_29: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_772: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_31, [8, 1, 1024, 513]);  select_scatter_31 = None
        permute_694: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None
        slice_920: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_694, 1, 0, 256);  permute_694 = None
        slice_921: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_920, 3, 0, 257);  slice_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_61: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_228, full_default_29, slice_921);  convert_element_type_228 = full_default_29 = slice_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_94: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_926, where_61);  slice_926 = where_61 = None
        slice_scatter_172: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_927, copy_94, 3, 0, 257);  slice_927 = copy_94 = None
        slice_scatter_173: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_698, slice_scatter_172, 1, 0, 256);  permute_698 = slice_scatter_172 = None
        permute_699: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_173, [0, 2, 1, 3]);  slice_scatter_173 = None
        view_776: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_699, [8, 4, 256, 513]);  permute_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_783: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_776, [8, 1, 1024, 513])
        permute_707: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_783, [0, 2, 1, 3]);  view_783 = None
        slice_940: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_707, 1, -256, 9223372036854775807)
        view_782: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_776, [8, 1, 1024, 513])
        permute_706: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_782, [0, 2, 1, 3]);  view_782 = None
        slice_938: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_706, 1, -256, 9223372036854775807);  permute_706 = None
        slice_939: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_938, 3, -257, 9223372036854775807);  slice_938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_31: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_163, [1, 3]);  unsqueeze_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_31: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_31, [8, 256, 1, 257]);  rev_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_229: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_31, torch.bool);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_30: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_780: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_776, [8, 1, 1024, 513]);  view_776 = None
        permute_703: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_780, [0, 2, 1, 3]);  view_780 = None
        slice_933: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_703, 1, -256, 9223372036854775807);  permute_703 = None
        slice_934: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_933, 3, -257, 9223372036854775807);  slice_933 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_62: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_229, full_default_30, slice_934);  convert_element_type_229 = full_default_30 = slice_934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_95: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_939, where_62);  slice_939 = where_62 = None
        slice_scatter_174: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_940, copy_95, 3, -257, 9223372036854775807);  slice_940 = copy_95 = None
        slice_scatter_175: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_707, slice_scatter_174, 1, -256, 9223372036854775807);  permute_707 = slice_scatter_174 = None
        permute_708: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_175, [0, 2, 1, 3]);  slice_scatter_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_711: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_708, [0, 2, 1, 3]);  permute_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_110: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_710, permute_711);  permute_710 = permute_711 = None
        permute_712: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_110, [0, 2, 1, 3]);  add_110 = None
        permute_713: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_712, [0, 2, 1, 3]);  permute_712 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_230: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_713, torch.float32);  permute_713 = None
        clone_128: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_230, memory_format = torch.contiguous_format);  convert_element_type_230 = None
        amax_7: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_128, [-1], True)
        sub_60: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_128, amax_7);  clone_128 = amax_7 = None
        exp_7: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_8: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_77: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_63: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_165, full_default_31, div_77);  unsqueeze_165 = full_default_31 = div_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_231: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_63, torch.bfloat16);  where_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_715: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_231, [0, 2, 1, 3]);  convert_element_type_231 = None
        clone_130: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_715, memory_format = torch.contiguous_format);  permute_715 = None
        view_792: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [96, 4, 256, 513]);  clone_130 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_31: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_792, [0, 257], 0.0);  view_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_794: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_31, [96, 4, -1]);  constant_pad_nd_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_943: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_794, 2, 0, -256);  view_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_795: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_943, [96, 4, 256, 769]);  slice_943 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_944: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_795, 3, 0, -1);  view_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_166: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_944, 4);  slice_944 = None
        view_796: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_166, [384, 256, 768]);  unsqueeze_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_121: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_637, memory_format = torch.contiguous_format);  permute_637 = None
        view_711: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [8192, 768]);  clone_121 = None
        permute_640: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_30: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_711, permute_640);  view_711 = permute_640 = None
        view_712: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [1024, 8, 768]);  mm_30 = None
        add_107: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_712, arg120_1);  view_712 = arg120_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_791: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_107, [1024, 8, 12, 64]);  add_107 = None
        permute_714: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_791, [1, 0, 2, 3]);  view_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_716: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_714, [0, 2, 1, 3]);  permute_714 = None
        view_793: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_716, [96, 1024, 64]);  permute_716 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_30: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_793, [0, 0, 256, 256], -1.0);  view_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_71: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_30, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_167: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_71, 4);  as_strided_71 = None
        clone_131: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_167, memory_format = torch.contiguous_format);  unsqueeze_167 = None
        view_797: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [384, 768, 64]);  clone_131 = None
        bmm_15: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_796, view_797);  view_796 = view_797 = None
        view_798: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [96, 4, 256, 1, 64]);  bmm_15 = None
        permute_721: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_798, [0, 1, 2, 4, 3]);  view_798 = None
        view_799: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_721, [96, 4, 256, 64]);  permute_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_800: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_799, [8, 12, 1024, 64]);  view_799 = None
        permute_722: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_800, [0, 2, 1, 3]);  view_800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_723: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_722, [1, 0, 2, 3]);  permute_722 = None
        clone_132: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_723, memory_format = torch.contiguous_format);  permute_723 = None
        view_801: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_132, [1024, 8, 768]);  clone_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_724: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_801, [1, 0, 2]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_133: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_724, memory_format = torch.contiguous_format);  permute_724 = None
        view_802: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_133, [8192, 768]);  clone_133 = None
        permute_725: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_31: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_802, permute_725);  view_802 = permute_725 = None
        view_803: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [8, 1024, 768]);  mm_31 = None
        add_112: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_803, arg122_1);  view_803 = arg122_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_113: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_112, convert_element_type_216);  add_112 = convert_element_type_216 = None
        convert_element_type_236: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.float32);  add_113 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_236, [2], correction = 0, keepdim = True)
        getitem_28: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_62: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_236, getitem_29);  convert_element_type_236 = getitem_29 = None
        add_114: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_57: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = rsqrt_14 = None
        mul_58: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg123_1);  mul_57 = arg123_1 = None
        add_115: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg124_1);  mul_58 = arg124_1 = None
        convert_element_type_237: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_804: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_237, [8192, 768])
        permute_726: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_14: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg126_1, view_804, permute_726);  arg126_1 = view_804 = permute_726 = None
        view_805: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [8, 1024, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_241: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_805, torch.float32);  view_805 = None
        mul_59: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_241, 0.5)
        mul_60: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_241, 0.7071067811865476);  convert_element_type_241 = None
        erf_7: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_116: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_61: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, add_116);  mul_59 = add_116 = None
        convert_element_type_242: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_806: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_242, [8192, 3072]);  convert_element_type_242 = None
        permute_727: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_15: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg128_1, view_806, permute_727);  arg128_1 = view_806 = permute_727 = None
        view_807: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [8, 1024, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_117: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_807, convert_element_type_237);  view_807 = convert_element_type_237 = None
        convert_element_type_246: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.float32);  add_117 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_246, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_79: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_80: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_185: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_186: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
        full_default_35: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_72: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_949: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_72, 1, 0, -1)
        slice_947: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_72, 1, 0, -1)
        slice_948: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_947, 3, 256, 9223372036854775807);  slice_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_63: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_246, getitem_31);  convert_element_type_246 = getitem_31 = None
        add_118: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_62: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_15);  sub_63 = rsqrt_15 = None
        mul_63: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, arg129_1);  mul_62 = arg129_1 = None
        add_119: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_63, arg130_1);  mul_63 = arg130_1 = None
        convert_element_type_247: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_728: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_247, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_136: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_728, memory_format = torch.contiguous_format)
        view_808: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [8192, 768]);  clone_136 = None
        permute_729: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_32: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_808, permute_729);  view_808 = permute_729 = None
        view_809: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [1024, 8, 768]);  mm_32 = None
        add_120: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_809, arg132_1);  view_809 = arg132_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_80: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_120, 8.0);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_837: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_80, [1024, 8, 12, 64]);  div_80 = None
        permute_753: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_837, [1, 0, 2, 3]);  view_837 = None
        permute_754: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_753, [0, 2, 1, 3]);  permute_753 = None
        view_838: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_754, [96, 1024, 64]);  permute_754 = None
        view_839: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_838, [96, 2, 512, 64]);  view_838 = None
        as_strided_77: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_839, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_839 = None
        unsqueeze_172: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_77, 4);  as_strided_77 = None
        clone_139: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_172, memory_format = torch.contiguous_format);  unsqueeze_172 = None
        view_840: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_139, [288, 512, 64]);  clone_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_137: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_728, memory_format = torch.contiguous_format)
        view_810: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [8192, 768]);  clone_137 = None
        permute_730: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_33: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_810, permute_730);  view_810 = permute_730 = None
        view_811: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [1024, 8, 768]);  mm_33 = None
        add_121: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_811, arg134_1);  view_811 = arg134_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_816: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_121, [1024, 8, 12, 64]);  add_121 = None
        permute_733: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_816, [1, 0, 2, 3]);  view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_738: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_733, [0, 2, 1, 3]);  permute_733 = None
        view_820: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_738, [96, 1024, 64]);  permute_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_827: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_820, [96, 2, 512, 64]);  view_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_73: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_827, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_170: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_73, 4);  as_strided_73 = None
        permute_748: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_170, [0, 1, 4, 2, 3]);  unsqueeze_170 = None
        permute_757: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_748, [0, 1, 4, 3, 2]);  permute_748 = None
        clone_140: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_757, memory_format = torch.contiguous_format);  permute_757 = None
        view_841: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_140, [288, 64, 512]);  clone_140 = None
        bmm_16: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_840, view_841);  view_840 = view_841 = None
        view_842: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [96, 3, 512, 1, 512]);  bmm_16 = None
        permute_758: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_842, [0, 1, 2, 4, 3]);  view_842 = None
        view_843: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_758, [96, 3, 512, 512]);  permute_758 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_32: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_843, [0, 0, 0, 1], 0.0);  view_843 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_844: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_32, [96, 3, 512, 513]);  constant_pad_nd_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_945: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_844, 2, 0, 256)
        slice_946: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_945, 3, 0, 257);  slice_945 = None
        copy_96: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_948, slice_946);  slice_948 = slice_946 = None
        slice_scatter_176: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_949, copy_96, 3, 256, 9223372036854775807);  slice_949 = copy_96 = None
        slice_scatter_177: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_72, slice_scatter_176, 1, 0, -1);  full_72 = slice_scatter_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_212: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_177, 1, -1)
        select_211: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_177, 1, -1)
        slice_955: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_211, 2, 256, 9223372036854775807);  select_211 = None
        select_208: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_844, 1, -1)
        slice_952: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_208, 1, 256, 9223372036854775807);  select_208 = None
        slice_953: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_952, 2, 0, 257);  slice_952 = None
        copy_97: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_955, slice_953);  slice_955 = slice_953 = None
        slice_scatter_178: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_212, copy_97, 2, 256, 9223372036854775807);  select_212 = copy_97 = None
        select_scatter_32: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_177, slice_scatter_178, 1, -1);  slice_scatter_177 = slice_scatter_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_964: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_32, 1, 1, 9223372036854775807)
        slice_962: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_32, 1, 1, 9223372036854775807)
        slice_963: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_962, 3, 0, 256);  slice_962 = None
        slice_957: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_844, 2, -257, -1)
        slice_958: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_957, 3, 257, 9223372036854775807);  slice_957 = None
        copy_98: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_963, slice_958);  slice_963 = slice_958 = None
        slice_scatter_179: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_964, copy_98, 3, 0, 256);  slice_964 = copy_98 = None
        slice_scatter_180: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_32, slice_scatter_179, 1, 1, 9223372036854775807);  select_scatter_32 = slice_scatter_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_219: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_180, 1, 0)
        slice_974: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_219, 1, 1, 256)
        select_218: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_180, 1, 0)
        slice_972: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_218, 1, 1, 256);  select_218 = None
        slice_973: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_972, 2, 1, 256);  slice_972 = None
        select_214: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_844, 1, 0);  view_844 = None
        slice_967: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_214, 1, 0, 255);  select_214 = None
        slice_968: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_967, 2, -255, 9223372036854775807);  slice_967 = None
        copy_99: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_973, slice_968);  slice_973 = slice_968 = None
        slice_scatter_181: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_974, copy_99, 2, 1, 256);  slice_974 = copy_99 = None
        slice_scatter_182: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_219, slice_scatter_181, 1, 1, 256);  select_219 = slice_scatter_181 = None
        select_scatter_33: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_180, slice_scatter_182, 1, 0);  slice_scatter_180 = slice_scatter_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_852: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_33, [8, 12, 1024, 513])
        permute_766: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_852, [0, 2, 1, 3]);  view_852 = None
        slice_987: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_766, 1, 0, 256)
        view_851: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_33, [8, 12, 1024, 513])
        permute_765: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_851, [0, 2, 1, 3]);  view_851 = None
        slice_985: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_765, 1, 0, 256);  permute_765 = None
        slice_986: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_985, 3, 0, 257);  slice_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_32: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_173: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_32, -2);  iota_32 = None
        iota_33: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_174: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
        sub_65: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_173, unsqueeze_174);  unsqueeze_173 = unsqueeze_174 = None
        le_16: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_65, 0);  sub_65 = None
        full_73: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_32: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_64: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_16, full_73, scalar_tensor_32);  le_16 = full_73 = scalar_tensor_32 = None
        rev_32: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_64, [0]);  where_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_175: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_32, 0);  rev_32 = None
        unsqueeze_176: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_32: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_176, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_256: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_32, torch.bool);  expand_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_74: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_763: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_74, [0, 2, 1, 3]);  full_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_849: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_33, [8, 12, 1024, 513]);  select_scatter_33 = None
        permute_762: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_849, [0, 2, 1, 3]);  view_849 = None
        slice_980: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_762, 1, 0, 256);  permute_762 = None
        slice_981: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_980, 3, 0, 257);  slice_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_65: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_256, permute_763, slice_981);  convert_element_type_256 = permute_763 = slice_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_100: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_986, where_65);  slice_986 = where_65 = None
        slice_scatter_183: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_987, copy_100, 3, 0, 257);  slice_987 = copy_100 = None
        slice_scatter_184: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_766, slice_scatter_183, 1, 0, 256);  permute_766 = slice_scatter_183 = None
        permute_767: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_184, [0, 2, 1, 3]);  slice_scatter_184 = None
        view_853: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_767, [96, 4, 256, 513]);  permute_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_860: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_853, [8, 12, 1024, 513])
        permute_775: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_860, [0, 2, 1, 3]);  view_860 = None
        slice_1000: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_775, 1, -256, 9223372036854775807)
        view_859: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_853, [8, 12, 1024, 513])
        permute_774: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_859, [0, 2, 1, 3]);  view_859 = None
        slice_998: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_774, 1, -256, 9223372036854775807);  permute_774 = None
        slice_999: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_998, 3, -257, 9223372036854775807);  slice_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_33: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_176, [1, 3]);  unsqueeze_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_33: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_33, [8, 256, 12, 257]);  rev_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_257: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_33, torch.bool);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_75: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_772: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_75, [0, 2, 1, 3]);  full_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_857: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_853, [8, 12, 1024, 513]);  view_853 = None
        permute_771: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_857, [0, 2, 1, 3]);  view_857 = None
        slice_993: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_771, 1, -256, 9223372036854775807);  permute_771 = None
        slice_994: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_993, 3, -257, 9223372036854775807);  slice_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_66: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_257, permute_772, slice_994);  convert_element_type_257 = permute_772 = slice_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_101: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_999, where_66);  slice_999 = where_66 = None
        slice_scatter_185: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1000, copy_101, 3, -257, 9223372036854775807);  slice_1000 = copy_101 = None
        slice_scatter_186: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_775, slice_scatter_185, 1, -256, 9223372036854775807);  permute_775 = slice_scatter_185 = None
        permute_776: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_186, [0, 2, 1, 3]);  slice_scatter_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_801: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_776, [0, 2, 1, 3]);  permute_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_77: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1007: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_77, 1, 0, -1)
        slice_1005: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_77, 1, 0, -1)
        slice_1006: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1005, 3, 256, 9223372036854775807);  slice_1005 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_76: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_778: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_76, [0, 2, 1, 3]);  full_76 = None
        view_863: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_778, [8, 1024, 1]);  permute_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_865: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_863, [8, 2, 512, 1]);  view_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_78: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_865, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_179: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_78, 4);  as_strided_78 = None
        permute_780: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_179, [0, 1, 2, 4, 3]);  unsqueeze_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_32: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_8: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_177: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_8, 2);  ne_8 = None
        unsqueeze_178: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_258: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_178, torch.bfloat16)
        where_67: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_178, full_default_32, convert_element_type_258);  unsqueeze_178 = full_default_32 = convert_element_type_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_779: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_67, [0, 2, 1, 3]);  where_67 = None
        view_864: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_779, [8, 1024, 1]);  permute_779 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_866: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_864, [8, 2, 512, 1]);  view_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_79: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_866, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_180: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_79, 4);  as_strided_79 = None
        permute_781: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_180, [0, 1, 4, 2, 3]);  unsqueeze_180 = None
        mul_64: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_780, permute_781);  permute_780 = permute_781 = None
        view_867: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_64, [8, 3, 512, 512]);  mul_64 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_33: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_867, [0, 0, 0, 1], 0.0);  view_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_868: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_33, [8, 3, 512, 513]);  constant_pad_nd_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1003: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_868, 2, 0, 256)
        slice_1004: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1003, 3, 0, 257);  slice_1003 = None
        copy_102: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1006, slice_1004);  slice_1006 = slice_1004 = None
        slice_scatter_187: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1007, copy_102, 3, 256, 9223372036854775807);  slice_1007 = copy_102 = None
        slice_scatter_188: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_77, slice_scatter_187, 1, 0, -1);  full_77 = slice_scatter_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_225: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_188, 1, -1)
        select_224: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_188, 1, -1)
        slice_1013: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_224, 2, 256, 9223372036854775807);  select_224 = None
        select_221: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_868, 1, -1)
        slice_1010: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_221, 1, 256, 9223372036854775807);  select_221 = None
        slice_1011: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1010, 2, 0, 257);  slice_1010 = None
        copy_103: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1013, slice_1011);  slice_1013 = slice_1011 = None
        slice_scatter_189: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_225, copy_103, 2, 256, 9223372036854775807);  select_225 = copy_103 = None
        select_scatter_34: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_188, slice_scatter_189, 1, -1);  slice_scatter_188 = slice_scatter_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1022: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_34, 1, 1, 9223372036854775807)
        slice_1020: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_34, 1, 1, 9223372036854775807)
        slice_1021: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1020, 3, 0, 256);  slice_1020 = None
        slice_1015: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_868, 2, -257, -1)
        slice_1016: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1015, 3, 257, 9223372036854775807);  slice_1015 = None
        copy_104: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1021, slice_1016);  slice_1021 = slice_1016 = None
        slice_scatter_190: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1022, copy_104, 3, 0, 256);  slice_1022 = copy_104 = None
        slice_scatter_191: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_34, slice_scatter_190, 1, 1, 9223372036854775807);  select_scatter_34 = slice_scatter_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_232: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_191, 1, 0)
        slice_1032: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_232, 1, 1, 256)
        select_231: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_191, 1, 0)
        slice_1030: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_231, 1, 1, 256);  select_231 = None
        slice_1031: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1030, 2, 1, 256);  slice_1030 = None
        select_227: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_868, 1, 0);  view_868 = None
        slice_1025: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_227, 1, 0, 255);  select_227 = None
        slice_1026: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1025, 2, -255, 9223372036854775807);  slice_1025 = None
        copy_105: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1031, slice_1026);  slice_1031 = slice_1026 = None
        slice_scatter_192: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1032, copy_105, 2, 1, 256);  slice_1032 = copy_105 = None
        slice_scatter_193: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_232, slice_scatter_192, 1, 1, 256);  select_232 = slice_scatter_192 = None
        select_scatter_35: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_191, slice_scatter_193, 1, 0);  slice_scatter_191 = slice_scatter_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_876: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_35, [8, 1, 1024, 513])
        permute_789: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_876, [0, 2, 1, 3]);  view_876 = None
        slice_1045: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_789, 1, 0, 256)
        view_875: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_35, [8, 1, 1024, 513])
        permute_788: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_875, [0, 2, 1, 3]);  view_875 = None
        slice_1043: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_788, 1, 0, 256);  permute_788 = None
        slice_1044: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1043, 3, 0, 257);  slice_1043 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_34: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_181: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_34, -2);  iota_34 = None
        iota_35: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_182: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
        sub_67: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_181, unsqueeze_182);  unsqueeze_181 = unsqueeze_182 = None
        le_17: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_67, 0);  sub_67 = None
        full_78: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_34: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_68: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_17, full_78, scalar_tensor_34);  le_17 = full_78 = scalar_tensor_34 = None
        rev_34: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_68, [0]);  where_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_183: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_34, 0);  rev_34 = None
        unsqueeze_184: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_34: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_184, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_259: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_34, torch.bool);  expand_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_33: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_873: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_35, [8, 1, 1024, 513]);  select_scatter_35 = None
        permute_785: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_873, [0, 2, 1, 3]);  view_873 = None
        slice_1038: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_785, 1, 0, 256);  permute_785 = None
        slice_1039: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1038, 3, 0, 257);  slice_1038 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_69: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_259, full_default_33, slice_1039);  convert_element_type_259 = full_default_33 = slice_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_106: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1044, where_69);  slice_1044 = where_69 = None
        slice_scatter_194: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1045, copy_106, 3, 0, 257);  slice_1045 = copy_106 = None
        slice_scatter_195: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_789, slice_scatter_194, 1, 0, 256);  permute_789 = slice_scatter_194 = None
        permute_790: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_195, [0, 2, 1, 3]);  slice_scatter_195 = None
        view_877: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_790, [8, 4, 256, 513]);  permute_790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_884: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_877, [8, 1, 1024, 513])
        permute_798: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_884, [0, 2, 1, 3]);  view_884 = None
        slice_1058: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_798, 1, -256, 9223372036854775807)
        view_883: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_877, [8, 1, 1024, 513])
        permute_797: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_883, [0, 2, 1, 3]);  view_883 = None
        slice_1056: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_797, 1, -256, 9223372036854775807);  permute_797 = None
        slice_1057: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1056, 3, -257, 9223372036854775807);  slice_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_35: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_184, [1, 3]);  unsqueeze_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_35: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_35, [8, 256, 1, 257]);  rev_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_260: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_35, torch.bool);  expand_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_34: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_881: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_877, [8, 1, 1024, 513]);  view_877 = None
        permute_794: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_881, [0, 2, 1, 3]);  view_881 = None
        slice_1051: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_794, 1, -256, 9223372036854775807);  permute_794 = None
        slice_1052: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1051, 3, -257, 9223372036854775807);  slice_1051 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_70: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_260, full_default_34, slice_1052);  convert_element_type_260 = full_default_34 = slice_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_107: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1057, where_70);  slice_1057 = where_70 = None
        slice_scatter_196: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1058, copy_107, 3, -257, 9223372036854775807);  slice_1058 = copy_107 = None
        slice_scatter_197: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_798, slice_scatter_196, 1, -256, 9223372036854775807);  permute_798 = slice_scatter_196 = None
        permute_799: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_197, [0, 2, 1, 3]);  slice_scatter_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_802: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_799, [0, 2, 1, 3]);  permute_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_125: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_801, permute_802);  permute_801 = permute_802 = None
        permute_803: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_125, [0, 2, 1, 3]);  add_125 = None
        permute_804: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_803, [0, 2, 1, 3]);  permute_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_261: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_804, torch.float32);  permute_804 = None
        clone_145: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_261, memory_format = torch.contiguous_format);  convert_element_type_261 = None
        amax_8: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_145, [-1], True)
        sub_68: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_145, amax_8);  clone_145 = amax_8 = None
        exp_8: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
        sum_9: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_87: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_71: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_186, full_default_35, div_87);  unsqueeze_186 = full_default_35 = div_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_262: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_71, torch.bfloat16);  where_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_806: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_262, [0, 2, 1, 3]);  convert_element_type_262 = None
        clone_147: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_806, memory_format = torch.contiguous_format);  permute_806 = None
        view_893: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_147, [96, 4, 256, 513]);  clone_147 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_35: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_893, [0, 257], 0.0);  view_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_895: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_35, [96, 4, -1]);  constant_pad_nd_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1061: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_895, 2, 0, -256);  view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_896: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_1061, [96, 4, 256, 769]);  slice_1061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1062: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_896, 3, 0, -1);  view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_187: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_1062, 4);  slice_1062 = None
        view_897: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_187, [384, 256, 768]);  unsqueeze_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_138: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_728, memory_format = torch.contiguous_format);  permute_728 = None
        view_812: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [8192, 768]);  clone_138 = None
        permute_731: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_34: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_812, permute_731);  view_812 = permute_731 = None
        view_813: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [1024, 8, 768]);  mm_34 = None
        add_122: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_813, arg136_1);  view_813 = arg136_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_892: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_122, [1024, 8, 12, 64]);  add_122 = None
        permute_805: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_892, [1, 0, 2, 3]);  view_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_807: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_805, [0, 2, 1, 3]);  permute_805 = None
        view_894: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_807, [96, 1024, 64]);  permute_807 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_34: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_894, [0, 0, 256, 256], -1.0);  view_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_80: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_34, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_188: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_80, 4);  as_strided_80 = None
        clone_148: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_188, memory_format = torch.contiguous_format);  unsqueeze_188 = None
        view_898: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_148, [384, 768, 64]);  clone_148 = None
        bmm_17: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_897, view_898);  view_897 = view_898 = None
        view_899: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [96, 4, 256, 1, 64]);  bmm_17 = None
        permute_812: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_899, [0, 1, 2, 4, 3]);  view_899 = None
        view_900: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_812, [96, 4, 256, 64]);  permute_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_901: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_900, [8, 12, 1024, 64]);  view_900 = None
        permute_813: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_901, [0, 2, 1, 3]);  view_901 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_814: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_813, [1, 0, 2, 3]);  permute_813 = None
        clone_149: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_814, memory_format = torch.contiguous_format);  permute_814 = None
        view_902: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_149, [1024, 8, 768]);  clone_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_815: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_902, [1, 0, 2]);  view_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_150: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_815, memory_format = torch.contiguous_format);  permute_815 = None
        view_903: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_150, [8192, 768]);  clone_150 = None
        permute_816: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_35: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_903, permute_816);  view_903 = permute_816 = None
        view_904: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [8, 1024, 768]);  mm_35 = None
        add_127: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_904, arg138_1);  view_904 = arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_128: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_127, convert_element_type_247);  add_127 = convert_element_type_247 = None
        convert_element_type_267: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_128, torch.float32);  add_128 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_267, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_70: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_267, getitem_33);  convert_element_type_267 = getitem_33 = None
        add_129: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_65: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_16);  sub_70 = rsqrt_16 = None
        mul_66: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, arg139_1);  mul_65 = arg139_1 = None
        add_130: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_66, arg140_1);  mul_66 = arg140_1 = None
        convert_element_type_268: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_130, torch.bfloat16);  add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_905: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_268, [8192, 768])
        permute_817: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_16: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg142_1, view_905, permute_817);  arg142_1 = view_905 = permute_817 = None
        view_906: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [8, 1024, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_272: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_906, torch.float32);  view_906 = None
        mul_67: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_272, 0.5)
        mul_68: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_272, 0.7071067811865476);  convert_element_type_272 = None
        erf_8: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_131: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_69: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, add_131);  mul_67 = add_131 = None
        convert_element_type_273: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_907: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_273, [8192, 3072]);  convert_element_type_273 = None
        permute_818: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_17: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg144_1, view_907, permute_818);  arg144_1 = view_907 = permute_818 = None
        view_908: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [8, 1024, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_132: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_908, convert_element_type_268);  view_908 = convert_element_type_268 = None
        convert_element_type_277: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_132, torch.float32);  add_132 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_277, [2], correction = 0, keepdim = True)
        getitem_34: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_88: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_89: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_206: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_207: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
        full_default_39: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_81: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1067: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_81, 1, 0, -1)
        slice_1065: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_81, 1, 0, -1)
        slice_1066: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1065, 3, 256, 9223372036854775807);  slice_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_71: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_277, getitem_35);  convert_element_type_277 = getitem_35 = None
        add_133: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_70: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_17);  sub_71 = rsqrt_17 = None
        mul_71: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, arg145_1);  mul_70 = arg145_1 = None
        add_134: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, arg146_1);  mul_71 = arg146_1 = None
        convert_element_type_278: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_819: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_278, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_153: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_819, memory_format = torch.contiguous_format)
        view_909: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [8192, 768]);  clone_153 = None
        permute_820: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_36: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_909, permute_820);  view_909 = permute_820 = None
        view_910: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [1024, 8, 768]);  mm_36 = None
        add_135: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_910, arg148_1);  view_910 = arg148_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_90: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_135, 8.0);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_938: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_90, [1024, 8, 12, 64]);  div_90 = None
        permute_844: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_938, [1, 0, 2, 3]);  view_938 = None
        permute_845: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_844, [0, 2, 1, 3]);  permute_844 = None
        view_939: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_845, [96, 1024, 64]);  permute_845 = None
        view_940: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_939, [96, 2, 512, 64]);  view_939 = None
        as_strided_86: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_940, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_940 = None
        unsqueeze_193: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_86, 4);  as_strided_86 = None
        clone_156: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_193, memory_format = torch.contiguous_format);  unsqueeze_193 = None
        view_941: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_156, [288, 512, 64]);  clone_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_154: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_819, memory_format = torch.contiguous_format)
        view_911: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [8192, 768]);  clone_154 = None
        permute_821: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_37: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_911, permute_821);  view_911 = permute_821 = None
        view_912: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [1024, 8, 768]);  mm_37 = None
        add_136: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_912, arg150_1);  view_912 = arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_917: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_136, [1024, 8, 12, 64]);  add_136 = None
        permute_824: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_917, [1, 0, 2, 3]);  view_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_829: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_824, [0, 2, 1, 3]);  permute_824 = None
        view_921: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_829, [96, 1024, 64]);  permute_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_928: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_921, [96, 2, 512, 64]);  view_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_82: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_928, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_191: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_82, 4);  as_strided_82 = None
        permute_839: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_191, [0, 1, 4, 2, 3]);  unsqueeze_191 = None
        permute_848: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_839, [0, 1, 4, 3, 2]);  permute_839 = None
        clone_157: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_848, memory_format = torch.contiguous_format);  permute_848 = None
        view_942: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_157, [288, 64, 512]);  clone_157 = None
        bmm_18: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_941, view_942);  view_941 = view_942 = None
        view_943: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [96, 3, 512, 1, 512]);  bmm_18 = None
        permute_849: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_943, [0, 1, 2, 4, 3]);  view_943 = None
        view_944: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_849, [96, 3, 512, 512]);  permute_849 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_36: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_944, [0, 0, 0, 1], 0.0);  view_944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_945: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_36, [96, 3, 512, 513]);  constant_pad_nd_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1063: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_945, 2, 0, 256)
        slice_1064: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1063, 3, 0, 257);  slice_1063 = None
        copy_108: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1066, slice_1064);  slice_1066 = slice_1064 = None
        slice_scatter_198: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1067, copy_108, 3, 256, 9223372036854775807);  slice_1067 = copy_108 = None
        slice_scatter_199: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_81, slice_scatter_198, 1, 0, -1);  full_81 = slice_scatter_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_238: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_199, 1, -1)
        select_237: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_199, 1, -1)
        slice_1073: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_237, 2, 256, 9223372036854775807);  select_237 = None
        select_234: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_945, 1, -1)
        slice_1070: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_234, 1, 256, 9223372036854775807);  select_234 = None
        slice_1071: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1070, 2, 0, 257);  slice_1070 = None
        copy_109: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1073, slice_1071);  slice_1073 = slice_1071 = None
        slice_scatter_200: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_238, copy_109, 2, 256, 9223372036854775807);  select_238 = copy_109 = None
        select_scatter_36: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_199, slice_scatter_200, 1, -1);  slice_scatter_199 = slice_scatter_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1082: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_36, 1, 1, 9223372036854775807)
        slice_1080: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_36, 1, 1, 9223372036854775807)
        slice_1081: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1080, 3, 0, 256);  slice_1080 = None
        slice_1075: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_945, 2, -257, -1)
        slice_1076: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1075, 3, 257, 9223372036854775807);  slice_1075 = None
        copy_110: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1081, slice_1076);  slice_1081 = slice_1076 = None
        slice_scatter_201: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1082, copy_110, 3, 0, 256);  slice_1082 = copy_110 = None
        slice_scatter_202: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_36, slice_scatter_201, 1, 1, 9223372036854775807);  select_scatter_36 = slice_scatter_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_245: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_202, 1, 0)
        slice_1092: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_245, 1, 1, 256)
        select_244: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_202, 1, 0)
        slice_1090: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_244, 1, 1, 256);  select_244 = None
        slice_1091: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1090, 2, 1, 256);  slice_1090 = None
        select_240: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_945, 1, 0);  view_945 = None
        slice_1085: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_240, 1, 0, 255);  select_240 = None
        slice_1086: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1085, 2, -255, 9223372036854775807);  slice_1085 = None
        copy_111: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1091, slice_1086);  slice_1091 = slice_1086 = None
        slice_scatter_203: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1092, copy_111, 2, 1, 256);  slice_1092 = copy_111 = None
        slice_scatter_204: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_245, slice_scatter_203, 1, 1, 256);  select_245 = slice_scatter_203 = None
        select_scatter_37: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_202, slice_scatter_204, 1, 0);  slice_scatter_202 = slice_scatter_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_953: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_37, [8, 12, 1024, 513])
        permute_857: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_953, [0, 2, 1, 3]);  view_953 = None
        slice_1105: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_857, 1, 0, 256)
        view_952: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_37, [8, 12, 1024, 513])
        permute_856: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_952, [0, 2, 1, 3]);  view_952 = None
        slice_1103: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_856, 1, 0, 256);  permute_856 = None
        slice_1104: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1103, 3, 0, 257);  slice_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_36: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_194: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_36, -2);  iota_36 = None
        iota_37: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_195: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
        sub_73: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_194, unsqueeze_195);  unsqueeze_194 = unsqueeze_195 = None
        le_18: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_73, 0);  sub_73 = None
        full_82: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_36: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_72: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_18, full_82, scalar_tensor_36);  le_18 = full_82 = scalar_tensor_36 = None
        rev_36: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_72, [0]);  where_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_196: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_36, 0);  rev_36 = None
        unsqueeze_197: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_36: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_197, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_287: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_36, torch.bool);  expand_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_83: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_854: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_83, [0, 2, 1, 3]);  full_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_950: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_37, [8, 12, 1024, 513]);  select_scatter_37 = None
        permute_853: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_950, [0, 2, 1, 3]);  view_950 = None
        slice_1098: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_853, 1, 0, 256);  permute_853 = None
        slice_1099: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1098, 3, 0, 257);  slice_1098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_73: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_287, permute_854, slice_1099);  convert_element_type_287 = permute_854 = slice_1099 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_112: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1104, where_73);  slice_1104 = where_73 = None
        slice_scatter_205: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1105, copy_112, 3, 0, 257);  slice_1105 = copy_112 = None
        slice_scatter_206: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_857, slice_scatter_205, 1, 0, 256);  permute_857 = slice_scatter_205 = None
        permute_858: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_206, [0, 2, 1, 3]);  slice_scatter_206 = None
        view_954: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_858, [96, 4, 256, 513]);  permute_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_961: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_954, [8, 12, 1024, 513])
        permute_866: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_961, [0, 2, 1, 3]);  view_961 = None
        slice_1118: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_866, 1, -256, 9223372036854775807)
        view_960: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_954, [8, 12, 1024, 513])
        permute_865: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_960, [0, 2, 1, 3]);  view_960 = None
        slice_1116: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_865, 1, -256, 9223372036854775807);  permute_865 = None
        slice_1117: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1116, 3, -257, 9223372036854775807);  slice_1116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_37: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_197, [1, 3]);  unsqueeze_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_37: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_37, [8, 256, 12, 257]);  rev_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_288: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_37, torch.bool);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_84: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_863: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_84, [0, 2, 1, 3]);  full_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_958: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_954, [8, 12, 1024, 513]);  view_954 = None
        permute_862: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_958, [0, 2, 1, 3]);  view_958 = None
        slice_1111: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_862, 1, -256, 9223372036854775807);  permute_862 = None
        slice_1112: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1111, 3, -257, 9223372036854775807);  slice_1111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_74: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_288, permute_863, slice_1112);  convert_element_type_288 = permute_863 = slice_1112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_113: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1117, where_74);  slice_1117 = where_74 = None
        slice_scatter_207: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1118, copy_113, 3, -257, 9223372036854775807);  slice_1118 = copy_113 = None
        slice_scatter_208: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_866, slice_scatter_207, 1, -256, 9223372036854775807);  permute_866 = slice_scatter_207 = None
        permute_867: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_208, [0, 2, 1, 3]);  slice_scatter_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_892: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_867, [0, 2, 1, 3]);  permute_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_86: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1125: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_86, 1, 0, -1)
        slice_1123: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_86, 1, 0, -1)
        slice_1124: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1123, 3, 256, 9223372036854775807);  slice_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_85: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_869: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_85, [0, 2, 1, 3]);  full_85 = None
        view_964: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_869, [8, 1024, 1]);  permute_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_966: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_964, [8, 2, 512, 1]);  view_964 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_87: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_966, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_200: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_87, 4);  as_strided_87 = None
        permute_871: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_200, [0, 1, 2, 4, 3]);  unsqueeze_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_36: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_9: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_198: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_9, 2);  ne_9 = None
        unsqueeze_199: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_289: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_199, torch.bfloat16)
        where_75: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_199, full_default_36, convert_element_type_289);  unsqueeze_199 = full_default_36 = convert_element_type_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_870: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_75, [0, 2, 1, 3]);  where_75 = None
        view_965: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_870, [8, 1024, 1]);  permute_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_967: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_965, [8, 2, 512, 1]);  view_965 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_88: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_967, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_967 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_201: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_88, 4);  as_strided_88 = None
        permute_872: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_201, [0, 1, 4, 2, 3]);  unsqueeze_201 = None
        mul_72: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_871, permute_872);  permute_871 = permute_872 = None
        view_968: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_72, [8, 3, 512, 512]);  mul_72 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_37: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_968, [0, 0, 0, 1], 0.0);  view_968 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_969: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_37, [8, 3, 512, 513]);  constant_pad_nd_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1121: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_969, 2, 0, 256)
        slice_1122: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1121, 3, 0, 257);  slice_1121 = None
        copy_114: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1124, slice_1122);  slice_1124 = slice_1122 = None
        slice_scatter_209: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1125, copy_114, 3, 256, 9223372036854775807);  slice_1125 = copy_114 = None
        slice_scatter_210: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_86, slice_scatter_209, 1, 0, -1);  full_86 = slice_scatter_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_251: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_210, 1, -1)
        select_250: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_210, 1, -1)
        slice_1131: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_250, 2, 256, 9223372036854775807);  select_250 = None
        select_247: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_969, 1, -1)
        slice_1128: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_247, 1, 256, 9223372036854775807);  select_247 = None
        slice_1129: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1128, 2, 0, 257);  slice_1128 = None
        copy_115: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1131, slice_1129);  slice_1131 = slice_1129 = None
        slice_scatter_211: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_251, copy_115, 2, 256, 9223372036854775807);  select_251 = copy_115 = None
        select_scatter_38: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_210, slice_scatter_211, 1, -1);  slice_scatter_210 = slice_scatter_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1140: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_38, 1, 1, 9223372036854775807)
        slice_1138: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_38, 1, 1, 9223372036854775807)
        slice_1139: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1138, 3, 0, 256);  slice_1138 = None
        slice_1133: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_969, 2, -257, -1)
        slice_1134: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1133, 3, 257, 9223372036854775807);  slice_1133 = None
        copy_116: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1139, slice_1134);  slice_1139 = slice_1134 = None
        slice_scatter_212: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1140, copy_116, 3, 0, 256);  slice_1140 = copy_116 = None
        slice_scatter_213: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_38, slice_scatter_212, 1, 1, 9223372036854775807);  select_scatter_38 = slice_scatter_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_258: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_213, 1, 0)
        slice_1150: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_258, 1, 1, 256)
        select_257: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_213, 1, 0)
        slice_1148: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_257, 1, 1, 256);  select_257 = None
        slice_1149: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1148, 2, 1, 256);  slice_1148 = None
        select_253: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_969, 1, 0);  view_969 = None
        slice_1143: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_253, 1, 0, 255);  select_253 = None
        slice_1144: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1143, 2, -255, 9223372036854775807);  slice_1143 = None
        copy_117: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1149, slice_1144);  slice_1149 = slice_1144 = None
        slice_scatter_214: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1150, copy_117, 2, 1, 256);  slice_1150 = copy_117 = None
        slice_scatter_215: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_258, slice_scatter_214, 1, 1, 256);  select_258 = slice_scatter_214 = None
        select_scatter_39: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_213, slice_scatter_215, 1, 0);  slice_scatter_213 = slice_scatter_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_977: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_39, [8, 1, 1024, 513])
        permute_880: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_977, [0, 2, 1, 3]);  view_977 = None
        slice_1163: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_880, 1, 0, 256)
        view_976: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_39, [8, 1, 1024, 513])
        permute_879: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_976, [0, 2, 1, 3]);  view_976 = None
        slice_1161: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_879, 1, 0, 256);  permute_879 = None
        slice_1162: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1161, 3, 0, 257);  slice_1161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_38: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_202: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_38, -2);  iota_38 = None
        iota_39: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_203: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
        sub_75: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_202, unsqueeze_203);  unsqueeze_202 = unsqueeze_203 = None
        le_19: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_75, 0);  sub_75 = None
        full_87: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_38: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_76: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_19, full_87, scalar_tensor_38);  le_19 = full_87 = scalar_tensor_38 = None
        rev_38: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_76, [0]);  where_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_204: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_38, 0);  rev_38 = None
        unsqueeze_205: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_38: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_205, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_290: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_38, torch.bool);  expand_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_37: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_974: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_39, [8, 1, 1024, 513]);  select_scatter_39 = None
        permute_876: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_974, [0, 2, 1, 3]);  view_974 = None
        slice_1156: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_876, 1, 0, 256);  permute_876 = None
        slice_1157: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1156, 3, 0, 257);  slice_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_77: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_290, full_default_37, slice_1157);  convert_element_type_290 = full_default_37 = slice_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_118: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1162, where_77);  slice_1162 = where_77 = None
        slice_scatter_216: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1163, copy_118, 3, 0, 257);  slice_1163 = copy_118 = None
        slice_scatter_217: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_880, slice_scatter_216, 1, 0, 256);  permute_880 = slice_scatter_216 = None
        permute_881: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_217, [0, 2, 1, 3]);  slice_scatter_217 = None
        view_978: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_881, [8, 4, 256, 513]);  permute_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_985: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_978, [8, 1, 1024, 513])
        permute_889: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_985, [0, 2, 1, 3]);  view_985 = None
        slice_1176: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_889, 1, -256, 9223372036854775807)
        view_984: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_978, [8, 1, 1024, 513])
        permute_888: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_984, [0, 2, 1, 3]);  view_984 = None
        slice_1174: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_888, 1, -256, 9223372036854775807);  permute_888 = None
        slice_1175: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1174, 3, -257, 9223372036854775807);  slice_1174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_39: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_205, [1, 3]);  unsqueeze_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_39: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_39, [8, 256, 1, 257]);  rev_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_291: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_39, torch.bool);  expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_38: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_982: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_978, [8, 1, 1024, 513]);  view_978 = None
        permute_885: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_982, [0, 2, 1, 3]);  view_982 = None
        slice_1169: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_885, 1, -256, 9223372036854775807);  permute_885 = None
        slice_1170: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1169, 3, -257, 9223372036854775807);  slice_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_78: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_291, full_default_38, slice_1170);  convert_element_type_291 = full_default_38 = slice_1170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_119: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1175, where_78);  slice_1175 = where_78 = None
        slice_scatter_218: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1176, copy_119, 3, -257, 9223372036854775807);  slice_1176 = copy_119 = None
        slice_scatter_219: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_889, slice_scatter_218, 1, -256, 9223372036854775807);  permute_889 = slice_scatter_218 = None
        permute_890: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_219, [0, 2, 1, 3]);  slice_scatter_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_893: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_890, [0, 2, 1, 3]);  permute_890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_140: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_892, permute_893);  permute_892 = permute_893 = None
        permute_894: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_140, [0, 2, 1, 3]);  add_140 = None
        permute_895: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_894, [0, 2, 1, 3]);  permute_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_292: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_895, torch.float32);  permute_895 = None
        clone_162: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_292, memory_format = torch.contiguous_format);  convert_element_type_292 = None
        amax_9: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_162, [-1], True)
        sub_76: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_162, amax_9);  clone_162 = amax_9 = None
        exp_9: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_10: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_97: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_79: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_207, full_default_39, div_97);  unsqueeze_207 = full_default_39 = div_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_293: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_79, torch.bfloat16);  where_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_897: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_293, [0, 2, 1, 3]);  convert_element_type_293 = None
        clone_164: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_897, memory_format = torch.contiguous_format);  permute_897 = None
        view_994: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_164, [96, 4, 256, 513]);  clone_164 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_39: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_994, [0, 257], 0.0);  view_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_996: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_39, [96, 4, -1]);  constant_pad_nd_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1179: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_996, 2, 0, -256);  view_996 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_997: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_1179, [96, 4, 256, 769]);  slice_1179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1180: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_997, 3, 0, -1);  view_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_208: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_1180, 4);  slice_1180 = None
        view_998: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_208, [384, 256, 768]);  unsqueeze_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_155: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_819, memory_format = torch.contiguous_format);  permute_819 = None
        view_913: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_155, [8192, 768]);  clone_155 = None
        permute_822: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_38: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_913, permute_822);  view_913 = permute_822 = None
        view_914: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [1024, 8, 768]);  mm_38 = None
        add_137: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_914, arg152_1);  view_914 = arg152_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_993: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_137, [1024, 8, 12, 64]);  add_137 = None
        permute_896: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_993, [1, 0, 2, 3]);  view_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_898: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_896, [0, 2, 1, 3]);  permute_896 = None
        view_995: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_898, [96, 1024, 64]);  permute_898 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_38: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_995, [0, 0, 256, 256], -1.0);  view_995 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_89: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_38, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_209: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_89, 4);  as_strided_89 = None
        clone_165: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_209, memory_format = torch.contiguous_format);  unsqueeze_209 = None
        view_999: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_165, [384, 768, 64]);  clone_165 = None
        bmm_19: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_998, view_999);  view_998 = view_999 = None
        view_1000: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [96, 4, 256, 1, 64]);  bmm_19 = None
        permute_903: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_1000, [0, 1, 2, 4, 3]);  view_1000 = None
        view_1001: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_903, [96, 4, 256, 64]);  permute_903 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1002: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1001, [8, 12, 1024, 64]);  view_1001 = None
        permute_904: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_1002, [0, 2, 1, 3]);  view_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_905: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_904, [1, 0, 2, 3]);  permute_904 = None
        clone_166: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_905, memory_format = torch.contiguous_format);  permute_905 = None
        view_1003: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_166, [1024, 8, 768]);  clone_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_906: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_1003, [1, 0, 2]);  view_1003 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_167: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_906, memory_format = torch.contiguous_format);  permute_906 = None
        view_1004: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_167, [8192, 768]);  clone_167 = None
        permute_907: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        mm_39: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1004, permute_907);  view_1004 = permute_907 = None
        view_1005: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [8, 1024, 768]);  mm_39 = None
        add_142: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1005, arg154_1);  view_1005 = arg154_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_143: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_142, convert_element_type_278);  add_142 = convert_element_type_278 = None
        convert_element_type_298: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.float32);  add_143 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_298, [2], correction = 0, keepdim = True)
        getitem_36: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_78: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_298, getitem_37);  convert_element_type_298 = getitem_37 = None
        add_144: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_73: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_18);  sub_78 = rsqrt_18 = None
        mul_74: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, arg155_1);  mul_73 = arg155_1 = None
        add_145: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, arg156_1);  mul_74 = arg156_1 = None
        convert_element_type_299: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1006: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_299, [8192, 768])
        permute_908: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_18: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg158_1, view_1006, permute_908);  arg158_1 = view_1006 = permute_908 = None
        view_1007: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [8, 1024, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_303: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1007, torch.float32);  view_1007 = None
        mul_75: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_303, 0.5)
        mul_76: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_303, 0.7071067811865476);  convert_element_type_303 = None
        erf_9: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_146: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_77: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, add_146);  mul_75 = add_146 = None
        convert_element_type_304: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1008: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_304, [8192, 3072]);  convert_element_type_304 = None
        permute_909: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_19: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, view_1008, permute_909);  arg160_1 = view_1008 = permute_909 = None
        view_1009: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [8, 1024, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_147: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1009, convert_element_type_299);  view_1009 = convert_element_type_299 = None
        convert_element_type_308: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.float32);  add_147 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_308, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_97: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_98: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_227: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_228: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        full_default_43: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_90: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1185: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_90, 1, 0, -1)
        slice_1183: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_90, 1, 0, -1)
        slice_1184: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1183, 3, 256, 9223372036854775807);  slice_1183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_79: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_308, getitem_39);  convert_element_type_308 = getitem_39 = None
        add_148: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_78: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_19);  sub_79 = rsqrt_19 = None
        mul_79: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, arg161_1);  mul_78 = arg161_1 = None
        add_149: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, arg162_1);  mul_79 = arg162_1 = None
        convert_element_type_309: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_910: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_309, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_170: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_910, memory_format = torch.contiguous_format)
        view_1010: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [8192, 768]);  clone_170 = None
        permute_911: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_40: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1010, permute_911);  view_1010 = permute_911 = None
        view_1011: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [1024, 8, 768]);  mm_40 = None
        add_150: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1011, arg164_1);  view_1011 = arg164_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_100: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_150, 8.0);  add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1039: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_100, [1024, 8, 12, 64]);  div_100 = None
        permute_935: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1039, [1, 0, 2, 3]);  view_1039 = None
        permute_936: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_935, [0, 2, 1, 3]);  permute_935 = None
        view_1040: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_936, [96, 1024, 64]);  permute_936 = None
        view_1041: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1040, [96, 2, 512, 64]);  view_1040 = None
        as_strided_95: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_1041, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_1041 = None
        unsqueeze_214: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_95, 4);  as_strided_95 = None
        clone_173: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_214, memory_format = torch.contiguous_format);  unsqueeze_214 = None
        view_1042: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_173, [288, 512, 64]);  clone_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_171: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_910, memory_format = torch.contiguous_format)
        view_1012: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [8192, 768]);  clone_171 = None
        permute_912: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        mm_41: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1012, permute_912);  view_1012 = permute_912 = None
        view_1013: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [1024, 8, 768]);  mm_41 = None
        add_151: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1013, arg166_1);  view_1013 = arg166_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1018: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_151, [1024, 8, 12, 64]);  add_151 = None
        permute_915: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1018, [1, 0, 2, 3]);  view_1018 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_920: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_915, [0, 2, 1, 3]);  permute_915 = None
        view_1022: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_920, [96, 1024, 64]);  permute_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1029: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1022, [96, 2, 512, 64]);  view_1022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_91: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_1029, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_1029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_212: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_91, 4);  as_strided_91 = None
        permute_930: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_212, [0, 1, 4, 2, 3]);  unsqueeze_212 = None
        permute_939: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_930, [0, 1, 4, 3, 2]);  permute_930 = None
        clone_174: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_939, memory_format = torch.contiguous_format);  permute_939 = None
        view_1043: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_174, [288, 64, 512]);  clone_174 = None
        bmm_20: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_1042, view_1043);  view_1042 = view_1043 = None
        view_1044: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [96, 3, 512, 1, 512]);  bmm_20 = None
        permute_940: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_1044, [0, 1, 2, 4, 3]);  view_1044 = None
        view_1045: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_940, [96, 3, 512, 512]);  permute_940 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_40: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1045, [0, 0, 0, 1], 0.0);  view_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1046: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_40, [96, 3, 512, 513]);  constant_pad_nd_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1181: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1046, 2, 0, 256)
        slice_1182: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1181, 3, 0, 257);  slice_1181 = None
        copy_120: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1184, slice_1182);  slice_1184 = slice_1182 = None
        slice_scatter_220: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1185, copy_120, 3, 256, 9223372036854775807);  slice_1185 = copy_120 = None
        slice_scatter_221: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_90, slice_scatter_220, 1, 0, -1);  full_90 = slice_scatter_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_264: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_221, 1, -1)
        select_263: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_221, 1, -1)
        slice_1191: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_263, 2, 256, 9223372036854775807);  select_263 = None
        select_260: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_1046, 1, -1)
        slice_1188: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_260, 1, 256, 9223372036854775807);  select_260 = None
        slice_1189: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1188, 2, 0, 257);  slice_1188 = None
        copy_121: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1191, slice_1189);  slice_1191 = slice_1189 = None
        slice_scatter_222: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_264, copy_121, 2, 256, 9223372036854775807);  select_264 = copy_121 = None
        select_scatter_40: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_221, slice_scatter_222, 1, -1);  slice_scatter_221 = slice_scatter_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1200: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_40, 1, 1, 9223372036854775807)
        slice_1198: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_40, 1, 1, 9223372036854775807)
        slice_1199: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1198, 3, 0, 256);  slice_1198 = None
        slice_1193: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1046, 2, -257, -1)
        slice_1194: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1193, 3, 257, 9223372036854775807);  slice_1193 = None
        copy_122: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1199, slice_1194);  slice_1199 = slice_1194 = None
        slice_scatter_223: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1200, copy_122, 3, 0, 256);  slice_1200 = copy_122 = None
        slice_scatter_224: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_40, slice_scatter_223, 1, 1, 9223372036854775807);  select_scatter_40 = slice_scatter_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_271: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_224, 1, 0)
        slice_1210: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_271, 1, 1, 256)
        select_270: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_224, 1, 0)
        slice_1208: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_270, 1, 1, 256);  select_270 = None
        slice_1209: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1208, 2, 1, 256);  slice_1208 = None
        select_266: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_1046, 1, 0);  view_1046 = None
        slice_1203: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_266, 1, 0, 255);  select_266 = None
        slice_1204: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1203, 2, -255, 9223372036854775807);  slice_1203 = None
        copy_123: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1209, slice_1204);  slice_1209 = slice_1204 = None
        slice_scatter_225: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1210, copy_123, 2, 1, 256);  slice_1210 = copy_123 = None
        slice_scatter_226: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_271, slice_scatter_225, 1, 1, 256);  select_271 = slice_scatter_225 = None
        select_scatter_41: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_224, slice_scatter_226, 1, 0);  slice_scatter_224 = slice_scatter_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1054: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_41, [8, 12, 1024, 513])
        permute_948: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1054, [0, 2, 1, 3]);  view_1054 = None
        slice_1223: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_948, 1, 0, 256)
        view_1053: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_41, [8, 12, 1024, 513])
        permute_947: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1053, [0, 2, 1, 3]);  view_1053 = None
        slice_1221: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_947, 1, 0, 256);  permute_947 = None
        slice_1222: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1221, 3, 0, 257);  slice_1221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_40: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_215: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_40, -2);  iota_40 = None
        iota_41: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_216: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
        sub_81: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_215, unsqueeze_216);  unsqueeze_215 = unsqueeze_216 = None
        le_20: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_81, 0);  sub_81 = None
        full_91: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_40: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_80: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_20, full_91, scalar_tensor_40);  le_20 = full_91 = scalar_tensor_40 = None
        rev_40: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_80, [0]);  where_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_217: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_40, 0);  rev_40 = None
        unsqueeze_218: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_40: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_218, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_318: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_40, torch.bool);  expand_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_92: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_945: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_92, [0, 2, 1, 3]);  full_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1051: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_41, [8, 12, 1024, 513]);  select_scatter_41 = None
        permute_944: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1051, [0, 2, 1, 3]);  view_1051 = None
        slice_1216: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_944, 1, 0, 256);  permute_944 = None
        slice_1217: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1216, 3, 0, 257);  slice_1216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_81: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_318, permute_945, slice_1217);  convert_element_type_318 = permute_945 = slice_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_124: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1222, where_81);  slice_1222 = where_81 = None
        slice_scatter_227: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1223, copy_124, 3, 0, 257);  slice_1223 = copy_124 = None
        slice_scatter_228: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_948, slice_scatter_227, 1, 0, 256);  permute_948 = slice_scatter_227 = None
        permute_949: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_228, [0, 2, 1, 3]);  slice_scatter_228 = None
        view_1055: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_949, [96, 4, 256, 513]);  permute_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1062: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1055, [8, 12, 1024, 513])
        permute_957: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1062, [0, 2, 1, 3]);  view_1062 = None
        slice_1236: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_957, 1, -256, 9223372036854775807)
        view_1061: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1055, [8, 12, 1024, 513])
        permute_956: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1061, [0, 2, 1, 3]);  view_1061 = None
        slice_1234: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_956, 1, -256, 9223372036854775807);  permute_956 = None
        slice_1235: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1234, 3, -257, 9223372036854775807);  slice_1234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_41: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_218, [1, 3]);  unsqueeze_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_41: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_41, [8, 256, 12, 257]);  rev_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_319: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_41, torch.bool);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_93: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_954: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_93, [0, 2, 1, 3]);  full_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1059: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1055, [8, 12, 1024, 513]);  view_1055 = None
        permute_953: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1059, [0, 2, 1, 3]);  view_1059 = None
        slice_1229: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_953, 1, -256, 9223372036854775807);  permute_953 = None
        slice_1230: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1229, 3, -257, 9223372036854775807);  slice_1229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_82: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_319, permute_954, slice_1230);  convert_element_type_319 = permute_954 = slice_1230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_125: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1235, where_82);  slice_1235 = where_82 = None
        slice_scatter_229: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1236, copy_125, 3, -257, 9223372036854775807);  slice_1236 = copy_125 = None
        slice_scatter_230: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_957, slice_scatter_229, 1, -256, 9223372036854775807);  permute_957 = slice_scatter_229 = None
        permute_958: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_230, [0, 2, 1, 3]);  slice_scatter_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_983: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_958, [0, 2, 1, 3]);  permute_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_95: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1243: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_95, 1, 0, -1)
        slice_1241: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_95, 1, 0, -1)
        slice_1242: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1241, 3, 256, 9223372036854775807);  slice_1241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_94: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_960: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_94, [0, 2, 1, 3]);  full_94 = None
        view_1065: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_960, [8, 1024, 1]);  permute_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1067: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1065, [8, 2, 512, 1]);  view_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_96: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_1067, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_1067 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_221: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_96, 4);  as_strided_96 = None
        permute_962: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_221, [0, 1, 2, 4, 3]);  unsqueeze_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_40: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_10: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_219: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_10, 2);  ne_10 = None
        unsqueeze_220: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_320: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_220, torch.bfloat16)
        where_83: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_220, full_default_40, convert_element_type_320);  unsqueeze_220 = full_default_40 = convert_element_type_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_961: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_83, [0, 2, 1, 3]);  where_83 = None
        view_1066: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_961, [8, 1024, 1]);  permute_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1068: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1066, [8, 2, 512, 1]);  view_1066 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_97: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_1068, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_222: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_97, 4);  as_strided_97 = None
        permute_963: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_222, [0, 1, 4, 2, 3]);  unsqueeze_222 = None
        mul_80: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_962, permute_963);  permute_962 = permute_963 = None
        view_1069: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_80, [8, 3, 512, 512]);  mul_80 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_41: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1069, [0, 0, 0, 1], 0.0);  view_1069 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1070: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_41, [8, 3, 512, 513]);  constant_pad_nd_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1239: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1070, 2, 0, 256)
        slice_1240: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1239, 3, 0, 257);  slice_1239 = None
        copy_126: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1242, slice_1240);  slice_1242 = slice_1240 = None
        slice_scatter_231: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1243, copy_126, 3, 256, 9223372036854775807);  slice_1243 = copy_126 = None
        slice_scatter_232: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_95, slice_scatter_231, 1, 0, -1);  full_95 = slice_scatter_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_277: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_232, 1, -1)
        select_276: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_232, 1, -1)
        slice_1249: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_276, 2, 256, 9223372036854775807);  select_276 = None
        select_273: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_1070, 1, -1)
        slice_1246: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_273, 1, 256, 9223372036854775807);  select_273 = None
        slice_1247: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1246, 2, 0, 257);  slice_1246 = None
        copy_127: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1249, slice_1247);  slice_1249 = slice_1247 = None
        slice_scatter_233: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_277, copy_127, 2, 256, 9223372036854775807);  select_277 = copy_127 = None
        select_scatter_42: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_232, slice_scatter_233, 1, -1);  slice_scatter_232 = slice_scatter_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1258: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_42, 1, 1, 9223372036854775807)
        slice_1256: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_42, 1, 1, 9223372036854775807)
        slice_1257: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1256, 3, 0, 256);  slice_1256 = None
        slice_1251: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1070, 2, -257, -1)
        slice_1252: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1251, 3, 257, 9223372036854775807);  slice_1251 = None
        copy_128: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1257, slice_1252);  slice_1257 = slice_1252 = None
        slice_scatter_234: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1258, copy_128, 3, 0, 256);  slice_1258 = copy_128 = None
        slice_scatter_235: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_42, slice_scatter_234, 1, 1, 9223372036854775807);  select_scatter_42 = slice_scatter_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_284: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_235, 1, 0)
        slice_1268: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_284, 1, 1, 256)
        select_283: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_235, 1, 0)
        slice_1266: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_283, 1, 1, 256);  select_283 = None
        slice_1267: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1266, 2, 1, 256);  slice_1266 = None
        select_279: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_1070, 1, 0);  view_1070 = None
        slice_1261: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_279, 1, 0, 255);  select_279 = None
        slice_1262: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1261, 2, -255, 9223372036854775807);  slice_1261 = None
        copy_129: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1267, slice_1262);  slice_1267 = slice_1262 = None
        slice_scatter_236: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1268, copy_129, 2, 1, 256);  slice_1268 = copy_129 = None
        slice_scatter_237: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_284, slice_scatter_236, 1, 1, 256);  select_284 = slice_scatter_236 = None
        select_scatter_43: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_235, slice_scatter_237, 1, 0);  slice_scatter_235 = slice_scatter_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1078: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_43, [8, 1, 1024, 513])
        permute_971: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1078, [0, 2, 1, 3]);  view_1078 = None
        slice_1281: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_971, 1, 0, 256)
        view_1077: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_43, [8, 1, 1024, 513])
        permute_970: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1077, [0, 2, 1, 3]);  view_1077 = None
        slice_1279: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_970, 1, 0, 256);  permute_970 = None
        slice_1280: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1279, 3, 0, 257);  slice_1279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_42: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_223: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_42, -2);  iota_42 = None
        iota_43: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_224: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
        sub_83: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_223, unsqueeze_224);  unsqueeze_223 = unsqueeze_224 = None
        le_21: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_83, 0);  sub_83 = None
        full_96: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_42: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_84: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_21, full_96, scalar_tensor_42);  le_21 = full_96 = scalar_tensor_42 = None
        rev_42: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_84, [0]);  where_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_225: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_42, 0);  rev_42 = None
        unsqueeze_226: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_42: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_226, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_321: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_42, torch.bool);  expand_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_41: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1075: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_43, [8, 1, 1024, 513]);  select_scatter_43 = None
        permute_967: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1075, [0, 2, 1, 3]);  view_1075 = None
        slice_1274: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_967, 1, 0, 256);  permute_967 = None
        slice_1275: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1274, 3, 0, 257);  slice_1274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_85: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_321, full_default_41, slice_1275);  convert_element_type_321 = full_default_41 = slice_1275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_130: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1280, where_85);  slice_1280 = where_85 = None
        slice_scatter_238: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1281, copy_130, 3, 0, 257);  slice_1281 = copy_130 = None
        slice_scatter_239: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_971, slice_scatter_238, 1, 0, 256);  permute_971 = slice_scatter_238 = None
        permute_972: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_239, [0, 2, 1, 3]);  slice_scatter_239 = None
        view_1079: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_972, [8, 4, 256, 513]);  permute_972 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1086: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1079, [8, 1, 1024, 513])
        permute_980: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1086, [0, 2, 1, 3]);  view_1086 = None
        slice_1294: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_980, 1, -256, 9223372036854775807)
        view_1085: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1079, [8, 1, 1024, 513])
        permute_979: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1085, [0, 2, 1, 3]);  view_1085 = None
        slice_1292: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_979, 1, -256, 9223372036854775807);  permute_979 = None
        slice_1293: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1292, 3, -257, 9223372036854775807);  slice_1292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_43: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_226, [1, 3]);  unsqueeze_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_43: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_43, [8, 256, 1, 257]);  rev_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_322: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_43, torch.bool);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_42: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1083: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1079, [8, 1, 1024, 513]);  view_1079 = None
        permute_976: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1083, [0, 2, 1, 3]);  view_1083 = None
        slice_1287: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_976, 1, -256, 9223372036854775807);  permute_976 = None
        slice_1288: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1287, 3, -257, 9223372036854775807);  slice_1287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_86: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_322, full_default_42, slice_1288);  convert_element_type_322 = full_default_42 = slice_1288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_131: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1293, where_86);  slice_1293 = where_86 = None
        slice_scatter_240: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1294, copy_131, 3, -257, 9223372036854775807);  slice_1294 = copy_131 = None
        slice_scatter_241: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_980, slice_scatter_240, 1, -256, 9223372036854775807);  permute_980 = slice_scatter_240 = None
        permute_981: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_241, [0, 2, 1, 3]);  slice_scatter_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_984: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_981, [0, 2, 1, 3]);  permute_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_155: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_983, permute_984);  permute_983 = permute_984 = None
        permute_985: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_155, [0, 2, 1, 3]);  add_155 = None
        permute_986: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_985, [0, 2, 1, 3]);  permute_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_323: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_986, torch.float32);  permute_986 = None
        clone_179: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_323, memory_format = torch.contiguous_format);  convert_element_type_323 = None
        amax_10: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_179, [-1], True)
        sub_84: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_179, amax_10);  clone_179 = amax_10 = None
        exp_10: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
        sum_11: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_107: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_87: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_228, full_default_43, div_107);  unsqueeze_228 = full_default_43 = div_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_324: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_87, torch.bfloat16);  where_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_988: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_324, [0, 2, 1, 3]);  convert_element_type_324 = None
        clone_181: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_988, memory_format = torch.contiguous_format);  permute_988 = None
        view_1095: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_181, [96, 4, 256, 513]);  clone_181 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_43: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1095, [0, 257], 0.0);  view_1095 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1097: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_43, [96, 4, -1]);  constant_pad_nd_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1297: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1097, 2, 0, -256);  view_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1098: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_1297, [96, 4, 256, 769]);  slice_1297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1298: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1098, 3, 0, -1);  view_1098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_229: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_1298, 4);  slice_1298 = None
        view_1099: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_229, [384, 256, 768]);  unsqueeze_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_172: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_910, memory_format = torch.contiguous_format);  permute_910 = None
        view_1014: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [8192, 768]);  clone_172 = None
        permute_913: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        mm_42: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1014, permute_913);  view_1014 = permute_913 = None
        view_1015: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [1024, 8, 768]);  mm_42 = None
        add_152: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1015, arg168_1);  view_1015 = arg168_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1094: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_152, [1024, 8, 12, 64]);  add_152 = None
        permute_987: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1094, [1, 0, 2, 3]);  view_1094 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_989: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_987, [0, 2, 1, 3]);  permute_987 = None
        view_1096: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_989, [96, 1024, 64]);  permute_989 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_42: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1096, [0, 0, 256, 256], -1.0);  view_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_98: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_42, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_230: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_98, 4);  as_strided_98 = None
        clone_182: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_230, memory_format = torch.contiguous_format);  unsqueeze_230 = None
        view_1100: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_182, [384, 768, 64]);  clone_182 = None
        bmm_21: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1099, view_1100);  view_1099 = view_1100 = None
        view_1101: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [96, 4, 256, 1, 64]);  bmm_21 = None
        permute_994: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_1101, [0, 1, 2, 4, 3]);  view_1101 = None
        view_1102: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_994, [96, 4, 256, 64]);  permute_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1103: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1102, [8, 12, 1024, 64]);  view_1102 = None
        permute_995: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_1103, [0, 2, 1, 3]);  view_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_996: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_995, [1, 0, 2, 3]);  permute_995 = None
        clone_183: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_996, memory_format = torch.contiguous_format);  permute_996 = None
        view_1104: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_183, [1024, 8, 768]);  clone_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_997: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_1104, [1, 0, 2]);  view_1104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_184: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_997, memory_format = torch.contiguous_format);  permute_997 = None
        view_1105: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_184, [8192, 768]);  clone_184 = None
        permute_998: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_43: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1105, permute_998);  view_1105 = permute_998 = None
        view_1106: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [8, 1024, 768]);  mm_43 = None
        add_157: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1106, arg170_1);  view_1106 = arg170_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_158: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_157, convert_element_type_309);  add_157 = convert_element_type_309 = None
        convert_element_type_329: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_158, torch.float32);  add_158 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_329, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_86: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_329, getitem_41);  convert_element_type_329 = getitem_41 = None
        add_159: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        mul_81: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_20);  sub_86 = rsqrt_20 = None
        mul_82: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, arg171_1);  mul_81 = arg171_1 = None
        add_160: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, arg172_1);  mul_82 = arg172_1 = None
        convert_element_type_330: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16);  add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1107: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_330, [8192, 768])
        permute_999: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_20: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg174_1, view_1107, permute_999);  arg174_1 = view_1107 = permute_999 = None
        view_1108: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [8, 1024, 3072]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_334: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1108, torch.float32);  view_1108 = None
        mul_83: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_334, 0.5)
        mul_84: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_334, 0.7071067811865476);  convert_element_type_334 = None
        erf_10: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_161: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_85: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, add_161);  mul_83 = add_161 = None
        convert_element_type_335: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1109: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_335, [8192, 3072]);  convert_element_type_335 = None
        permute_1000: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_21: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg176_1, view_1109, permute_1000);  arg176_1 = view_1109 = permute_1000 = None
        view_1110: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [8, 1024, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_162: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1110, convert_element_type_330);  view_1110 = convert_element_type_330 = None
        convert_element_type_339: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_162, torch.float32);  add_162 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_339, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_106: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_107: "bf16[8, 1, 256, 257][65792, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_248: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 2);  arg8_1 = None
        unsqueeze_249: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        full_default_47: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_99: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1303: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_99, 1, 0, -1)
        slice_1301: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_99, 1, 0, -1)
        slice_1302: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1301, 3, 256, 9223372036854775807);  slice_1301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_87: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_339, getitem_43);  convert_element_type_339 = getitem_43 = None
        add_163: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        mul_86: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_21);  sub_87 = rsqrt_21 = None
        mul_87: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, arg177_1);  mul_86 = arg177_1 = None
        add_164: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_87, arg178_1);  mul_87 = arg178_1 = None
        convert_element_type_340: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1001: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_340, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_187: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_1001, memory_format = torch.contiguous_format)
        view_1111: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_187, [8192, 768]);  clone_187 = None
        permute_1002: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        mm_44: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1111, permute_1002);  view_1111 = permute_1002 = None
        view_1112: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [1024, 8, 768]);  mm_44 = None
        add_165: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1112, arg180_1);  view_1112 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_110: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_165, 8.0);  add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1140: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(div_110, [1024, 8, 12, 64]);  div_110 = None
        permute_1026: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1140, [1, 0, 2, 3]);  view_1140 = None
        permute_1027: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_1026, [0, 2, 1, 3]);  permute_1026 = None
        view_1141: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1027, [96, 1024, 64]);  permute_1027 = None
        view_1142: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1141, [96, 2, 512, 64]);  view_1141 = None
        as_strided_104: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_1142, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_1142 = None
        unsqueeze_235: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_104, 4);  as_strided_104 = None
        clone_190: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_235, memory_format = torch.contiguous_format);  unsqueeze_235 = None
        view_1143: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_190, [288, 512, 64]);  clone_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_188: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_1001, memory_format = torch.contiguous_format)
        view_1113: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_188, [8192, 768]);  clone_188 = None
        permute_1003: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        mm_45: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1113, permute_1003);  view_1113 = permute_1003 = None
        view_1114: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [1024, 8, 768]);  mm_45 = None
        add_166: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1114, arg182_1);  view_1114 = arg182_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1119: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_166, [1024, 8, 12, 64]);  add_166 = None
        permute_1006: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1119, [1, 0, 2, 3]);  view_1119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1011: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_1006, [0, 2, 1, 3]);  permute_1006 = None
        view_1123: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1011, [96, 1024, 64]);  permute_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1130: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1123, [96, 2, 512, 64]);  view_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_100: "bf16[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(view_1130, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_1130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_233: "bf16[96, 3, 512, 64, 1][64, 1572864, 6144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_100, 4);  as_strided_100 = None
        permute_1021: "bf16[96, 3, 1, 512, 64][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_233, [0, 1, 4, 2, 3]);  unsqueeze_233 = None
        permute_1030: "bf16[96, 3, 64, 512, 1][64, 1572864, 1, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_1021, [0, 1, 4, 3, 2]);  permute_1021 = None
        clone_191: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_1030, memory_format = torch.contiguous_format);  permute_1030 = None
        view_1144: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_191, [288, 64, 512]);  clone_191 = None
        bmm_22: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_1143, view_1144);  view_1143 = view_1144 = None
        view_1145: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [96, 3, 512, 1, 512]);  bmm_22 = None
        permute_1031: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_1145, [0, 1, 2, 4, 3]);  view_1145 = None
        view_1146: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1031, [96, 3, 512, 512]);  permute_1031 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_44: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1146, [0, 0, 0, 1], 0.0);  view_1146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1147: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_44, [96, 3, 512, 513]);  constant_pad_nd_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1299: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1147, 2, 0, 256)
        slice_1300: "bf16[96, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1299, 3, 0, 257);  slice_1299 = None
        copy_132: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1302, slice_1300);  slice_1302 = slice_1300 = None
        slice_scatter_242: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1303, copy_132, 3, 256, 9223372036854775807);  slice_1303 = copy_132 = None
        slice_scatter_243: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_99, slice_scatter_242, 1, 0, -1);  full_99 = slice_scatter_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_290: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_243, 1, -1)
        select_289: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_243, 1, -1)
        slice_1309: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_289, 2, 256, 9223372036854775807);  select_289 = None
        select_286: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_1147, 1, -1)
        slice_1306: "bf16[96, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_286, 1, 256, 9223372036854775807);  select_286 = None
        slice_1307: "bf16[96, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1306, 2, 0, 257);  slice_1306 = None
        copy_133: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1309, slice_1307);  slice_1309 = slice_1307 = None
        slice_scatter_244: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_290, copy_133, 2, 256, 9223372036854775807);  select_290 = copy_133 = None
        select_scatter_44: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_243, slice_scatter_244, 1, -1);  slice_scatter_243 = slice_scatter_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1318: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_44, 1, 1, 9223372036854775807)
        slice_1316: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_44, 1, 1, 9223372036854775807)
        slice_1317: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1316, 3, 0, 256);  slice_1316 = None
        slice_1311: "bf16[96, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1147, 2, -257, -1)
        slice_1312: "bf16[96, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1311, 3, 257, 9223372036854775807);  slice_1311 = None
        copy_134: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1317, slice_1312);  slice_1317 = slice_1312 = None
        slice_scatter_245: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1318, copy_134, 3, 0, 256);  slice_1318 = copy_134 = None
        slice_scatter_246: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_44, slice_scatter_245, 1, 1, 9223372036854775807);  select_scatter_44 = slice_scatter_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_297: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_246, 1, 0)
        slice_1328: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_297, 1, 1, 256)
        select_296: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_246, 1, 0)
        slice_1326: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_296, 1, 1, 256);  select_296 = None
        slice_1327: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1326, 2, 1, 256);  slice_1326 = None
        select_292: "bf16[96, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_1147, 1, 0);  view_1147 = None
        slice_1321: "bf16[96, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_292, 1, 0, 255);  select_292 = None
        slice_1322: "bf16[96, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1321, 2, -255, 9223372036854775807);  slice_1321 = None
        copy_135: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1327, slice_1322);  slice_1327 = slice_1322 = None
        slice_scatter_247: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1328, copy_135, 2, 1, 256);  slice_1328 = copy_135 = None
        slice_scatter_248: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_297, slice_scatter_247, 1, 1, 256);  select_297 = slice_scatter_247 = None
        select_scatter_45: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_246, slice_scatter_248, 1, 0);  slice_scatter_246 = slice_scatter_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1155: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_45, [8, 12, 1024, 513])
        permute_1039: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1155, [0, 2, 1, 3]);  view_1155 = None
        slice_1341: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1039, 1, 0, 256)
        view_1154: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_45, [8, 12, 1024, 513])
        permute_1038: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1154, [0, 2, 1, 3]);  view_1154 = None
        slice_1339: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1038, 1, 0, 256);  permute_1038 = None
        slice_1340: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1339, 3, 0, 257);  slice_1339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_44: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_236: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_44, -2);  iota_44 = None
        iota_45: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_237: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
        sub_89: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_236, unsqueeze_237);  unsqueeze_236 = unsqueeze_237 = None
        le_22: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_89, 0);  sub_89 = None
        full_100: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_44: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_88: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_22, full_100, scalar_tensor_44);  le_22 = full_100 = scalar_tensor_44 = None
        rev_44: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_88, [0]);  where_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_238: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_44, 0);  rev_44 = None
        unsqueeze_239: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_44: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_239, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_349: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_44, torch.bool);  expand_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_101: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_1036: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_101, [0, 2, 1, 3]);  full_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1152: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_45, [8, 12, 1024, 513]);  select_scatter_45 = None
        permute_1035: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1152, [0, 2, 1, 3]);  view_1152 = None
        slice_1334: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1035, 1, 0, 256);  permute_1035 = None
        slice_1335: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1334, 3, 0, 257);  slice_1334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_89: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_349, permute_1036, slice_1335);  convert_element_type_349 = permute_1036 = slice_1335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_136: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1340, where_89);  slice_1340 = where_89 = None
        slice_scatter_249: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1341, copy_136, 3, 0, 257);  slice_1341 = copy_136 = None
        slice_scatter_250: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1039, slice_scatter_249, 1, 0, 256);  permute_1039 = slice_scatter_249 = None
        permute_1040: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_250, [0, 2, 1, 3]);  slice_scatter_250 = None
        view_1156: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1040, [96, 4, 256, 513]);  permute_1040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1163: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1156, [8, 12, 1024, 513])
        permute_1048: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1163, [0, 2, 1, 3]);  view_1163 = None
        slice_1354: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1048, 1, -256, 9223372036854775807)
        view_1162: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1156, [8, 12, 1024, 513])
        permute_1047: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1162, [0, 2, 1, 3]);  view_1162 = None
        slice_1352: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1047, 1, -256, 9223372036854775807);  permute_1047 = None
        slice_1353: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1352, 3, -257, 9223372036854775807);  slice_1352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_45: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_239, [1, 3]);  unsqueeze_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_45: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_45, [8, 256, 12, 257]);  rev_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_350: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_45, torch.bool);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_102: "bf16[8, 12, 256, 257][789504, 65792, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_1045: "bf16[8, 256, 12, 257][789504, 257, 65792, 1]cuda:0" = torch.ops.aten.permute.default(full_102, [0, 2, 1, 3]);  full_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1160: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1156, [8, 12, 1024, 513]);  view_1156 = None
        permute_1044: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1160, [0, 2, 1, 3]);  view_1160 = None
        slice_1347: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1044, 1, -256, 9223372036854775807);  permute_1044 = None
        slice_1348: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1347, 3, -257, 9223372036854775807);  slice_1347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_90: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_350, permute_1045, slice_1348);  convert_element_type_350 = permute_1045 = slice_1348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_137: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1353, where_90);  slice_1353 = where_90 = None
        slice_scatter_251: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1354, copy_137, 3, -257, 9223372036854775807);  slice_1354 = copy_137 = None
        slice_scatter_252: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1048, slice_scatter_251, 1, -256, 9223372036854775807);  permute_1048 = slice_scatter_251 = None
        permute_1049: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_252, [0, 2, 1, 3]);  slice_scatter_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1074: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_1049, [0, 2, 1, 3]);  permute_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_104: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1361: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1)
        slice_1359: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1)
        slice_1360: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1359, 3, 256, 9223372036854775807);  slice_1359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_103: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1051: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(full_103, [0, 2, 1, 3]);  full_103 = None
        view_1166: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1051, [8, 1024, 1]);  permute_1051 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1168: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1166, [8, 2, 512, 1]);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_105: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_1168, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_1168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_242: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_105, 4);  as_strided_105 = None
        permute_1053: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_242, [0, 1, 2, 4, 3]);  unsqueeze_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_44: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_11: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg7_1, 0);  arg7_1 = None
        unsqueeze_240: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(ne_11, 2);  ne_11 = None
        unsqueeze_241: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_351: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_241, torch.bfloat16)
        where_91: "bf16[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_241, full_default_44, convert_element_type_351);  unsqueeze_241 = full_default_44 = convert_element_type_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1052: "bf16[8, 1, 1024, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(where_91, [0, 2, 1, 3]);  where_91 = None
        view_1167: "bf16[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1052, [8, 1024, 1]);  permute_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1169: "bf16[8, 2, 512, 1][1024, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1167, [8, 2, 512, 1]);  view_1167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_106: "bf16[8, 3, 512, 1][1024, 256, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(view_1169, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_243: "bf16[8, 3, 512, 1, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_106, 4);  as_strided_106 = None
        permute_1054: "bf16[8, 3, 1, 512, 1][1024, 256, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_243, [0, 1, 4, 2, 3]);  unsqueeze_243 = None
        mul_88: "bf16[8, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1053, permute_1054);  permute_1053 = permute_1054 = None
        view_1170: "bf16[8, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_88, [8, 3, 512, 512]);  mul_88 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_45: "bf16[8, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1170, [0, 0, 0, 1], 0.0);  view_1170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1171: "bf16[8, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_45, [8, 3, 512, 513]);  constant_pad_nd_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1357: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1171, 2, 0, 256)
        slice_1358: "bf16[8, 3, 256, 257][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1357, 3, 0, 257);  slice_1357 = None
        copy_138: "bf16[8, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1360, slice_1358);  slice_1360 = slice_1358 = None
        slice_scatter_253: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1361, copy_138, 3, 256, 9223372036854775807);  slice_1361 = copy_138 = None
        slice_scatter_254: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_104, slice_scatter_253, 1, 0, -1);  full_104 = slice_scatter_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_303: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_254, 1, -1)
        select_302: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_254, 1, -1)
        slice_1367: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_302, 2, 256, 9223372036854775807);  select_302 = None
        select_299: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_1171, 1, -1)
        slice_1364: "bf16[8, 256, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_299, 1, 256, 9223372036854775807);  select_299 = None
        slice_1365: "bf16[8, 256, 257][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1364, 2, 0, 257);  slice_1364 = None
        copy_139: "bf16[8, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1367, slice_1365);  slice_1367 = slice_1365 = None
        slice_scatter_255: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_303, copy_139, 2, 256, 9223372036854775807);  select_303 = copy_139 = None
        select_scatter_46: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_254, slice_scatter_255, 1, -1);  slice_scatter_254 = slice_scatter_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1376: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_46, 1, 1, 9223372036854775807)
        slice_1374: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_46, 1, 1, 9223372036854775807)
        slice_1375: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1374, 3, 0, 256);  slice_1374 = None
        slice_1369: "bf16[8, 3, 256, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1171, 2, -257, -1)
        slice_1370: "bf16[8, 3, 256, 256][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1369, 3, 257, 9223372036854775807);  slice_1369 = None
        copy_140: "bf16[8, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1375, slice_1370);  slice_1375 = slice_1370 = None
        slice_scatter_256: "bf16[8, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1376, copy_140, 3, 0, 256);  slice_1376 = copy_140 = None
        slice_scatter_257: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_46, slice_scatter_256, 1, 1, 9223372036854775807);  select_scatter_46 = slice_scatter_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_310: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_257, 1, 0)
        slice_1386: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_310, 1, 1, 256)
        select_309: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_257, 1, 0)
        slice_1384: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_309, 1, 1, 256);  select_309 = None
        slice_1385: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1384, 2, 1, 256);  slice_1384 = None
        select_305: "bf16[8, 512, 513][787968, 513, 1]cuda:0" = torch.ops.aten.select.int(view_1171, 1, 0);  view_1171 = None
        slice_1379: "bf16[8, 255, 513][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_305, 1, 0, 255);  select_305 = None
        slice_1380: "bf16[8, 255, 255][787968, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1379, 2, -255, 9223372036854775807);  slice_1379 = None
        copy_141: "bf16[8, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1385, slice_1380);  slice_1385 = slice_1380 = None
        slice_scatter_258: "bf16[8, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1386, copy_141, 2, 1, 256);  slice_1386 = copy_141 = None
        slice_scatter_259: "bf16[8, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_310, slice_scatter_258, 1, 1, 256);  select_310 = slice_scatter_258 = None
        select_scatter_47: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_257, slice_scatter_259, 1, 0);  slice_scatter_257 = slice_scatter_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1179: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_47, [8, 1, 1024, 513])
        permute_1062: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1179, [0, 2, 1, 3]);  view_1179 = None
        slice_1399: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1062, 1, 0, 256)
        view_1178: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_47, [8, 1, 1024, 513])
        permute_1061: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1178, [0, 2, 1, 3]);  view_1178 = None
        slice_1397: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1061, 1, 0, 256);  permute_1061 = None
        slice_1398: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1397, 3, 0, 257);  slice_1397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_46: "i64[257][1]cuda:0" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_244: "i64[1, 257][257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_46, -2);  iota_46 = None
        iota_47: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_245: "i64[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
        sub_91: "i64[256, 257][257, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_244, unsqueeze_245);  unsqueeze_244 = unsqueeze_245 = None
        le_23: "b8[256, 257][257, 1]cuda:0" = torch.ops.aten.le.Scalar(sub_91, 0);  sub_91 = None
        full_105: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scalar_tensor_46: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_92: "bf16[256, 257][257, 1]cuda:0" = torch.ops.aten.where.self(le_23, full_105, scalar_tensor_46);  le_23 = full_105 = scalar_tensor_46 = None
        rev_46: "bf16[256, 257][257, 1]cuda:0" = torch.ops.prims.rev.default(where_92, [0]);  where_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_246: "bf16[1, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(rev_46, 0);  rev_46 = None
        unsqueeze_247: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_46: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_247, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_352: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_46, torch.bool);  expand_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_45: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1176: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter_47, [8, 1, 1024, 513]);  select_scatter_47 = None
        permute_1058: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1176, [0, 2, 1, 3]);  view_1176 = None
        slice_1392: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1058, 1, 0, 256);  permute_1058 = None
        slice_1393: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1392, 3, 0, 257);  slice_1392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_93: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_352, full_default_45, slice_1393);  convert_element_type_352 = full_default_45 = slice_1393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_142: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1398, where_93);  slice_1398 = where_93 = None
        slice_scatter_260: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1399, copy_142, 3, 0, 257);  slice_1399 = copy_142 = None
        slice_scatter_261: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1062, slice_scatter_260, 1, 0, 256);  permute_1062 = slice_scatter_260 = None
        permute_1063: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_261, [0, 2, 1, 3]);  slice_scatter_261 = None
        view_1180: "bf16[8, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1063, [8, 4, 256, 513]);  permute_1063 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1187: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1180, [8, 1, 1024, 513])
        permute_1071: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1187, [0, 2, 1, 3]);  view_1187 = None
        slice_1412: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1071, 1, -256, 9223372036854775807)
        view_1186: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1180, [8, 1, 1024, 513])
        permute_1070: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1186, [0, 2, 1, 3]);  view_1186 = None
        slice_1410: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1070, 1, -256, 9223372036854775807);  permute_1070 = None
        slice_1411: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1410, 3, -257, 9223372036854775807);  slice_1410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_47: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.rev.default(unsqueeze_247, [1, 3]);  unsqueeze_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_47: "bf16[8, 256, 1, 257][0, 257, 257, 1]cuda:0" = torch.ops.aten.expand.default(rev_47, [8, 256, 1, 257]);  rev_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_353: "b8[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_47, torch.bool);  expand_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_46: "bf16[8, 256, 1, 257][65792, 257, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1184: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1180, [8, 1, 1024, 513]);  view_1180 = None
        permute_1067: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1184, [0, 2, 1, 3]);  view_1184 = None
        slice_1405: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1067, 1, -256, 9223372036854775807);  permute_1067 = None
        slice_1406: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1405, 3, -257, 9223372036854775807);  slice_1405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_94: "bf16[8, 256, 1, 257][65792, 257, 65792, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_353, full_default_46, slice_1406);  convert_element_type_353 = full_default_46 = slice_1406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_143: "bf16[8, 256, 1, 257][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1411, where_94);  slice_1411 = where_94 = None
        slice_scatter_262: "bf16[8, 256, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1412, copy_143, 3, -257, 9223372036854775807);  slice_1412 = copy_143 = None
        slice_scatter_263: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1071, slice_scatter_262, 1, -256, 9223372036854775807);  permute_1071 = slice_scatter_262 = None
        permute_1072: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_263, [0, 2, 1, 3]);  slice_scatter_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1075: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_1072, [0, 2, 1, 3]);  permute_1072 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_170: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_1074, permute_1075);  permute_1074 = permute_1075 = None
        permute_1076: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_170, [0, 2, 1, 3]);  add_170 = None
        permute_1077: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_1076, [0, 2, 1, 3]);  permute_1076 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_354: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1077, torch.float32);  permute_1077 = None
        clone_196: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_354, memory_format = torch.contiguous_format);  convert_element_type_354 = None
        amax_11: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.amax.default(clone_196, [-1], True)
        sub_92: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_196, amax_11);  clone_196 = amax_11 = None
        exp_11: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_92);  sub_92 = None
        sum_12: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_117: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_95: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_249, full_default_47, div_117);  unsqueeze_249 = full_default_47 = div_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_355: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_95, torch.bfloat16);  where_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_1079: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_355, [0, 2, 1, 3]);  convert_element_type_355 = None
        clone_198: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1079, memory_format = torch.contiguous_format);  permute_1079 = None
        view_1196: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_198, [96, 4, 256, 513]);  clone_198 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_47: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1196, [0, 257], 0.0);  view_1196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1198: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_47, [96, 4, -1]);  constant_pad_nd_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1415: "bf16[96, 4, 196864][788480, 197120, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1198, 2, 0, -256);  view_1198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1199: "bf16[96, 4, 256, 769][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(slice_1415, [96, 4, 256, 769]);  slice_1415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1416: "bf16[96, 4, 256, 768][788480, 197120, 769, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_1199, 3, 0, -1);  view_1199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_250: "bf16[96, 4, 256, 768, 1][788480, 197120, 769, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_1416, 4);  slice_1416 = None
        view_1200: "bf16[384, 256, 768][197120, 769, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_250, [384, 256, 768]);  unsqueeze_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        clone_189: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_1001, memory_format = torch.contiguous_format);  permute_1001 = None
        view_1115: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_189, [8192, 768]);  clone_189 = None
        permute_1004: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        mm_46: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1115, permute_1004);  view_1115 = permute_1004 = None
        view_1116: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [1024, 8, 768]);  mm_46 = None
        add_167: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1116, arg184_1);  view_1116 = arg184_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1195: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(add_167, [1024, 8, 12, 64]);  add_167 = None
        permute_1078: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1195, [1, 0, 2, 3]);  view_1195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1080: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.permute.default(permute_1078, [0, 2, 1, 3]);  permute_1078 = None
        view_1197: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1080, [96, 1024, 64]);  permute_1080 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_46: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1197, [0, 0, 256, 256], -1.0);  view_1197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_107: "bf16[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(constant_pad_nd_46, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_251: "bf16[96, 4, 768, 64, 1][98304, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(as_strided_107, 4);  as_strided_107 = None
        clone_199: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.clone.default(unsqueeze_251, memory_format = torch.contiguous_format);  unsqueeze_251 = None
        view_1201: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_199, [384, 768, 64]);  clone_199 = None
        bmm_23: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1200, view_1201);  view_1200 = view_1201 = None
        view_1202: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [96, 4, 256, 1, 64]);  bmm_23 = None
        permute_1085: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_1202, [0, 1, 2, 4, 3]);  view_1202 = None
        view_1203: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1085, [96, 4, 256, 64]);  permute_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1204: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1203, [8, 12, 1024, 64]);  view_1203 = None
        permute_1086: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_1204, [0, 2, 1, 3]);  view_1204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_1087: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1086, [1, 0, 2, 3]);  permute_1086 = None
        clone_200: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1087, memory_format = torch.contiguous_format);  permute_1087 = None
        view_1205: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_200, [1024, 8, 768]);  clone_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1088: "bf16[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(view_1205, [1, 0, 2]);  view_1205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        clone_201: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_1088, memory_format = torch.contiguous_format);  permute_1088 = None
        view_1206: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [8192, 768]);  clone_201 = None
        permute_1089: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_47: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1206, permute_1089);  view_1206 = permute_1089 = None
        view_1207: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [8, 1024, 768]);  mm_47 = None
        add_172: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1207, arg186_1);  view_1207 = arg186_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_173: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_172, convert_element_type_340);  add_172 = convert_element_type_340 = None
        convert_element_type_360: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.float32);  add_173 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_360, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        sub_94: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_360, getitem_45);  convert_element_type_360 = getitem_45 = None
        add_174: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_89: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_22);  sub_94 = rsqrt_22 = None
        mul_90: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, arg187_1);  mul_89 = arg187_1 = None
        add_175: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, arg188_1);  mul_90 = arg188_1 = None
        convert_element_type_361: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1208: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [8192, 768])
        permute_1090: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_22: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(arg190_1, view_1208, permute_1090);  arg190_1 = view_1208 = permute_1090 = None
        view_1209: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [8, 1024, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_365: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1209, torch.float32);  view_1209 = None
        mul_91: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_365, 0.5)
        mul_92: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_365, 0.7071067811865476);  convert_element_type_365 = None
        erf_11: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_176: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_93: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, add_176);  mul_91 = add_176 = None
        convert_element_type_366: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1210: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_366, [8192, 3072]);  convert_element_type_366 = None
        permute_1091: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_23: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(arg192_1, view_1210, permute_1091);  arg192_1 = view_1210 = permute_1091 = None
        view_1211: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [8, 1024, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_177: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1211, convert_element_type_361);  view_1211 = convert_element_type_361 = None
        convert_element_type_370: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_177, torch.float32);  add_177 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_370, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        sub_95: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_370, getitem_47);  convert_element_type_370 = getitem_47 = None
        add_178: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_94: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_23);  sub_95 = rsqrt_23 = None
        mul_95: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, arg193_1);  mul_94 = arg193_1 = None
        add_179: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, arg194_1);  mul_95 = arg194_1 = None
        convert_element_type_371: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None
        return (convert_element_type_371,)

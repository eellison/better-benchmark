class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 1024, 768]", primals_2: "f32[768, 768]", primals_3: "f32[768]", primals_4: "f32[768, 768]", primals_5: "f32[768]", primals_6: "f32[768, 768]", primals_7: "f32[768]", primals_8: "f32[8, 1024]", primals_9: "b8[8, 1024]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[3072, 768]", primals_15: "f32[3072]", primals_16: "f32[768, 3072]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[768, 768]", primals_21: "f32[768]", primals_22: "f32[768, 768]", primals_23: "f32[768]", primals_24: "f32[768, 768]", primals_25: "f32[768]", primals_26: "f32[768, 768]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[3072, 768]", primals_31: "f32[3072]", primals_32: "f32[768, 3072]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768, 768]", primals_37: "f32[768]", primals_38: "f32[768, 768]", primals_39: "f32[768]", primals_40: "f32[768, 768]", primals_41: "f32[768]", primals_42: "f32[768, 768]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[3072, 768]", primals_47: "f32[3072]", primals_48: "f32[768, 3072]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768, 768]", primals_53: "f32[768]", primals_54: "f32[768, 768]", primals_55: "f32[768]", primals_56: "f32[768, 768]", primals_57: "f32[768]", primals_58: "f32[768, 768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[3072, 768]", primals_63: "f32[3072]", primals_64: "f32[768, 3072]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[768, 768]", primals_69: "f32[768]", primals_70: "f32[768, 768]", primals_71: "f32[768]", primals_72: "f32[768, 768]", primals_73: "f32[768]", primals_74: "f32[768, 768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[3072, 768]", primals_79: "f32[3072]", primals_80: "f32[768, 3072]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[768, 768]", primals_85: "f32[768]", primals_86: "f32[768, 768]", primals_87: "f32[768]", primals_88: "f32[768, 768]", primals_89: "f32[768]", primals_90: "f32[768, 768]", primals_91: "f32[768]", primals_92: "f32[768]", primals_93: "f32[768]", primals_94: "f32[3072, 768]", primals_95: "f32[3072]", primals_96: "f32[768, 3072]", primals_97: "f32[768]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768, 768]", primals_101: "f32[768]", primals_102: "f32[768, 768]", primals_103: "f32[768]", primals_104: "f32[768, 768]", primals_105: "f32[768]", primals_106: "f32[768, 768]", primals_107: "f32[768]", primals_108: "f32[768]", primals_109: "f32[768]", primals_110: "f32[3072, 768]", primals_111: "f32[3072]", primals_112: "f32[768, 3072]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[768]", primals_116: "f32[768, 768]", primals_117: "f32[768]", primals_118: "f32[768, 768]", primals_119: "f32[768]", primals_120: "f32[768, 768]", primals_121: "f32[768]", primals_122: "f32[768, 768]", primals_123: "f32[768]", primals_124: "f32[768]", primals_125: "f32[768]", primals_126: "f32[3072, 768]", primals_127: "f32[3072]", primals_128: "f32[768, 3072]", primals_129: "f32[768]", primals_130: "f32[768]", primals_131: "f32[768]", primals_132: "f32[768, 768]", primals_133: "f32[768]", primals_134: "f32[768, 768]", primals_135: "f32[768]", primals_136: "f32[768, 768]", primals_137: "f32[768]", primals_138: "f32[768, 768]", primals_139: "f32[768]", primals_140: "f32[768]", primals_141: "f32[768]", primals_142: "f32[3072, 768]", primals_143: "f32[3072]", primals_144: "f32[768, 3072]", primals_145: "f32[768]", primals_146: "f32[768]", primals_147: "f32[768]", primals_148: "f32[768, 768]", primals_149: "f32[768]", primals_150: "f32[768, 768]", primals_151: "f32[768]", primals_152: "f32[768, 768]", primals_153: "f32[768]", primals_154: "f32[768, 768]", primals_155: "f32[768]", primals_156: "f32[768]", primals_157: "f32[768]", primals_158: "f32[3072, 768]", primals_159: "f32[3072]", primals_160: "f32[768, 3072]", primals_161: "f32[768]", primals_162: "f32[768]", primals_163: "f32[768]", primals_164: "f32[768, 768]", primals_165: "f32[768]", primals_166: "f32[768, 768]", primals_167: "f32[768]", primals_168: "f32[768, 768]", primals_169: "f32[768]", primals_170: "f32[768, 768]", primals_171: "f32[768]", primals_172: "f32[768]", primals_173: "f32[768]", primals_174: "f32[3072, 768]", primals_175: "f32[3072]", primals_176: "f32[768, 3072]", primals_177: "f32[768]", primals_178: "f32[768]", primals_179: "f32[768]", primals_180: "f32[768, 768]", primals_181: "f32[768]", primals_182: "f32[768, 768]", primals_183: "f32[768]", primals_184: "f32[768, 768]", primals_185: "f32[768]", primals_186: "f32[768, 768]", primals_187: "f32[768]", primals_188: "f32[768]", primals_189: "f32[768]", primals_190: "f32[3072, 768]", primals_191: "f32[3072]", primals_192: "f32[768, 3072]", primals_193: "f32[768]", primals_194: "f32[768]", primals_195: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(primals_1, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        clone: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone, [8192, 768]);  clone = None
        mm: "f32[8192, 768]" = torch.ops.aten.mm.default(view, permute_1);  permute_1 = None
        view_1: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm, [1024, 8, 768]);  mm = None
        add: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1, primals_3);  view_1 = primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        mm_1: "f32[8192, 768]" = torch.ops.aten.mm.default(view, permute_2);  permute_2 = None
        view_3: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_1, [1024, 8, 768]);  mm_1 = None
        add_1: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_3, primals_5);  view_3 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        mm_2: "f32[8192, 768]" = torch.ops.aten.mm.default(view, permute_3);  permute_3 = None
        view_5: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_2, [1024, 8, 768]);  mm_2 = None
        add_2: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_5, primals_7);  view_5 = primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add, 8.0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_8: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_1, [1024, 8, 12, 64]);  add_1 = None
        permute_5: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_8, [1, 0, 2, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_10: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        view_12: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_10, [96, 1024, 64]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_19: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_12, [96, 2, 512, 64]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_1: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_19, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_2: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        permute_20: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 4, 2, 3]);  unsqueeze_2 = None
        view_29: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div, [1024, 8, 12, 64]);  div = None
        permute_25: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_29, [1, 0, 2, 3]);  view_29 = None
        permute_26: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_25, [0, 2, 1, 3]);  permute_25 = None
        view_30: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_26, [96, 1024, 64]);  permute_26 = None
        view_31: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_30, [96, 2, 512, 64]);  view_30 = None
        as_strided_5: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_31, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_31 = None
        unsqueeze_4: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_5, 4);  as_strided_5 = None
        clone_3: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_4, memory_format = torch.contiguous_format);  unsqueeze_4 = None
        view_32: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_3, [288, 512, 64]);  clone_3 = None
        permute_29: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_20, [0, 1, 4, 3, 2]);  permute_20 = None
        clone_4: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_33: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_4, [288, 64, 512]);  clone_4 = None
        bmm: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_32, view_33)
        view_34: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm, [96, 3, 512, 1, 512]);  bmm = None
        permute_30: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_34, [0, 1, 2, 4, 3]);  view_34 = None
        view_35: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_30, [96, 3, 512, 512]);  permute_30 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_35, [0, 0, 0, 1], 0.0);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_36: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd, [96, 3, 512, 513]);  constant_pad_nd = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full: "f32[96, 4, 256, 513]" = torch.ops.aten.full.default([96, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_36, 2, 0, 256)
        slice_2: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 257);  slice_1 = None
        slice_3: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_4: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 256, 9223372036854775807)
        copy: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_2);  slice_2 = None
        slice_scatter: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy, 3, 256, 9223372036854775807);  copy = None
        slice_scatter_1: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter, 1, 0, -1);  slice_scatter = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_36, 1, -1)
        slice_11: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807);  select = None
        slice_12: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_11, 2, 0, 257);  slice_11 = None
        select_3: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        slice_14: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_3, 2, 256, 9223372036854775807)
        copy_1: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_14, slice_12);  slice_14 = slice_12 = None
        slice_scatter_2: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_3, copy_1, 2, 256, 9223372036854775807);  select_3 = copy_1 = None
        select_scatter: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_1, slice_scatter_2, 1, -1);  slice_scatter_1 = slice_scatter_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_17: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_36, 2, -257, -1)
        slice_18: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_17, 3, 257, 9223372036854775807);  slice_17 = None
        slice_22: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_23: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 256)
        copy_2: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_23, slice_18);  slice_23 = slice_18 = None
        slice_scatter_3: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_22, copy_2, 3, 0, 256);  slice_22 = copy_2 = None
        slice_scatter_4: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_3, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_8: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_36, 1, 0);  view_36 = None
        slice_30: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_8, 1, 0, 255);  select_8 = None
        slice_31: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_30, 2, -255, 9223372036854775807);  slice_30 = None
        select_12: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_35: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_12, 1, 1, 256)
        slice_36: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_35, 2, 1, 256)
        copy_3: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_36, slice_31);  slice_36 = slice_31 = None
        slice_scatter_5: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_35, copy_3, 2, 1, 256);  slice_35 = copy_3 = None
        slice_scatter_6: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_12, slice_scatter_5, 1, 1, 256);  select_12 = slice_scatter_5 = None
        select_scatter_1: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_4, slice_scatter_6, 1, 0);  slice_scatter_4 = slice_scatter_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        full_default: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_5: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub_1: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_5, unsqueeze_6);  unsqueeze_5 = unsqueeze_6 = None
        le: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_1, 0);  sub_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[256, 257]" = torch.ops.aten.where.self(le, full_default, full_default_1);  le = full_default = None
        rev: "f32[256, 257]" = torch.ops.prims.rev.default(where, [0]);  where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:745 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_7: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
        unsqueeze_8: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_1: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_8, [1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand: "f32[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_8, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_41: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_1, [8, 12, 1024, 513]);  select_scatter_1 = None
        permute_34: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        slice_46: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_34, 1, 0, 256)
        slice_47: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_2: "f32[8, 12, 256, 257]" = torch.ops.aten.full.default([8, 12, 256, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_35: "f32[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_2, [0, 2, 1, 3]);  full_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        where_1: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_47)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_4: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_47, where_1);  slice_47 = where_1 = None
        slice_scatter_7: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_46, copy_4, 3, 0, 257);  slice_46 = copy_4 = None
        slice_scatter_8: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_34, slice_scatter_7, 1, 0, 256);  permute_34 = slice_scatter_7 = None
        permute_39: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3]);  slice_scatter_8 = None
        view_45: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_39, [96, 4, 256, 513]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_1: "f32[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, [8, 256, 12, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_55: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_45, [8, 12, 1024, 513]);  view_45 = None
        permute_47: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        slice_62: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_47, 1, -256, 9223372036854775807)
        slice_63: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_62, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_1: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        where_2: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_63)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_5: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_63, where_2);  slice_63 = where_2 = None
        slice_scatter_9: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_62, copy_5, 3, -257, 9223372036854775807);  slice_62 = copy_5 = None
        slice_scatter_10: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_47, slice_scatter_9, 1, -256, 9223372036854775807);  permute_47 = slice_scatter_9 = None
        permute_52: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3]);  slice_scatter_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:523 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(primals_8, 0);  primals_8 = None
        unsqueeze_9: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne, 2);  ne = None
        unsqueeze_10: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:526 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_2: "f32[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_10, torch.float32)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_10, full_default_2, convert_element_type_2);  unsqueeze_10 = full_default_2 = convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:531 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_4: "f32[8, 1024, 1, 1]" = torch.ops.aten.full.default([8, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:773 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_57: "f32[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
        view_65: "f32[8, 1024, 1]" = torch.ops.aten.reshape.default(permute_57, [8, 1024, 1]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_58: "f32[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
        view_66: "f32[8, 1024, 1]" = torch.ops.aten.reshape.default(permute_58, [8, 1024, 1]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_67: "f32[8, 2, 512, 1]" = torch.ops.aten.reshape.default(view_65, [8, 2, 512, 1]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_6: "f32[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_67, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_68: "f32[8, 2, 512, 1]" = torch.ops.aten.reshape.default(view_66, [8, 2, 512, 1]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_7: "f32[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_68, [8, 3, 512, 1], [1024, 256, 1, 1]);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_11: "f32[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_6, 4);  as_strided_6 = None
        permute_59: "f32[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_11, [0, 1, 2, 4, 3]);  unsqueeze_11 = None
        unsqueeze_12: "f32[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_7, 4);  as_strided_7 = None
        permute_60: "f32[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_12, [0, 1, 4, 2, 3]);  unsqueeze_12 = None
        mul: "f32[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_59, permute_60);  permute_59 = permute_60 = None
        view_69: "f32[8, 3, 512, 512]" = torch.ops.aten.reshape.default(mul, [8, 3, 512, 512]);  mul = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "f32[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_69, [0, 0, 0, 1], 0.0);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_70: "f32[8, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_1, [8, 3, 512, 513]);  constant_pad_nd_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_5: "f32[8, 4, 256, 513]" = torch.ops.aten.full.default([8, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_75: "f32[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_70, 2, 0, 256)
        slice_76: "f32[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_75, 3, 0, 257);  slice_75 = None
        slice_77: "f32[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_78: "f32[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_77, 3, 256, 9223372036854775807)
        copy_6: "f32[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_78, slice_76);  slice_78 = slice_76 = None
        slice_scatter_11: "f32[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_77, copy_6, 3, 256, 9223372036854775807);  slice_77 = copy_6 = None
        slice_scatter_12: "f32[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_11, 1, 0, -1);  full_5 = slice_scatter_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_18: "f32[8, 512, 513]" = torch.ops.aten.select.int(view_70, 1, -1)
        slice_82: "f32[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_18, 1, 256, 9223372036854775807);  select_18 = None
        slice_83: "f32[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_82, 2, 0, 257);  slice_82 = None
        select_21: "f32[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        slice_85: "f32[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_21, 2, 256, 9223372036854775807)
        copy_7: "f32[8, 256, 257]" = torch.ops.aten.copy.default(slice_85, slice_83);  slice_85 = slice_83 = None
        slice_scatter_13: "f32[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_21, copy_7, 2, 256, 9223372036854775807);  select_21 = copy_7 = None
        select_scatter_2: "f32[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_12, slice_scatter_13, 1, -1);  slice_scatter_12 = slice_scatter_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_87: "f32[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_70, 2, -257, -1)
        slice_88: "f32[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_87, 3, 257, 9223372036854775807);  slice_87 = None
        slice_92: "f32[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_93: "f32[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_92, 3, 0, 256)
        copy_8: "f32[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_93, slice_88);  slice_93 = slice_88 = None
        slice_scatter_14: "f32[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_92, copy_8, 3, 0, 256);  slice_92 = copy_8 = None
        slice_scatter_15: "f32[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_2, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter_2 = slice_scatter_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_24: "f32[8, 512, 513]" = torch.ops.aten.select.int(view_70, 1, 0);  view_70 = None
        slice_97: "f32[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_24, 1, 0, 255);  select_24 = None
        slice_98: "f32[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_97, 2, -255, 9223372036854775807);  slice_97 = None
        select_28: "f32[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_102: "f32[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_28, 1, 1, 256)
        slice_103: "f32[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_102, 2, 1, 256)
        copy_9: "f32[8, 255, 255]" = torch.ops.aten.copy.default(slice_103, slice_98);  slice_103 = slice_98 = None
        slice_scatter_16: "f32[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_102, copy_9, 2, 1, 256);  slice_102 = copy_9 = None
        slice_scatter_17: "f32[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_28, slice_scatter_16, 1, 1, 256);  select_28 = slice_scatter_16 = None
        select_scatter_3: "f32[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_17, 1, 0);  slice_scatter_15 = slice_scatter_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_2: "f32[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_8, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_75: "f32[8, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_3, [8, 1, 1024, 513]);  select_scatter_3 = None
        permute_64: "f32[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        slice_110: "f32[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_64, 1, 0, 256)
        slice_111: "f32[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_110, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_5: "f32[8, 256, 1, 257]" = torch.ops.aten.full.default([8, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_3: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None
        where_5: "f32[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_3, full_default_5, slice_111);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_10: "f32[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_111, where_5);  slice_111 = where_5 = None
        slice_scatter_18: "f32[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_110, copy_10, 3, 0, 257);  slice_110 = copy_10 = None
        slice_scatter_19: "f32[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_64, slice_scatter_18, 1, 0, 256);  permute_64 = slice_scatter_18 = None
        permute_69: "f32[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_19, [0, 2, 1, 3]);  slice_scatter_19 = None
        view_79: "f32[8, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_69, [8, 4, 256, 513]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_3: "f32[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_1, [8, 256, 1, 257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_83: "f32[8, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_79, [8, 1, 1024, 513]);  view_79 = None
        permute_73: "f32[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        slice_123: "f32[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_73, 1, -256, 9223372036854775807)
        slice_124: "f32[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_123, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_4: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None
        where_6: "f32[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_4, full_default_5, slice_124);  convert_element_type_4 = full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_11: "f32[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_124, where_6);  slice_124 = where_6 = None
        slice_scatter_20: "f32[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_123, copy_11, 3, -257, 9223372036854775807);  slice_123 = copy_11 = None
        slice_scatter_21: "f32[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_73, slice_scatter_20, 1, -256, 9223372036854775807);  permute_73 = slice_scatter_20 = None
        permute_78: "f32[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_21, [0, 2, 1, 3]);  slice_scatter_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_81: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_52, [0, 2, 1, 3]);  permute_52 = None
        permute_82: "f32[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_78, [0, 2, 1, 3]);  permute_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_5: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_81, permute_82);  permute_81 = None
        permute_83: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_5, [0, 2, 1, 3]);  add_5 = None
        permute_84: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_83, [0, 2, 1, 3]);  permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_9: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        amax: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_9, [-1], True)
        sub_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_9, amax);  clone_9 = amax = None
        exp: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_17: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(primals_9, 2)
        unsqueeze_18: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        where_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_7)

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[36]" = torch.ops.prims.inductor_seeds.default(36, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_35: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_35, 1e-30);  inductor_random_default_35 = None
        mul_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt, where_7);  where_7 = None
        mul_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_1, 1.0);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_98: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_2, [1024, 8, 12, 64]);  add_2 = None
        permute_86: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_98, [1, 0, 2, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_87: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1, 3]);  mul_2 = None
        clone_10: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
        view_99: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_10, [96, 4, 256, 513]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_88: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_86, [0, 2, 1, 3]);  permute_86 = None
        view_100: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_88, [96, 1024, 64]);  permute_88 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_100, [0, 0, 256, 256], -1.0);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_8: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_2, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_2 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_99, [0, 257], 0.0);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_101: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_3, [96, 4, -1]);  constant_pad_nd_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_133: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_101, 2, 0, -256);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_102: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_133, [96, 4, 256, 769]);  slice_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_134: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_102, 3, 0, -1);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_19: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_134, 4);  slice_134 = None
        unsqueeze_20: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_8, 4);  as_strided_8 = None
        view_103: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_19, [384, 256, 768]);  unsqueeze_19 = None
        clone_11: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_20, memory_format = torch.contiguous_format);  unsqueeze_20 = None
        view_104: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_11, [384, 768, 64]);  clone_11 = None
        bmm_1: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_103, view_104)
        view_105: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_1, [96, 4, 256, 1, 64]);  bmm_1 = None
        permute_93: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_105, [0, 1, 2, 4, 3]);  view_105 = None
        view_106: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_93, [96, 4, 256, 64]);  permute_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_107: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_106, [8, 12, 1024, 64]);  view_106 = None
        permute_94: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_95: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_94, [1, 0, 2, 3]);  permute_94 = None
        clone_12: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_108: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_12, [1024, 8, 768]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_96: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_108, [1, 0, 2]);  view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        clone_13: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
        view_109: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_13, [8192, 768]);  clone_13 = None
        mm_3: "f32[8192, 768]" = torch.ops.aten.mm.default(view_109, permute_97);  permute_97 = None
        view_110: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_3, [8, 1024, 768]);  mm_3 = None
        add_7: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_110, primals_11);  view_110 = primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_34: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_34, 1e-30);  inductor_random_default_34 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_1, add_7);  add_7 = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_3, 1.0);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_8: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_4, primals_1);  mul_4 = primals_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_9: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_6: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_1);  add_8 = getitem_1 = None
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = None
        mul_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_12)
        add_10: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_6, primals_13);  mul_6 = primals_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_111: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_10, [8192, 768])
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_15, view_111, permute_98);  primals_15 = permute_98 = None
        view_112: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_7: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.5)
        mul_8: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.7071067811865476);  view_112 = None
        erf: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_11: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_11);  mul_7 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_113: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_9, [8192, 3072]);  mul_9 = None
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_1: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_17, view_113, permute_99);  primals_17 = permute_99 = None
        view_114: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_1, [8, 1024, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_33: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_33, 1e-30);  inductor_random_default_33 = None
        mul_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_2, view_114);  view_114 = None
        mul_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, 1.0);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_12: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_11, add_10);  mul_11 = add_10 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_13: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_7: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_3);  add_12 = getitem_3 = None
        mul_12: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_1);  sub_7 = None
        mul_13: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_12, primals_18)
        add_14: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_13, primals_19);  mul_13 = primals_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_100: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_14, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
        clone_14: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_115: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_14, [8192, 768]);  clone_14 = None
        mm_4: "f32[8192, 768]" = torch.ops.aten.mm.default(view_115, permute_101);  permute_101 = None
        view_116: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_4, [1024, 8, 768]);  mm_4 = None
        add_15: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_116, primals_21);  view_116 = primals_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0])
        mm_5: "f32[8192, 768]" = torch.ops.aten.mm.default(view_115, permute_102);  permute_102 = None
        view_118: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_5, [1024, 8, 768]);  mm_5 = None
        add_16: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_118, primals_23);  view_118 = primals_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0])
        mm_6: "f32[8192, 768]" = torch.ops.aten.mm.default(view_115, permute_103);  permute_103 = None
        view_120: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_6, [1024, 8, 768]);  mm_6 = None
        add_17: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_120, primals_25);  view_120 = primals_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_10: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_15, 8.0);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_123: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_16, [1024, 8, 12, 64]);  add_16 = None
        permute_105: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_123, [1, 0, 2, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_110: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_105, [0, 2, 1, 3]);  permute_105 = None
        view_127: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_110, [96, 1024, 64]);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_134: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_127, [96, 2, 512, 64]);  view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_10: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_134, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_23: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_10, 4);  as_strided_10 = None
        permute_120: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_23, [0, 1, 4, 2, 3]);  unsqueeze_23 = None
        view_144: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_10, [1024, 8, 12, 64]);  div_10 = None
        permute_125: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_144, [1, 0, 2, 3]);  view_144 = None
        permute_126: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_125, [0, 2, 1, 3]);  permute_125 = None
        view_145: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_126, [96, 1024, 64]);  permute_126 = None
        view_146: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_145, [96, 2, 512, 64]);  view_145 = None
        as_strided_14: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_146, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_146 = None
        unsqueeze_25: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_14, 4);  as_strided_14 = None
        clone_17: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_25, memory_format = torch.contiguous_format);  unsqueeze_25 = None
        view_147: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_17, [288, 512, 64]);  clone_17 = None
        permute_129: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_120, [0, 1, 4, 3, 2]);  permute_120 = None
        clone_18: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_148: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_18, [288, 64, 512]);  clone_18 = None
        bmm_2: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_147, view_148)
        view_149: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_2, [96, 3, 512, 1, 512]);  bmm_2 = None
        permute_130: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_149, [0, 1, 2, 4, 3]);  view_149 = None
        view_150: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_130, [96, 3, 512, 512]);  permute_130 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_4: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_150, [0, 0, 0, 1], 0.0);  view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_151: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_4, [96, 3, 512, 513]);  constant_pad_nd_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_135: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_151, 2, 0, 256)
        slice_136: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_135, 3, 0, 257);  slice_135 = None
        copy_12: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_136);  slice_136 = None
        slice_scatter_22: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_12, 3, 256, 9223372036854775807);  copy_12 = None
        slice_scatter_23: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_22, 1, 0, -1);  slice_scatter_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_31: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_151, 1, -1)
        slice_145: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_31, 1, 256, 9223372036854775807);  select_31 = None
        slice_146: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_145, 2, 0, 257);  slice_145 = None
        select_34: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_23, 1, -1)
        slice_148: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_34, 2, 256, 9223372036854775807)
        copy_13: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_148, slice_146);  slice_148 = slice_146 = None
        slice_scatter_24: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_34, copy_13, 2, 256, 9223372036854775807);  select_34 = copy_13 = None
        select_scatter_4: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_23, slice_scatter_24, 1, -1);  slice_scatter_23 = slice_scatter_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_151: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_151, 2, -257, -1)
        slice_152: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_151, 3, 257, 9223372036854775807);  slice_151 = None
        slice_156: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_4, 1, 1, 9223372036854775807)
        slice_157: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_156, 3, 0, 256)
        copy_14: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_157, slice_152);  slice_157 = slice_152 = None
        slice_scatter_25: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_156, copy_14, 3, 0, 256);  slice_156 = copy_14 = None
        slice_scatter_26: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_4, slice_scatter_25, 1, 1, 9223372036854775807);  select_scatter_4 = slice_scatter_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_39: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_151, 1, 0);  view_151 = None
        slice_164: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_39, 1, 0, 255);  select_39 = None
        slice_165: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_164, 2, -255, 9223372036854775807);  slice_164 = None
        select_43: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_26, 1, 0)
        slice_169: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_43, 1, 1, 256)
        slice_170: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_169, 2, 1, 256)
        copy_15: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_170, slice_165);  slice_170 = slice_165 = None
        slice_scatter_27: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_169, copy_15, 2, 1, 256);  slice_169 = copy_15 = None
        slice_scatter_28: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_43, slice_scatter_27, 1, 1, 256);  select_43 = slice_scatter_27 = None
        select_scatter_5: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_26, slice_scatter_28, 1, 0);  slice_scatter_26 = slice_scatter_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_156: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_5, [8, 12, 1024, 513]);  select_scatter_5 = None
        permute_134: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        slice_180: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_134, 1, 0, 256)
        slice_181: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_180, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_9: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_181)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_16: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_181, where_9);  slice_181 = where_9 = None
        slice_scatter_29: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_180, copy_16, 3, 0, 257);  slice_180 = copy_16 = None
        slice_scatter_30: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_134, slice_scatter_29, 1, 0, 256);  permute_134 = slice_scatter_29 = None
        permute_139: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_30, [0, 2, 1, 3]);  slice_scatter_30 = None
        view_160: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_139, [96, 4, 256, 513]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_170: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_160, [8, 12, 1024, 513]);  view_160 = None
        permute_147: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        slice_196: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_147, 1, -256, 9223372036854775807)
        slice_197: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_196, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_10: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_197)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_17: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_197, where_10);  slice_197 = where_10 = None
        slice_scatter_31: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_196, copy_17, 3, -257, 9223372036854775807);  slice_196 = copy_17 = None
        slice_scatter_32: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_147, slice_scatter_31, 1, -256, 9223372036854775807);  permute_147 = slice_scatter_31 = None
        permute_152: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_32, [0, 2, 1, 3]);  slice_scatter_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_181: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_152, [0, 2, 1, 3]);  permute_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_20: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_181, permute_82);  permute_181 = None
        permute_183: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_20, [0, 2, 1, 3]);  add_20 = None
        permute_184: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_183, [0, 2, 1, 3]);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_23: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        amax_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_23, [-1], True)
        sub_12: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_23, amax_1);  clone_23 = amax_1 = None
        exp_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_2: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_17: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_15: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_32: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 1e-30);  inductor_random_default_32 = None
        mul_15: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_3, where_15);  where_15 = None
        mul_16: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_15, 1.0);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_213: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_17, [1024, 8, 12, 64]);  add_17 = None
        permute_186: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_213, [1, 0, 2, 3]);  view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_187: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_16, [0, 2, 1, 3]);  mul_16 = None
        clone_24: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_214: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_24, [96, 4, 256, 513]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_188: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_186, [0, 2, 1, 3]);  permute_186 = None
        view_215: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_188, [96, 1024, 64]);  permute_188 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_6: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_215, [0, 0, 256, 256], -1.0);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_17: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_6, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_6 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_7: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_214, [0, 257], 0.0);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_216: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_7, [96, 4, -1]);  constant_pad_nd_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_267: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_216, 2, 0, -256);  view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_217: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_267, [96, 4, 256, 769]);  slice_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_268: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_217, 3, 0, -1);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_40: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_268, 4);  slice_268 = None
        unsqueeze_41: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_17, 4);  as_strided_17 = None
        view_218: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_40, [384, 256, 768]);  unsqueeze_40 = None
        clone_25: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_41, memory_format = torch.contiguous_format);  unsqueeze_41 = None
        view_219: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_25, [384, 768, 64]);  clone_25 = None
        bmm_3: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_218, view_219)
        view_220: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_3, [96, 4, 256, 1, 64]);  bmm_3 = None
        permute_193: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_220, [0, 1, 2, 4, 3]);  view_220 = None
        view_221: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_193, [96, 4, 256, 64]);  permute_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_222: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_221, [8, 12, 1024, 64]);  view_221 = None
        permute_194: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_195: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_194, [1, 0, 2, 3]);  permute_194 = None
        clone_26: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_223: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_26, [1024, 8, 768]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_196: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_223, [1, 0, 2]);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0])
        clone_27: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
        view_224: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_27, [8192, 768]);  clone_27 = None
        mm_7: "f32[8192, 768]" = torch.ops.aten.mm.default(view_224, permute_197);  permute_197 = None
        view_225: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_7, [8, 1024, 768]);  mm_7 = None
        add_22: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_225, primals_27);  view_225 = primals_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_31: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_31, 1e-30);  inductor_random_default_31 = None
        mul_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_4, add_22);  add_22 = None
        mul_18: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_17, 1.0);  mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_18, add_14);  mul_18 = add_14 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 1024, 1]" = var_mean_2[0]
        getitem_5: "f32[8, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        add_24: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_14: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_5);  add_23 = getitem_5 = None
        mul_19: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_2);  sub_14 = None
        mul_20: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_19, primals_28)
        add_25: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_20, primals_29);  mul_20 = primals_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_226: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_25, [8192, 768])
        permute_198: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        addmm_2: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_31, view_226, permute_198);  primals_31 = permute_198 = None
        view_227: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_2, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_21: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_227, 0.5)
        mul_22: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_227, 0.7071067811865476);  view_227 = None
        erf_1: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
        add_26: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_23: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_21, add_26);  mul_21 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_228: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_23, [8192, 3072]);  mul_23 = None
        permute_199: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_32, [1, 0])
        addmm_3: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_33, view_228, permute_199);  primals_33 = permute_199 = None
        view_229: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_3, [8, 1024, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_30: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_30, 1e-30);  inductor_random_default_30 = None
        mul_24: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_5, view_229);  view_229 = None
        mul_25: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, 1.0);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_25, add_25);  mul_25 = add_25 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 1024, 1]" = var_mean_3[0]
        getitem_7: "f32[8, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        add_28: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_15: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_7);  add_27 = getitem_7 = None
        mul_26: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_3);  sub_15 = None
        mul_27: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_34)
        add_29: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_35);  mul_27 = primals_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_200: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_29, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0])
        clone_28: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_230: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_28, [8192, 768]);  clone_28 = None
        mm_8: "f32[8192, 768]" = torch.ops.aten.mm.default(view_230, permute_201);  permute_201 = None
        view_231: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_8, [1024, 8, 768]);  mm_8 = None
        add_30: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_231, primals_37);  view_231 = primals_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0])
        mm_9: "f32[8192, 768]" = torch.ops.aten.mm.default(view_230, permute_202);  permute_202 = None
        view_233: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_9, [1024, 8, 768]);  mm_9 = None
        add_31: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_233, primals_39);  view_233 = primals_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0])
        mm_10: "f32[8192, 768]" = torch.ops.aten.mm.default(view_230, permute_203);  permute_203 = None
        view_235: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_10, [1024, 8, 768]);  mm_10 = None
        add_32: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_235, primals_41);  view_235 = primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_20: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_30, 8.0);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_238: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_31, [1024, 8, 12, 64]);  add_31 = None
        permute_205: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_238, [1, 0, 2, 3]);  view_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_210: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_205, [0, 2, 1, 3]);  permute_205 = None
        view_242: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_210, [96, 1024, 64]);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_249: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_242, [96, 2, 512, 64]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_19: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_249, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_44: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_19, 4);  as_strided_19 = None
        permute_220: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_44, [0, 1, 4, 2, 3]);  unsqueeze_44 = None
        view_259: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_20, [1024, 8, 12, 64]);  div_20 = None
        permute_225: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_259, [1, 0, 2, 3]);  view_259 = None
        permute_226: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
        view_260: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_226, [96, 1024, 64]);  permute_226 = None
        view_261: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_260, [96, 2, 512, 64]);  view_260 = None
        as_strided_23: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_261, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_261 = None
        unsqueeze_46: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_23, 4);  as_strided_23 = None
        clone_31: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_46, memory_format = torch.contiguous_format);  unsqueeze_46 = None
        view_262: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_31, [288, 512, 64]);  clone_31 = None
        permute_229: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_220, [0, 1, 4, 3, 2]);  permute_220 = None
        clone_32: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
        view_263: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_32, [288, 64, 512]);  clone_32 = None
        bmm_4: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_262, view_263)
        view_264: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_4, [96, 3, 512, 1, 512]);  bmm_4 = None
        permute_230: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_264, [0, 1, 2, 4, 3]);  view_264 = None
        view_265: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_230, [96, 3, 512, 512]);  permute_230 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_8: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_265, [0, 0, 0, 1], 0.0);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_266: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_8, [96, 3, 512, 513]);  constant_pad_nd_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_269: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_266, 2, 0, 256)
        slice_270: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_269, 3, 0, 257);  slice_269 = None
        copy_24: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_270);  slice_270 = None
        slice_scatter_44: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_24, 3, 256, 9223372036854775807);  copy_24 = None
        slice_scatter_45: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_44, 1, 0, -1);  slice_scatter_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_62: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_266, 1, -1)
        slice_279: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_62, 1, 256, 9223372036854775807);  select_62 = None
        slice_280: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_279, 2, 0, 257);  slice_279 = None
        select_65: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_45, 1, -1)
        slice_282: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_65, 2, 256, 9223372036854775807)
        copy_25: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_282, slice_280);  slice_282 = slice_280 = None
        slice_scatter_46: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_65, copy_25, 2, 256, 9223372036854775807);  select_65 = copy_25 = None
        select_scatter_8: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_45, slice_scatter_46, 1, -1);  slice_scatter_45 = slice_scatter_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_285: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_266, 2, -257, -1)
        slice_286: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_285, 3, 257, 9223372036854775807);  slice_285 = None
        slice_290: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_8, 1, 1, 9223372036854775807)
        slice_291: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_290, 3, 0, 256)
        copy_26: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_291, slice_286);  slice_291 = slice_286 = None
        slice_scatter_47: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_290, copy_26, 3, 0, 256);  slice_290 = copy_26 = None
        slice_scatter_48: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_8, slice_scatter_47, 1, 1, 9223372036854775807);  select_scatter_8 = slice_scatter_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_70: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_266, 1, 0);  view_266 = None
        slice_298: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_70, 1, 0, 255);  select_70 = None
        slice_299: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_298, 2, -255, 9223372036854775807);  slice_298 = None
        select_74: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_48, 1, 0)
        slice_303: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_74, 1, 1, 256)
        slice_304: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_303, 2, 1, 256)
        copy_27: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_304, slice_299);  slice_304 = slice_299 = None
        slice_scatter_49: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_303, copy_27, 2, 1, 256);  slice_303 = copy_27 = None
        slice_scatter_50: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_74, slice_scatter_49, 1, 1, 256);  select_74 = slice_scatter_49 = None
        select_scatter_9: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_48, slice_scatter_50, 1, 0);  slice_scatter_48 = slice_scatter_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_271: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_9, [8, 12, 1024, 513]);  select_scatter_9 = None
        permute_234: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
        slice_314: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_234, 1, 0, 256)
        slice_315: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_314, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_17: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_315)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_28: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_315, where_17);  slice_315 = where_17 = None
        slice_scatter_51: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_314, copy_28, 3, 0, 257);  slice_314 = copy_28 = None
        slice_scatter_52: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_234, slice_scatter_51, 1, 0, 256);  permute_234 = slice_scatter_51 = None
        permute_239: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_52, [0, 2, 1, 3]);  slice_scatter_52 = None
        view_275: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_239, [96, 4, 256, 513]);  permute_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_285: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_275, [8, 12, 1024, 513]);  view_275 = None
        permute_247: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
        slice_330: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_247, 1, -256, 9223372036854775807)
        slice_331: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_330, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_18: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_331)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_29: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_331, where_18);  slice_331 = where_18 = None
        slice_scatter_53: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_330, copy_29, 3, -257, 9223372036854775807);  slice_330 = copy_29 = None
        slice_scatter_54: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_247, slice_scatter_53, 1, -256, 9223372036854775807);  permute_247 = slice_scatter_53 = None
        permute_252: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_54, [0, 2, 1, 3]);  slice_scatter_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_281: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_252, [0, 2, 1, 3]);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_35: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_281, permute_82);  permute_281 = None
        permute_283: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_35, [0, 2, 1, 3]);  add_35 = None
        permute_284: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_283, [0, 2, 1, 3]);  permute_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_37: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
        amax_2: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_37, [-1], True)
        sub_20: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_37, amax_2);  clone_37 = amax_2 = None
        exp_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_3: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_27: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_23: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_29: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_29, 1e-30);  inductor_random_default_29 = None
        mul_29: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_6, where_23);  where_23 = None
        mul_30: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_29, 1.0);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_328: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_32, [1024, 8, 12, 64]);  add_32 = None
        permute_286: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_328, [1, 0, 2, 3]);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_287: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_30, [0, 2, 1, 3]);  mul_30 = None
        clone_38: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_329: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_38, [96, 4, 256, 513]);  clone_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_288: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_286, [0, 2, 1, 3]);  permute_286 = None
        view_330: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_288, [96, 1024, 64]);  permute_288 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_10: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_330, [0, 0, 256, 256], -1.0);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_26: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_10, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_10 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_11: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_329, [0, 257], 0.0);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_331: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_11, [96, 4, -1]);  constant_pad_nd_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_401: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_331, 2, 0, -256);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_332: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_401, [96, 4, 256, 769]);  slice_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_402: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_332, 3, 0, -1);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_61: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_402, 4);  slice_402 = None
        unsqueeze_62: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_26, 4);  as_strided_26 = None
        view_333: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_61, [384, 256, 768]);  unsqueeze_61 = None
        clone_39: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_62, memory_format = torch.contiguous_format);  unsqueeze_62 = None
        view_334: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_39, [384, 768, 64]);  clone_39 = None
        bmm_5: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_333, view_334)
        view_335: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_5, [96, 4, 256, 1, 64]);  bmm_5 = None
        permute_293: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_335, [0, 1, 2, 4, 3]);  view_335 = None
        view_336: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_293, [96, 4, 256, 64]);  permute_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_337: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_336, [8, 12, 1024, 64]);  view_336 = None
        permute_294: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_295: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_294, [1, 0, 2, 3]);  permute_294 = None
        clone_40: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
        view_338: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_40, [1024, 8, 768]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_296: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_338, [1, 0, 2]);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0])
        clone_41: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
        view_339: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_41, [8192, 768]);  clone_41 = None
        mm_11: "f32[8192, 768]" = torch.ops.aten.mm.default(view_339, permute_297);  permute_297 = None
        view_340: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_11, [8, 1024, 768]);  mm_11 = None
        add_37: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_340, primals_43);  view_340 = primals_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_28: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_28, 1e-30);  inductor_random_default_28 = None
        mul_31: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_7, add_37);  add_37 = None
        mul_32: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_31, 1.0);  mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_38: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_32, add_29);  mul_32 = add_29 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 1024, 1]" = var_mean_4[0]
        getitem_9: "f32[8, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        add_39: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_22: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_9);  add_38 = getitem_9 = None
        mul_33: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_4);  sub_22 = None
        mul_34: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_44)
        add_40: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_45);  mul_34 = primals_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_341: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_40, [8192, 768])
        permute_298: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_46, [1, 0])
        addmm_4: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_47, view_341, permute_298);  primals_47 = permute_298 = None
        view_342: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_4, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_35: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_342, 0.5)
        mul_36: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_342, 0.7071067811865476);  view_342 = None
        erf_2: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_41: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_37: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_35, add_41);  mul_35 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_343: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_37, [8192, 3072]);  mul_37 = None
        permute_299: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_48, [1, 0])
        addmm_5: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_49, view_343, permute_299);  primals_49 = permute_299 = None
        view_344: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_5, [8, 1024, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_27: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_27, 1e-30);  inductor_random_default_27 = None
        mul_38: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_8, view_344);  view_344 = None
        mul_39: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_38, 1.0);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_42: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_39, add_40);  mul_39 = add_40 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 1024, 1]" = var_mean_5[0]
        getitem_11: "f32[8, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        add_43: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_23: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_11);  add_42 = getitem_11 = None
        mul_40: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_5);  sub_23 = None
        mul_41: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_50)
        add_44: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_51);  mul_41 = primals_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_300: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_44, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0])
        clone_42: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
        view_345: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_42, [8192, 768]);  clone_42 = None
        mm_12: "f32[8192, 768]" = torch.ops.aten.mm.default(view_345, permute_301);  permute_301 = None
        view_346: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_12, [1024, 8, 768]);  mm_12 = None
        add_45: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_346, primals_53);  view_346 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0])
        mm_13: "f32[8192, 768]" = torch.ops.aten.mm.default(view_345, permute_302);  permute_302 = None
        view_348: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_13, [1024, 8, 768]);  mm_13 = None
        add_46: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_348, primals_55);  view_348 = primals_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0])
        mm_14: "f32[8192, 768]" = torch.ops.aten.mm.default(view_345, permute_303);  permute_303 = None
        view_350: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_14, [1024, 8, 768]);  mm_14 = None
        add_47: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_350, primals_57);  view_350 = primals_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_30: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_45, 8.0);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_353: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_46, [1024, 8, 12, 64]);  add_46 = None
        permute_305: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_353, [1, 0, 2, 3]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_310: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_305, [0, 2, 1, 3]);  permute_305 = None
        view_357: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_310, [96, 1024, 64]);  permute_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_364: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_357, [96, 2, 512, 64]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_28: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_364, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_65: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_28, 4);  as_strided_28 = None
        permute_320: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_65, [0, 1, 4, 2, 3]);  unsqueeze_65 = None
        view_374: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_30, [1024, 8, 12, 64]);  div_30 = None
        permute_325: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_374, [1, 0, 2, 3]);  view_374 = None
        permute_326: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_325, [0, 2, 1, 3]);  permute_325 = None
        view_375: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_326, [96, 1024, 64]);  permute_326 = None
        view_376: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_375, [96, 2, 512, 64]);  view_375 = None
        as_strided_32: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_376, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_376 = None
        unsqueeze_67: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_32, 4);  as_strided_32 = None
        clone_45: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_67, memory_format = torch.contiguous_format);  unsqueeze_67 = None
        view_377: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_45, [288, 512, 64]);  clone_45 = None
        permute_329: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_320, [0, 1, 4, 3, 2]);  permute_320 = None
        clone_46: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
        view_378: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_46, [288, 64, 512]);  clone_46 = None
        bmm_6: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_377, view_378)
        view_379: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_6, [96, 3, 512, 1, 512]);  bmm_6 = None
        permute_330: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_379, [0, 1, 2, 4, 3]);  view_379 = None
        view_380: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_330, [96, 3, 512, 512]);  permute_330 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_12: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_380, [0, 0, 0, 1], 0.0);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_381: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_12, [96, 3, 512, 513]);  constant_pad_nd_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_403: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_381, 2, 0, 256)
        slice_404: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_403, 3, 0, 257);  slice_403 = None
        copy_36: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_404);  slice_404 = None
        slice_scatter_66: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_36, 3, 256, 9223372036854775807);  copy_36 = None
        slice_scatter_67: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_66, 1, 0, -1);  slice_scatter_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_93: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_381, 1, -1)
        slice_413: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_93, 1, 256, 9223372036854775807);  select_93 = None
        slice_414: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_413, 2, 0, 257);  slice_413 = None
        select_96: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_67, 1, -1)
        slice_416: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_96, 2, 256, 9223372036854775807)
        copy_37: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_416, slice_414);  slice_416 = slice_414 = None
        slice_scatter_68: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_96, copy_37, 2, 256, 9223372036854775807);  select_96 = copy_37 = None
        select_scatter_12: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_67, slice_scatter_68, 1, -1);  slice_scatter_67 = slice_scatter_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_419: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_381, 2, -257, -1)
        slice_420: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_419, 3, 257, 9223372036854775807);  slice_419 = None
        slice_424: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_12, 1, 1, 9223372036854775807)
        slice_425: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_424, 3, 0, 256)
        copy_38: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_425, slice_420);  slice_425 = slice_420 = None
        slice_scatter_69: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_424, copy_38, 3, 0, 256);  slice_424 = copy_38 = None
        slice_scatter_70: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_12, slice_scatter_69, 1, 1, 9223372036854775807);  select_scatter_12 = slice_scatter_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_101: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_381, 1, 0);  view_381 = None
        slice_432: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_101, 1, 0, 255);  select_101 = None
        slice_433: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_432, 2, -255, 9223372036854775807);  slice_432 = None
        select_105: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_70, 1, 0)
        slice_437: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_105, 1, 1, 256)
        slice_438: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_437, 2, 1, 256)
        copy_39: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_438, slice_433);  slice_438 = slice_433 = None
        slice_scatter_71: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_437, copy_39, 2, 1, 256);  slice_437 = copy_39 = None
        slice_scatter_72: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_105, slice_scatter_71, 1, 1, 256);  select_105 = slice_scatter_71 = None
        select_scatter_13: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_70, slice_scatter_72, 1, 0);  slice_scatter_70 = slice_scatter_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_386: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_13, [8, 12, 1024, 513]);  select_scatter_13 = None
        permute_334: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
        slice_448: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_334, 1, 0, 256)
        slice_449: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_448, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_25: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_449)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_40: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_449, where_25);  slice_449 = where_25 = None
        slice_scatter_73: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_448, copy_40, 3, 0, 257);  slice_448 = copy_40 = None
        slice_scatter_74: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_334, slice_scatter_73, 1, 0, 256);  permute_334 = slice_scatter_73 = None
        permute_339: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_74, [0, 2, 1, 3]);  slice_scatter_74 = None
        view_390: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_339, [96, 4, 256, 513]);  permute_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_400: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_390, [8, 12, 1024, 513]);  view_390 = None
        permute_347: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        slice_464: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_347, 1, -256, 9223372036854775807)
        slice_465: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_464, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_26: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_465)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_41: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_465, where_26);  slice_465 = where_26 = None
        slice_scatter_75: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_464, copy_41, 3, -257, 9223372036854775807);  slice_464 = copy_41 = None
        slice_scatter_76: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_347, slice_scatter_75, 1, -256, 9223372036854775807);  permute_347 = slice_scatter_75 = None
        permute_352: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_76, [0, 2, 1, 3]);  slice_scatter_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_381: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_352, [0, 2, 1, 3]);  permute_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_50: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_381, permute_82);  permute_381 = None
        permute_383: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_50, [0, 2, 1, 3]);  add_50 = None
        permute_384: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_383, [0, 2, 1, 3]);  permute_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_51: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
        amax_3: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_51, [-1], True)
        sub_28: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_51, amax_3);  clone_51 = amax_3 = None
        exp_3: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_4: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_37: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_31: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_37)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_26: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 1e-30);  inductor_random_default_26 = None
        mul_43: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_9, where_31);  where_31 = None
        mul_44: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_43, 1.0);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_443: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_47, [1024, 8, 12, 64]);  add_47 = None
        permute_386: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_443, [1, 0, 2, 3]);  view_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_387: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_44, [0, 2, 1, 3]);  mul_44 = None
        clone_52: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
        view_444: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_52, [96, 4, 256, 513]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_388: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_386, [0, 2, 1, 3]);  permute_386 = None
        view_445: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_388, [96, 1024, 64]);  permute_388 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_14: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_445, [0, 0, 256, 256], -1.0);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_35: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_14, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_14 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_15: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_444, [0, 257], 0.0);  view_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_446: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_15, [96, 4, -1]);  constant_pad_nd_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_535: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_446, 2, 0, -256);  view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_447: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_535, [96, 4, 256, 769]);  slice_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_536: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_447, 3, 0, -1);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_82: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_536, 4);  slice_536 = None
        unsqueeze_83: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_35, 4);  as_strided_35 = None
        view_448: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_82, [384, 256, 768]);  unsqueeze_82 = None
        clone_53: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_83, memory_format = torch.contiguous_format);  unsqueeze_83 = None
        view_449: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_53, [384, 768, 64]);  clone_53 = None
        bmm_7: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_448, view_449)
        view_450: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_7, [96, 4, 256, 1, 64]);  bmm_7 = None
        permute_393: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_450, [0, 1, 2, 4, 3]);  view_450 = None
        view_451: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_393, [96, 4, 256, 64]);  permute_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_452: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_451, [8, 12, 1024, 64]);  view_451 = None
        permute_394: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_395: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_394, [1, 0, 2, 3]);  permute_394 = None
        clone_54: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
        view_453: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_54, [1024, 8, 768]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_396: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_453, [1, 0, 2]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0])
        clone_55: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_396, memory_format = torch.contiguous_format);  permute_396 = None
        view_454: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_55, [8192, 768]);  clone_55 = None
        mm_15: "f32[8192, 768]" = torch.ops.aten.mm.default(view_454, permute_397);  permute_397 = None
        view_455: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_15, [8, 1024, 768]);  mm_15 = None
        add_52: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_455, primals_59);  view_455 = primals_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_25: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 1e-30);  inductor_random_default_25 = None
        mul_45: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_10, add_52);  add_52 = None
        mul_46: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_45, 1.0);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_53: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_46, add_44);  mul_46 = add_44 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_12: "f32[8, 1024, 1]" = var_mean_6[0]
        getitem_13: "f32[8, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        add_54: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_30: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_53, getitem_13);  add_53 = getitem_13 = None
        mul_47: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_6);  sub_30 = None
        mul_48: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_47, primals_60)
        add_55: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_48, primals_61);  mul_48 = primals_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_456: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_55, [8192, 768])
        permute_398: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_62, [1, 0])
        addmm_6: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_63, view_456, permute_398);  primals_63 = permute_398 = None
        view_457: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_6, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_49: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_457, 0.5)
        mul_50: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_457, 0.7071067811865476);  view_457 = None
        erf_3: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_56: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_51: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_49, add_56);  mul_49 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_458: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_51, [8192, 3072]);  mul_51 = None
        permute_399: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_64, [1, 0])
        addmm_7: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_65, view_458, permute_399);  primals_65 = permute_399 = None
        view_459: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_7, [8, 1024, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_24: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 1e-30);  inductor_random_default_24 = None
        mul_52: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_11, view_459);  view_459 = None
        mul_53: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_52, 1.0);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_53, add_55);  mul_53 = add_55 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 1024, 1]" = var_mean_7[0]
        getitem_15: "f32[8, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        add_58: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_31: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_15);  add_57 = getitem_15 = None
        mul_54: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_7);  sub_31 = None
        mul_55: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_54, primals_66)
        add_59: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_55, primals_67);  mul_55 = primals_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_400: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_59, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0])
        clone_56: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_400, memory_format = torch.contiguous_format);  permute_400 = None
        view_460: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_56, [8192, 768]);  clone_56 = None
        mm_16: "f32[8192, 768]" = torch.ops.aten.mm.default(view_460, permute_401);  permute_401 = None
        view_461: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_16, [1024, 8, 768]);  mm_16 = None
        add_60: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_461, primals_69);  view_461 = primals_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0])
        mm_17: "f32[8192, 768]" = torch.ops.aten.mm.default(view_460, permute_402);  permute_402 = None
        view_463: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_17, [1024, 8, 768]);  mm_17 = None
        add_61: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_463, primals_71);  view_463 = primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_403: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0])
        mm_18: "f32[8192, 768]" = torch.ops.aten.mm.default(view_460, permute_403);  permute_403 = None
        view_465: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_18, [1024, 8, 768]);  mm_18 = None
        add_62: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_465, primals_73);  view_465 = primals_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_40: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_60, 8.0);  add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_468: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_61, [1024, 8, 12, 64]);  add_61 = None
        permute_405: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_468, [1, 0, 2, 3]);  view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_410: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_405, [0, 2, 1, 3]);  permute_405 = None
        view_472: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_410, [96, 1024, 64]);  permute_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_479: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_472, [96, 2, 512, 64]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_37: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_479, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_86: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_37, 4);  as_strided_37 = None
        permute_420: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_86, [0, 1, 4, 2, 3]);  unsqueeze_86 = None
        view_489: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_40, [1024, 8, 12, 64]);  div_40 = None
        permute_425: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_489, [1, 0, 2, 3]);  view_489 = None
        permute_426: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_425, [0, 2, 1, 3]);  permute_425 = None
        view_490: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_426, [96, 1024, 64]);  permute_426 = None
        view_491: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_490, [96, 2, 512, 64]);  view_490 = None
        as_strided_41: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_491, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_491 = None
        unsqueeze_88: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_41, 4);  as_strided_41 = None
        clone_59: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_88, memory_format = torch.contiguous_format);  unsqueeze_88 = None
        view_492: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_59, [288, 512, 64]);  clone_59 = None
        permute_429: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_420, [0, 1, 4, 3, 2]);  permute_420 = None
        clone_60: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_429, memory_format = torch.contiguous_format);  permute_429 = None
        view_493: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_60, [288, 64, 512]);  clone_60 = None
        bmm_8: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_492, view_493)
        view_494: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_8, [96, 3, 512, 1, 512]);  bmm_8 = None
        permute_430: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_494, [0, 1, 2, 4, 3]);  view_494 = None
        view_495: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_430, [96, 3, 512, 512]);  permute_430 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_16: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_495, [0, 0, 0, 1], 0.0);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_496: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_16, [96, 3, 512, 513]);  constant_pad_nd_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_537: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_496, 2, 0, 256)
        slice_538: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_537, 3, 0, 257);  slice_537 = None
        copy_48: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_538);  slice_538 = None
        slice_scatter_88: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_48, 3, 256, 9223372036854775807);  copy_48 = None
        slice_scatter_89: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_88, 1, 0, -1);  slice_scatter_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_124: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_496, 1, -1)
        slice_547: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_124, 1, 256, 9223372036854775807);  select_124 = None
        slice_548: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_547, 2, 0, 257);  slice_547 = None
        select_127: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_89, 1, -1)
        slice_550: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_127, 2, 256, 9223372036854775807)
        copy_49: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_550, slice_548);  slice_550 = slice_548 = None
        slice_scatter_90: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_127, copy_49, 2, 256, 9223372036854775807);  select_127 = copy_49 = None
        select_scatter_16: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_89, slice_scatter_90, 1, -1);  slice_scatter_89 = slice_scatter_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_553: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_496, 2, -257, -1)
        slice_554: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_553, 3, 257, 9223372036854775807);  slice_553 = None
        slice_558: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_16, 1, 1, 9223372036854775807)
        slice_559: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_558, 3, 0, 256)
        copy_50: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_559, slice_554);  slice_559 = slice_554 = None
        slice_scatter_91: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_558, copy_50, 3, 0, 256);  slice_558 = copy_50 = None
        slice_scatter_92: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_16, slice_scatter_91, 1, 1, 9223372036854775807);  select_scatter_16 = slice_scatter_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_132: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_496, 1, 0);  view_496 = None
        slice_566: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_132, 1, 0, 255);  select_132 = None
        slice_567: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_566, 2, -255, 9223372036854775807);  slice_566 = None
        select_136: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_92, 1, 0)
        slice_571: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_136, 1, 1, 256)
        slice_572: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_571, 2, 1, 256)
        copy_51: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_572, slice_567);  slice_572 = slice_567 = None
        slice_scatter_93: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_571, copy_51, 2, 1, 256);  slice_571 = copy_51 = None
        slice_scatter_94: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_136, slice_scatter_93, 1, 1, 256);  select_136 = slice_scatter_93 = None
        select_scatter_17: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_92, slice_scatter_94, 1, 0);  slice_scatter_92 = slice_scatter_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_501: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_17, [8, 12, 1024, 513]);  select_scatter_17 = None
        permute_434: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
        slice_582: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_434, 1, 0, 256)
        slice_583: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_582, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_33: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_583)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_52: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_583, where_33);  slice_583 = where_33 = None
        slice_scatter_95: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_582, copy_52, 3, 0, 257);  slice_582 = copy_52 = None
        slice_scatter_96: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_434, slice_scatter_95, 1, 0, 256);  permute_434 = slice_scatter_95 = None
        permute_439: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_96, [0, 2, 1, 3]);  slice_scatter_96 = None
        view_505: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_439, [96, 4, 256, 513]);  permute_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_515: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_505, [8, 12, 1024, 513]);  view_505 = None
        permute_447: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
        slice_598: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_447, 1, -256, 9223372036854775807)
        slice_599: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_598, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_34: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_599)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_53: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_599, where_34);  slice_599 = where_34 = None
        slice_scatter_97: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_598, copy_53, 3, -257, 9223372036854775807);  slice_598 = copy_53 = None
        slice_scatter_98: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_447, slice_scatter_97, 1, -256, 9223372036854775807);  permute_447 = slice_scatter_97 = None
        permute_452: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_98, [0, 2, 1, 3]);  slice_scatter_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_481: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_452, [0, 2, 1, 3]);  permute_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_65: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_481, permute_82);  permute_481 = None
        permute_483: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_65, [0, 2, 1, 3]);  add_65 = None
        permute_484: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_483, [0, 2, 1, 3]);  permute_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_65: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        amax_4: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_65, [-1], True)
        sub_36: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_65, amax_4);  clone_65 = amax_4 = None
        exp_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_5: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_47: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_39: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_47)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_23: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 1e-30);  inductor_random_default_23 = None
        mul_57: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_12, where_39);  where_39 = None
        mul_58: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_57, 1.0);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_558: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_62, [1024, 8, 12, 64]);  add_62 = None
        permute_486: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_558, [1, 0, 2, 3]);  view_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_487: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_58, [0, 2, 1, 3]);  mul_58 = None
        clone_66: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
        view_559: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_66, [96, 4, 256, 513]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_488: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_486, [0, 2, 1, 3]);  permute_486 = None
        view_560: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_488, [96, 1024, 64]);  permute_488 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_18: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_560, [0, 0, 256, 256], -1.0);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_44: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_18, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_18 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_19: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_559, [0, 257], 0.0);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_561: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_19, [96, 4, -1]);  constant_pad_nd_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_669: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_561, 2, 0, -256);  view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_562: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_669, [96, 4, 256, 769]);  slice_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_670: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_562, 3, 0, -1);  view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_103: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_670, 4);  slice_670 = None
        unsqueeze_104: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_44, 4);  as_strided_44 = None
        view_563: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_103, [384, 256, 768]);  unsqueeze_103 = None
        clone_67: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_104, memory_format = torch.contiguous_format);  unsqueeze_104 = None
        view_564: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_67, [384, 768, 64]);  clone_67 = None
        bmm_9: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_563, view_564)
        view_565: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_9, [96, 4, 256, 1, 64]);  bmm_9 = None
        permute_493: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_565, [0, 1, 2, 4, 3]);  view_565 = None
        view_566: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_493, [96, 4, 256, 64]);  permute_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_567: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_566, [8, 12, 1024, 64]);  view_566 = None
        permute_494: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_567, [0, 2, 1, 3]);  view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_495: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_494, [1, 0, 2, 3]);  permute_494 = None
        clone_68: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_495, memory_format = torch.contiguous_format);  permute_495 = None
        view_568: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_68, [1024, 8, 768]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_496: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_568, [1, 0, 2]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_497: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0])
        clone_69: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_496, memory_format = torch.contiguous_format);  permute_496 = None
        view_569: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_69, [8192, 768]);  clone_69 = None
        mm_19: "f32[8192, 768]" = torch.ops.aten.mm.default(view_569, permute_497);  permute_497 = None
        view_570: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_19, [8, 1024, 768]);  mm_19 = None
        add_67: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_570, primals_75);  view_570 = primals_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_22: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_13: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 1e-30);  inductor_random_default_22 = None
        mul_59: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_13, add_67);  add_67 = None
        mul_60: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_59, 1.0);  mul_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_68: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_60, add_59);  mul_60 = add_59 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 1024, 1]" = var_mean_8[0]
        getitem_17: "f32[8, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        add_69: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_38: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_17);  add_68 = getitem_17 = None
        mul_61: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_8);  sub_38 = None
        mul_62: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_61, primals_76)
        add_70: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_62, primals_77);  mul_62 = primals_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_571: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_70, [8192, 768])
        permute_498: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_78, [1, 0])
        addmm_8: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_79, view_571, permute_498);  primals_79 = permute_498 = None
        view_572: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_8, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_63: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_572, 0.5)
        mul_64: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_572, 0.7071067811865476);  view_572 = None
        erf_4: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_71: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_65: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_63, add_71);  mul_63 = add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_573: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_65, [8192, 3072]);  mul_65 = None
        permute_499: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0])
        addmm_9: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_81, view_573, permute_499);  primals_81 = permute_499 = None
        view_574: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_9, [8, 1024, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_21: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 1e-30);  inductor_random_default_21 = None
        mul_66: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_14, view_574);  view_574 = None
        mul_67: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, 1.0);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_72: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_67, add_70);  mul_67 = add_70 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 1024, 1]" = var_mean_9[0]
        getitem_19: "f32[8, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        add_73: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_39: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_19);  add_72 = getitem_19 = None
        mul_68: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_9);  sub_39 = None
        mul_69: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_68, primals_82)
        add_74: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_69, primals_83);  mul_69 = primals_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_500: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_74, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(primals_84, [1, 0])
        clone_70: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_575: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_70, [8192, 768]);  clone_70 = None
        mm_20: "f32[8192, 768]" = torch.ops.aten.mm.default(view_575, permute_501);  permute_501 = None
        view_576: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_20, [1024, 8, 768]);  mm_20 = None
        add_75: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_576, primals_85);  view_576 = primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_502: "f32[768, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0])
        mm_21: "f32[8192, 768]" = torch.ops.aten.mm.default(view_575, permute_502);  permute_502 = None
        view_578: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_21, [1024, 8, 768]);  mm_21 = None
        add_76: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_578, primals_87);  view_578 = primals_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_503: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0])
        mm_22: "f32[8192, 768]" = torch.ops.aten.mm.default(view_575, permute_503);  permute_503 = None
        view_580: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_22, [1024, 8, 768]);  mm_22 = None
        add_77: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_580, primals_89);  view_580 = primals_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_50: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_75, 8.0);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_583: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_76, [1024, 8, 12, 64]);  add_76 = None
        permute_505: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_583, [1, 0, 2, 3]);  view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_510: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_505, [0, 2, 1, 3]);  permute_505 = None
        view_587: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_510, [96, 1024, 64]);  permute_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_594: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_587, [96, 2, 512, 64]);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_46: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_594, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_107: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_46, 4);  as_strided_46 = None
        permute_520: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_107, [0, 1, 4, 2, 3]);  unsqueeze_107 = None
        view_604: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_50, [1024, 8, 12, 64]);  div_50 = None
        permute_525: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_604, [1, 0, 2, 3]);  view_604 = None
        permute_526: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_525, [0, 2, 1, 3]);  permute_525 = None
        view_605: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_526, [96, 1024, 64]);  permute_526 = None
        view_606: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_605, [96, 2, 512, 64]);  view_605 = None
        as_strided_50: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_606, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_606 = None
        unsqueeze_109: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_50, 4);  as_strided_50 = None
        clone_73: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_109, memory_format = torch.contiguous_format);  unsqueeze_109 = None
        view_607: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_73, [288, 512, 64]);  clone_73 = None
        permute_529: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_520, [0, 1, 4, 3, 2]);  permute_520 = None
        clone_74: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_529, memory_format = torch.contiguous_format);  permute_529 = None
        view_608: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_74, [288, 64, 512]);  clone_74 = None
        bmm_10: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_607, view_608)
        view_609: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_10, [96, 3, 512, 1, 512]);  bmm_10 = None
        permute_530: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_609, [0, 1, 2, 4, 3]);  view_609 = None
        view_610: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_530, [96, 3, 512, 512]);  permute_530 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_20: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_610, [0, 0, 0, 1], 0.0);  view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_611: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_20, [96, 3, 512, 513]);  constant_pad_nd_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_671: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_611, 2, 0, 256)
        slice_672: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_671, 3, 0, 257);  slice_671 = None
        copy_60: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_672);  slice_672 = None
        slice_scatter_110: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_60, 3, 256, 9223372036854775807);  copy_60 = None
        slice_scatter_111: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_110, 1, 0, -1);  slice_scatter_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_155: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_611, 1, -1)
        slice_681: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_155, 1, 256, 9223372036854775807);  select_155 = None
        slice_682: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_681, 2, 0, 257);  slice_681 = None
        select_158: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_111, 1, -1)
        slice_684: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_158, 2, 256, 9223372036854775807)
        copy_61: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_684, slice_682);  slice_684 = slice_682 = None
        slice_scatter_112: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_158, copy_61, 2, 256, 9223372036854775807);  select_158 = copy_61 = None
        select_scatter_20: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_111, slice_scatter_112, 1, -1);  slice_scatter_111 = slice_scatter_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_687: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_611, 2, -257, -1)
        slice_688: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_687, 3, 257, 9223372036854775807);  slice_687 = None
        slice_692: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_20, 1, 1, 9223372036854775807)
        slice_693: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_692, 3, 0, 256)
        copy_62: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_693, slice_688);  slice_693 = slice_688 = None
        slice_scatter_113: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_692, copy_62, 3, 0, 256);  slice_692 = copy_62 = None
        slice_scatter_114: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_20, slice_scatter_113, 1, 1, 9223372036854775807);  select_scatter_20 = slice_scatter_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_163: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_611, 1, 0);  view_611 = None
        slice_700: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_163, 1, 0, 255);  select_163 = None
        slice_701: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_700, 2, -255, 9223372036854775807);  slice_700 = None
        select_167: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_114, 1, 0)
        slice_705: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_167, 1, 1, 256)
        slice_706: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_705, 2, 1, 256)
        copy_63: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_706, slice_701);  slice_706 = slice_701 = None
        slice_scatter_115: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_705, copy_63, 2, 1, 256);  slice_705 = copy_63 = None
        slice_scatter_116: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_167, slice_scatter_115, 1, 1, 256);  select_167 = slice_scatter_115 = None
        select_scatter_21: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_114, slice_scatter_116, 1, 0);  slice_scatter_114 = slice_scatter_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_616: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_21, [8, 12, 1024, 513]);  select_scatter_21 = None
        permute_534: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_616, [0, 2, 1, 3]);  view_616 = None
        slice_716: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_534, 1, 0, 256)
        slice_717: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_716, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_41: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_717)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_64: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_717, where_41);  slice_717 = where_41 = None
        slice_scatter_117: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_716, copy_64, 3, 0, 257);  slice_716 = copy_64 = None
        slice_scatter_118: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_534, slice_scatter_117, 1, 0, 256);  permute_534 = slice_scatter_117 = None
        permute_539: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_118, [0, 2, 1, 3]);  slice_scatter_118 = None
        view_620: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_539, [96, 4, 256, 513]);  permute_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_630: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_620, [8, 12, 1024, 513]);  view_620 = None
        permute_547: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
        slice_732: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_547, 1, -256, 9223372036854775807)
        slice_733: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_732, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_42: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_733)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_65: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_733, where_42);  slice_733 = where_42 = None
        slice_scatter_119: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_732, copy_65, 3, -257, 9223372036854775807);  slice_732 = copy_65 = None
        slice_scatter_120: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_547, slice_scatter_119, 1, -256, 9223372036854775807);  permute_547 = slice_scatter_119 = None
        permute_552: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_120, [0, 2, 1, 3]);  slice_scatter_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_581: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_552, [0, 2, 1, 3]);  permute_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_80: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_581, permute_82);  permute_581 = None
        permute_583: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_80, [0, 2, 1, 3]);  add_80 = None
        permute_584: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_583, [0, 2, 1, 3]);  permute_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_79: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
        amax_5: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_79, [-1], True)
        sub_44: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_79, amax_5);  clone_79 = amax_5 = None
        exp_5: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
        sum_6: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_57: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_47: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_57)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_20: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_15: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 1e-30);  inductor_random_default_20 = None
        mul_71: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_15, where_47);  where_47 = None
        mul_72: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_71, 1.0);  mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_673: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_77, [1024, 8, 12, 64]);  add_77 = None
        permute_586: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_673, [1, 0, 2, 3]);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_587: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_72, [0, 2, 1, 3]);  mul_72 = None
        clone_80: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
        view_674: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_80, [96, 4, 256, 513]);  clone_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_588: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_586, [0, 2, 1, 3]);  permute_586 = None
        view_675: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_588, [96, 1024, 64]);  permute_588 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_22: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_675, [0, 0, 256, 256], -1.0);  view_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_53: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_22, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_22 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_23: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_674, [0, 257], 0.0);  view_674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_676: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_23, [96, 4, -1]);  constant_pad_nd_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_803: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_676, 2, 0, -256);  view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_677: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_803, [96, 4, 256, 769]);  slice_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_804: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_677, 3, 0, -1);  view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_124: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_804, 4);  slice_804 = None
        unsqueeze_125: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_53, 4);  as_strided_53 = None
        view_678: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_124, [384, 256, 768]);  unsqueeze_124 = None
        clone_81: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_125, memory_format = torch.contiguous_format);  unsqueeze_125 = None
        view_679: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_81, [384, 768, 64]);  clone_81 = None
        bmm_11: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_678, view_679)
        view_680: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_11, [96, 4, 256, 1, 64]);  bmm_11 = None
        permute_593: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_680, [0, 1, 2, 4, 3]);  view_680 = None
        view_681: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_593, [96, 4, 256, 64]);  permute_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_682: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_681, [8, 12, 1024, 64]);  view_681 = None
        permute_594: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_595: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_594, [1, 0, 2, 3]);  permute_594 = None
        clone_82: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_595, memory_format = torch.contiguous_format);  permute_595 = None
        view_683: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_82, [1024, 8, 768]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_596: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_683, [1, 0, 2]);  view_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_597: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0])
        clone_83: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_596, memory_format = torch.contiguous_format);  permute_596 = None
        view_684: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_83, [8192, 768]);  clone_83 = None
        mm_23: "f32[8192, 768]" = torch.ops.aten.mm.default(view_684, permute_597);  permute_597 = None
        view_685: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_23, [8, 1024, 768]);  mm_23 = None
        add_82: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_685, primals_91);  view_685 = primals_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_19: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_16: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 1e-30);  inductor_random_default_19 = None
        mul_73: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_16, add_82);  add_82 = None
        mul_74: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_73, 1.0);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_74, add_74);  mul_74 = add_74 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 1024, 1]" = var_mean_10[0]
        getitem_21: "f32[8, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        add_84: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_46: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_21);  add_83 = getitem_21 = None
        mul_75: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_10);  sub_46 = None
        mul_76: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_75, primals_92)
        add_85: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_76, primals_93);  mul_76 = primals_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_686: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_85, [8192, 768])
        permute_598: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_94, [1, 0])
        addmm_10: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_95, view_686, permute_598);  primals_95 = permute_598 = None
        view_687: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_77: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_687, 0.5)
        mul_78: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_687, 0.7071067811865476);  view_687 = None
        erf_5: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_86: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_79: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_86);  mul_77 = add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_688: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_79, [8192, 3072]);  mul_79 = None
        permute_599: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0])
        addmm_11: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_97, view_688, permute_599);  primals_97 = permute_599 = None
        view_689: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_11, [8, 1024, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_18: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_17: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 1e-30);  inductor_random_default_18 = None
        mul_80: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_17, view_689);  view_689 = None
        mul_81: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, 1.0);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_81, add_85);  mul_81 = add_85 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 1024, 1]" = var_mean_11[0]
        getitem_23: "f32[8, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        add_88: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_47: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_23);  add_87 = getitem_23 = None
        mul_82: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_11);  sub_47 = None
        mul_83: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, primals_98)
        add_89: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_83, primals_99);  mul_83 = primals_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_600: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_89, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_601: "f32[768, 768]" = torch.ops.aten.permute.default(primals_100, [1, 0])
        clone_84: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_600, memory_format = torch.contiguous_format);  permute_600 = None
        view_690: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_84, [8192, 768]);  clone_84 = None
        mm_24: "f32[8192, 768]" = torch.ops.aten.mm.default(view_690, permute_601);  permute_601 = None
        view_691: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_24, [1024, 8, 768]);  mm_24 = None
        add_90: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_691, primals_101);  view_691 = primals_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_602: "f32[768, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0])
        mm_25: "f32[8192, 768]" = torch.ops.aten.mm.default(view_690, permute_602);  permute_602 = None
        view_693: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_25, [1024, 8, 768]);  mm_25 = None
        add_91: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_693, primals_103);  view_693 = primals_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_603: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0])
        mm_26: "f32[8192, 768]" = torch.ops.aten.mm.default(view_690, permute_603);  permute_603 = None
        view_695: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_26, [1024, 8, 768]);  mm_26 = None
        add_92: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_695, primals_105);  view_695 = primals_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_60: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_90, 8.0);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_698: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_91, [1024, 8, 12, 64]);  add_91 = None
        permute_605: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_698, [1, 0, 2, 3]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_610: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_605, [0, 2, 1, 3]);  permute_605 = None
        view_702: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_610, [96, 1024, 64]);  permute_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_709: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_702, [96, 2, 512, 64]);  view_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_55: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_709, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_128: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_55, 4);  as_strided_55 = None
        permute_620: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_128, [0, 1, 4, 2, 3]);  unsqueeze_128 = None
        view_719: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_60, [1024, 8, 12, 64]);  div_60 = None
        permute_625: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_719, [1, 0, 2, 3]);  view_719 = None
        permute_626: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_625, [0, 2, 1, 3]);  permute_625 = None
        view_720: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_626, [96, 1024, 64]);  permute_626 = None
        view_721: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_720, [96, 2, 512, 64]);  view_720 = None
        as_strided_59: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_721, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_721 = None
        unsqueeze_130: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_59, 4);  as_strided_59 = None
        clone_87: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_130, memory_format = torch.contiguous_format);  unsqueeze_130 = None
        view_722: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_87, [288, 512, 64]);  clone_87 = None
        permute_629: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_620, [0, 1, 4, 3, 2]);  permute_620 = None
        clone_88: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
        view_723: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_88, [288, 64, 512]);  clone_88 = None
        bmm_12: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_722, view_723)
        view_724: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_12, [96, 3, 512, 1, 512]);  bmm_12 = None
        permute_630: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_724, [0, 1, 2, 4, 3]);  view_724 = None
        view_725: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_630, [96, 3, 512, 512]);  permute_630 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_24: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_725, [0, 0, 0, 1], 0.0);  view_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_726: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_24, [96, 3, 512, 513]);  constant_pad_nd_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_805: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_726, 2, 0, 256)
        slice_806: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_805, 3, 0, 257);  slice_805 = None
        copy_72: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_806);  slice_806 = None
        slice_scatter_132: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_72, 3, 256, 9223372036854775807);  copy_72 = None
        slice_scatter_133: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_132, 1, 0, -1);  slice_scatter_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_186: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_726, 1, -1)
        slice_815: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_186, 1, 256, 9223372036854775807);  select_186 = None
        slice_816: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_815, 2, 0, 257);  slice_815 = None
        select_189: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_133, 1, -1)
        slice_818: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_189, 2, 256, 9223372036854775807)
        copy_73: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_818, slice_816);  slice_818 = slice_816 = None
        slice_scatter_134: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_189, copy_73, 2, 256, 9223372036854775807);  select_189 = copy_73 = None
        select_scatter_24: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_133, slice_scatter_134, 1, -1);  slice_scatter_133 = slice_scatter_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_821: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_726, 2, -257, -1)
        slice_822: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_821, 3, 257, 9223372036854775807);  slice_821 = None
        slice_826: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_24, 1, 1, 9223372036854775807)
        slice_827: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_826, 3, 0, 256)
        copy_74: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_827, slice_822);  slice_827 = slice_822 = None
        slice_scatter_135: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_826, copy_74, 3, 0, 256);  slice_826 = copy_74 = None
        slice_scatter_136: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_24, slice_scatter_135, 1, 1, 9223372036854775807);  select_scatter_24 = slice_scatter_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_194: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_726, 1, 0);  view_726 = None
        slice_834: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_194, 1, 0, 255);  select_194 = None
        slice_835: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_834, 2, -255, 9223372036854775807);  slice_834 = None
        select_198: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_136, 1, 0)
        slice_839: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_198, 1, 1, 256)
        slice_840: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_839, 2, 1, 256)
        copy_75: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_840, slice_835);  slice_840 = slice_835 = None
        slice_scatter_137: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_839, copy_75, 2, 1, 256);  slice_839 = copy_75 = None
        slice_scatter_138: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_198, slice_scatter_137, 1, 1, 256);  select_198 = slice_scatter_137 = None
        select_scatter_25: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_136, slice_scatter_138, 1, 0);  slice_scatter_136 = slice_scatter_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_731: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_25, [8, 12, 1024, 513]);  select_scatter_25 = None
        permute_634: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_731, [0, 2, 1, 3]);  view_731 = None
        slice_850: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_634, 1, 0, 256)
        slice_851: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_850, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_49: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_851)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_76: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_851, where_49);  slice_851 = where_49 = None
        slice_scatter_139: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_850, copy_76, 3, 0, 257);  slice_850 = copy_76 = None
        slice_scatter_140: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_634, slice_scatter_139, 1, 0, 256);  permute_634 = slice_scatter_139 = None
        permute_639: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_140, [0, 2, 1, 3]);  slice_scatter_140 = None
        view_735: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_639, [96, 4, 256, 513]);  permute_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_745: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_735, [8, 12, 1024, 513]);  view_735 = None
        permute_647: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_745, [0, 2, 1, 3]);  view_745 = None
        slice_866: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_647, 1, -256, 9223372036854775807)
        slice_867: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_866, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_50: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_867)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_77: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_867, where_50);  slice_867 = where_50 = None
        slice_scatter_141: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_866, copy_77, 3, -257, 9223372036854775807);  slice_866 = copy_77 = None
        slice_scatter_142: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_647, slice_scatter_141, 1, -256, 9223372036854775807);  permute_647 = slice_scatter_141 = None
        permute_652: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_142, [0, 2, 1, 3]);  slice_scatter_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_681: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_652, [0, 2, 1, 3]);  permute_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_95: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_681, permute_82);  permute_681 = None
        permute_683: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_95, [0, 2, 1, 3]);  add_95 = None
        permute_684: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_683, [0, 2, 1, 3]);  permute_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_93: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_684, memory_format = torch.contiguous_format);  permute_684 = None
        amax_6: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_93, [-1], True)
        sub_52: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_93, amax_6);  clone_93 = amax_6 = None
        exp_6: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_7: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_67: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_55: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_67)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_17: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 1e-30);  inductor_random_default_17 = None
        mul_85: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_18, where_55);  where_55 = None
        mul_86: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_85, 1.0);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_788: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_92, [1024, 8, 12, 64]);  add_92 = None
        permute_686: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_788, [1, 0, 2, 3]);  view_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_687: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_86, [0, 2, 1, 3]);  mul_86 = None
        clone_94: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_687, memory_format = torch.contiguous_format);  permute_687 = None
        view_789: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_94, [96, 4, 256, 513]);  clone_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_688: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_686, [0, 2, 1, 3]);  permute_686 = None
        view_790: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_688, [96, 1024, 64]);  permute_688 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_26: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_790, [0, 0, 256, 256], -1.0);  view_790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_62: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_26, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_26 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_27: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_789, [0, 257], 0.0);  view_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_791: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_27, [96, 4, -1]);  constant_pad_nd_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_937: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_791, 2, 0, -256);  view_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_792: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_937, [96, 4, 256, 769]);  slice_937 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_938: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_792, 3, 0, -1);  view_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_145: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_938, 4);  slice_938 = None
        unsqueeze_146: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_62, 4);  as_strided_62 = None
        view_793: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_145, [384, 256, 768]);  unsqueeze_145 = None
        clone_95: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_146, memory_format = torch.contiguous_format);  unsqueeze_146 = None
        view_794: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_95, [384, 768, 64]);  clone_95 = None
        bmm_13: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_793, view_794)
        view_795: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_13, [96, 4, 256, 1, 64]);  bmm_13 = None
        permute_693: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_795, [0, 1, 2, 4, 3]);  view_795 = None
        view_796: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_693, [96, 4, 256, 64]);  permute_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_797: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_796, [8, 12, 1024, 64]);  view_796 = None
        permute_694: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_797, [0, 2, 1, 3]);  view_797 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_695: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_694, [1, 0, 2, 3]);  permute_694 = None
        clone_96: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_695, memory_format = torch.contiguous_format);  permute_695 = None
        view_798: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_96, [1024, 8, 768]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_696: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_798, [1, 0, 2]);  view_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_697: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0])
        clone_97: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_696, memory_format = torch.contiguous_format);  permute_696 = None
        view_799: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_97, [8192, 768]);  clone_97 = None
        mm_27: "f32[8192, 768]" = torch.ops.aten.mm.default(view_799, permute_697);  permute_697 = None
        view_800: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_27, [8, 1024, 768]);  mm_27 = None
        add_97: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_800, primals_107);  view_800 = primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_16: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_19: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 1e-30);  inductor_random_default_16 = None
        mul_87: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_19, add_97);  add_97 = None
        mul_88: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_87, 1.0);  mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_98: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_88, add_89);  mul_88 = add_89 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 1024, 1]" = var_mean_12[0]
        getitem_25: "f32[8, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        add_99: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_54: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_98, getitem_25);  add_98 = getitem_25 = None
        mul_89: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_12);  sub_54 = None
        mul_90: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_89, primals_108)
        add_100: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_90, primals_109);  mul_90 = primals_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_801: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_100, [8192, 768])
        permute_698: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_110, [1, 0])
        addmm_12: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_111, view_801, permute_698);  primals_111 = permute_698 = None
        view_802: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_12, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_91: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_802, 0.5)
        mul_92: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_802, 0.7071067811865476);  view_802 = None
        erf_6: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_101: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_93: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_91, add_101);  mul_91 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_803: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_93, [8192, 3072]);  mul_93 = None
        permute_699: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_112, [1, 0])
        addmm_13: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_113, view_803, permute_699);  primals_113 = permute_699 = None
        view_804: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_13, [8, 1024, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_15: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_20: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 1e-30);  inductor_random_default_15 = None
        mul_94: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_20, view_804);  view_804 = None
        mul_95: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_94, 1.0);  mul_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_102: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_95, add_100);  mul_95 = add_100 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 1024, 1]" = var_mean_13[0]
        getitem_27: "f32[8, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        add_103: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_55: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_102, getitem_27);  add_102 = getitem_27 = None
        mul_96: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_13);  sub_55 = None
        mul_97: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, primals_114)
        add_104: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_115);  mul_97 = primals_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_700: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_104, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_701: "f32[768, 768]" = torch.ops.aten.permute.default(primals_116, [1, 0])
        clone_98: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_700, memory_format = torch.contiguous_format);  permute_700 = None
        view_805: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_98, [8192, 768]);  clone_98 = None
        mm_28: "f32[8192, 768]" = torch.ops.aten.mm.default(view_805, permute_701);  permute_701 = None
        view_806: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_28, [1024, 8, 768]);  mm_28 = None
        add_105: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_806, primals_117);  view_806 = primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_702: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0])
        mm_29: "f32[8192, 768]" = torch.ops.aten.mm.default(view_805, permute_702);  permute_702 = None
        view_808: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_29, [1024, 8, 768]);  mm_29 = None
        add_106: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_808, primals_119);  view_808 = primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_703: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0])
        mm_30: "f32[8192, 768]" = torch.ops.aten.mm.default(view_805, permute_703);  permute_703 = None
        view_810: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_30, [1024, 8, 768]);  mm_30 = None
        add_107: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_810, primals_121);  view_810 = primals_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_70: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_105, 8.0);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_813: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_106, [1024, 8, 12, 64]);  add_106 = None
        permute_705: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_813, [1, 0, 2, 3]);  view_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_710: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_705, [0, 2, 1, 3]);  permute_705 = None
        view_817: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_710, [96, 1024, 64]);  permute_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_824: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_817, [96, 2, 512, 64]);  view_817 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_64: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_824, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_149: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_64, 4);  as_strided_64 = None
        permute_720: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_149, [0, 1, 4, 2, 3]);  unsqueeze_149 = None
        view_834: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_70, [1024, 8, 12, 64]);  div_70 = None
        permute_725: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_834, [1, 0, 2, 3]);  view_834 = None
        permute_726: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_725, [0, 2, 1, 3]);  permute_725 = None
        view_835: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_726, [96, 1024, 64]);  permute_726 = None
        view_836: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_835, [96, 2, 512, 64]);  view_835 = None
        as_strided_68: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_836, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_836 = None
        unsqueeze_151: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_68, 4);  as_strided_68 = None
        clone_101: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_151, memory_format = torch.contiguous_format);  unsqueeze_151 = None
        view_837: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_101, [288, 512, 64]);  clone_101 = None
        permute_729: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_720, [0, 1, 4, 3, 2]);  permute_720 = None
        clone_102: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_729, memory_format = torch.contiguous_format);  permute_729 = None
        view_838: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_102, [288, 64, 512]);  clone_102 = None
        bmm_14: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_837, view_838)
        view_839: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_14, [96, 3, 512, 1, 512]);  bmm_14 = None
        permute_730: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_839, [0, 1, 2, 4, 3]);  view_839 = None
        view_840: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_730, [96, 3, 512, 512]);  permute_730 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_28: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_840, [0, 0, 0, 1], 0.0);  view_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_841: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_28, [96, 3, 512, 513]);  constant_pad_nd_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_939: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_841, 2, 0, 256)
        slice_940: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_939, 3, 0, 257);  slice_939 = None
        copy_84: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_940);  slice_940 = None
        slice_scatter_154: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_84, 3, 256, 9223372036854775807);  copy_84 = None
        slice_scatter_155: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_154, 1, 0, -1);  slice_scatter_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_217: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_841, 1, -1)
        slice_949: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_217, 1, 256, 9223372036854775807);  select_217 = None
        slice_950: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_949, 2, 0, 257);  slice_949 = None
        select_220: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_155, 1, -1)
        slice_952: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_220, 2, 256, 9223372036854775807)
        copy_85: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_952, slice_950);  slice_952 = slice_950 = None
        slice_scatter_156: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_220, copy_85, 2, 256, 9223372036854775807);  select_220 = copy_85 = None
        select_scatter_28: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_155, slice_scatter_156, 1, -1);  slice_scatter_155 = slice_scatter_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_955: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_841, 2, -257, -1)
        slice_956: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_955, 3, 257, 9223372036854775807);  slice_955 = None
        slice_960: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_28, 1, 1, 9223372036854775807)
        slice_961: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_960, 3, 0, 256)
        copy_86: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_961, slice_956);  slice_961 = slice_956 = None
        slice_scatter_157: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_960, copy_86, 3, 0, 256);  slice_960 = copy_86 = None
        slice_scatter_158: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_28, slice_scatter_157, 1, 1, 9223372036854775807);  select_scatter_28 = slice_scatter_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_225: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_841, 1, 0);  view_841 = None
        slice_968: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_225, 1, 0, 255);  select_225 = None
        slice_969: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_968, 2, -255, 9223372036854775807);  slice_968 = None
        select_229: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_158, 1, 0)
        slice_973: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_229, 1, 1, 256)
        slice_974: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_973, 2, 1, 256)
        copy_87: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_974, slice_969);  slice_974 = slice_969 = None
        slice_scatter_159: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_973, copy_87, 2, 1, 256);  slice_973 = copy_87 = None
        slice_scatter_160: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_229, slice_scatter_159, 1, 1, 256);  select_229 = slice_scatter_159 = None
        select_scatter_29: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_158, slice_scatter_160, 1, 0);  slice_scatter_158 = slice_scatter_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_846: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_29, [8, 12, 1024, 513]);  select_scatter_29 = None
        permute_734: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_846, [0, 2, 1, 3]);  view_846 = None
        slice_984: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_734, 1, 0, 256)
        slice_985: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_984, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_57: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_985)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_88: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_985, where_57);  slice_985 = where_57 = None
        slice_scatter_161: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_984, copy_88, 3, 0, 257);  slice_984 = copy_88 = None
        slice_scatter_162: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_734, slice_scatter_161, 1, 0, 256);  permute_734 = slice_scatter_161 = None
        permute_739: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_162, [0, 2, 1, 3]);  slice_scatter_162 = None
        view_850: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_739, [96, 4, 256, 513]);  permute_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_860: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_850, [8, 12, 1024, 513]);  view_850 = None
        permute_747: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_860, [0, 2, 1, 3]);  view_860 = None
        slice_1000: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_747, 1, -256, 9223372036854775807)
        slice_1001: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1000, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_58: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_1001)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_89: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1001, where_58);  slice_1001 = where_58 = None
        slice_scatter_163: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1000, copy_89, 3, -257, 9223372036854775807);  slice_1000 = copy_89 = None
        slice_scatter_164: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_747, slice_scatter_163, 1, -256, 9223372036854775807);  permute_747 = slice_scatter_163 = None
        permute_752: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_164, [0, 2, 1, 3]);  slice_scatter_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_781: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_752, [0, 2, 1, 3]);  permute_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_110: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_781, permute_82);  permute_781 = None
        permute_783: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_110, [0, 2, 1, 3]);  add_110 = None
        permute_784: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_783, [0, 2, 1, 3]);  permute_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_107: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_784, memory_format = torch.contiguous_format);  permute_784 = None
        amax_7: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_107, [-1], True)
        sub_60: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_107, amax_7);  clone_107 = amax_7 = None
        exp_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_8: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_77: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_63: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_77)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_14: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_21: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 1e-30);  inductor_random_default_14 = None
        mul_99: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_21, where_63);  where_63 = None
        mul_100: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_99, 1.0);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_903: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_107, [1024, 8, 12, 64]);  add_107 = None
        permute_786: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_903, [1, 0, 2, 3]);  view_903 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_787: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_100, [0, 2, 1, 3]);  mul_100 = None
        clone_108: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_787, memory_format = torch.contiguous_format);  permute_787 = None
        view_904: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_108, [96, 4, 256, 513]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_788: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_786, [0, 2, 1, 3]);  permute_786 = None
        view_905: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_788, [96, 1024, 64]);  permute_788 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_30: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_905, [0, 0, 256, 256], -1.0);  view_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_71: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_30, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_30 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_31: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_904, [0, 257], 0.0);  view_904 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_906: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_31, [96, 4, -1]);  constant_pad_nd_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1071: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_906, 2, 0, -256);  view_906 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_907: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_1071, [96, 4, 256, 769]);  slice_1071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1072: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_907, 3, 0, -1);  view_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_166: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1072, 4);  slice_1072 = None
        unsqueeze_167: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_71, 4);  as_strided_71 = None
        view_908: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_166, [384, 256, 768]);  unsqueeze_166 = None
        clone_109: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_167, memory_format = torch.contiguous_format);  unsqueeze_167 = None
        view_909: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_109, [384, 768, 64]);  clone_109 = None
        bmm_15: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_908, view_909)
        view_910: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_15, [96, 4, 256, 1, 64]);  bmm_15 = None
        permute_793: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_910, [0, 1, 2, 4, 3]);  view_910 = None
        view_911: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_793, [96, 4, 256, 64]);  permute_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_912: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_911, [8, 12, 1024, 64]);  view_911 = None
        permute_794: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_912, [0, 2, 1, 3]);  view_912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_795: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_794, [1, 0, 2, 3]);  permute_794 = None
        clone_110: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
        view_913: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_110, [1024, 8, 768]);  clone_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_796: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_913, [1, 0, 2]);  view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_797: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0])
        clone_111: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_796, memory_format = torch.contiguous_format);  permute_796 = None
        view_914: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_111, [8192, 768]);  clone_111 = None
        mm_31: "f32[8192, 768]" = torch.ops.aten.mm.default(view_914, permute_797);  permute_797 = None
        view_915: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_31, [8, 1024, 768]);  mm_31 = None
        add_112: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_915, primals_123);  view_915 = primals_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_13: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 1e-30);  inductor_random_default_13 = None
        mul_101: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_22, add_112);  add_112 = None
        mul_102: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_101, 1.0);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_113: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_102, add_104);  mul_102 = add_104 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_28: "f32[8, 1024, 1]" = var_mean_14[0]
        getitem_29: "f32[8, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        add_114: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_62: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_113, getitem_29);  add_113 = getitem_29 = None
        mul_103: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = None
        mul_104: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_103, primals_124)
        add_115: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_104, primals_125);  mul_104 = primals_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_916: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_115, [8192, 768])
        permute_798: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_126, [1, 0])
        addmm_14: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_127, view_916, permute_798);  primals_127 = permute_798 = None
        view_917: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_14, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_105: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_917, 0.5)
        mul_106: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_917, 0.7071067811865476);  view_917 = None
        erf_7: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
        add_116: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_107: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_105, add_116);  mul_105 = add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_918: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_107, [8192, 3072]);  mul_107 = None
        permute_799: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0])
        addmm_15: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_129, view_918, permute_799);  primals_129 = permute_799 = None
        view_919: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_15, [8, 1024, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_12: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_23: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 1e-30);  inductor_random_default_12 = None
        mul_108: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_23, view_919);  view_919 = None
        mul_109: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_108, 1.0);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_117: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_109, add_115);  mul_109 = add_115 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 1024, 1]" = var_mean_15[0]
        getitem_31: "f32[8, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        add_118: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_63: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_117, getitem_31);  add_117 = getitem_31 = None
        mul_110: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_15);  sub_63 = None
        mul_111: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_110, primals_130)
        add_119: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_111, primals_131);  mul_111 = primals_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_800: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_119, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_801: "f32[768, 768]" = torch.ops.aten.permute.default(primals_132, [1, 0])
        clone_112: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_800, memory_format = torch.contiguous_format);  permute_800 = None
        view_920: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_112, [8192, 768]);  clone_112 = None
        mm_32: "f32[8192, 768]" = torch.ops.aten.mm.default(view_920, permute_801);  permute_801 = None
        view_921: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_32, [1024, 8, 768]);  mm_32 = None
        add_120: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_921, primals_133);  view_921 = primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_802: "f32[768, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0])
        mm_33: "f32[8192, 768]" = torch.ops.aten.mm.default(view_920, permute_802);  permute_802 = None
        view_923: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_33, [1024, 8, 768]);  mm_33 = None
        add_121: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_923, primals_135);  view_923 = primals_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_803: "f32[768, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0])
        mm_34: "f32[8192, 768]" = torch.ops.aten.mm.default(view_920, permute_803);  permute_803 = None
        view_925: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_34, [1024, 8, 768]);  mm_34 = None
        add_122: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_925, primals_137);  view_925 = primals_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_80: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_120, 8.0);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_928: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_121, [1024, 8, 12, 64]);  add_121 = None
        permute_805: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_928, [1, 0, 2, 3]);  view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_810: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_805, [0, 2, 1, 3]);  permute_805 = None
        view_932: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_810, [96, 1024, 64]);  permute_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_939: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_932, [96, 2, 512, 64]);  view_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_73: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_939, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_939 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_170: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_73, 4);  as_strided_73 = None
        permute_820: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_170, [0, 1, 4, 2, 3]);  unsqueeze_170 = None
        view_949: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_80, [1024, 8, 12, 64]);  div_80 = None
        permute_825: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_949, [1, 0, 2, 3]);  view_949 = None
        permute_826: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_825, [0, 2, 1, 3]);  permute_825 = None
        view_950: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_826, [96, 1024, 64]);  permute_826 = None
        view_951: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_950, [96, 2, 512, 64]);  view_950 = None
        as_strided_77: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_951, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_951 = None
        unsqueeze_172: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_77, 4);  as_strided_77 = None
        clone_115: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_172, memory_format = torch.contiguous_format);  unsqueeze_172 = None
        view_952: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_115, [288, 512, 64]);  clone_115 = None
        permute_829: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_820, [0, 1, 4, 3, 2]);  permute_820 = None
        clone_116: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_829, memory_format = torch.contiguous_format);  permute_829 = None
        view_953: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_116, [288, 64, 512]);  clone_116 = None
        bmm_16: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_952, view_953)
        view_954: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_16, [96, 3, 512, 1, 512]);  bmm_16 = None
        permute_830: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_954, [0, 1, 2, 4, 3]);  view_954 = None
        view_955: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_830, [96, 3, 512, 512]);  permute_830 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_32: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_955, [0, 0, 0, 1], 0.0);  view_955 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_956: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_32, [96, 3, 512, 513]);  constant_pad_nd_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1073: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_956, 2, 0, 256)
        slice_1074: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1073, 3, 0, 257);  slice_1073 = None
        copy_96: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_1074);  slice_1074 = None
        slice_scatter_176: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_96, 3, 256, 9223372036854775807);  copy_96 = None
        slice_scatter_177: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_176, 1, 0, -1);  slice_scatter_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_248: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_956, 1, -1)
        slice_1083: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_248, 1, 256, 9223372036854775807);  select_248 = None
        slice_1084: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1083, 2, 0, 257);  slice_1083 = None
        select_251: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_177, 1, -1)
        slice_1086: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_251, 2, 256, 9223372036854775807)
        copy_97: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_1086, slice_1084);  slice_1086 = slice_1084 = None
        slice_scatter_178: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_251, copy_97, 2, 256, 9223372036854775807);  select_251 = copy_97 = None
        select_scatter_32: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_177, slice_scatter_178, 1, -1);  slice_scatter_177 = slice_scatter_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1089: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_956, 2, -257, -1)
        slice_1090: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1089, 3, 257, 9223372036854775807);  slice_1089 = None
        slice_1094: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_32, 1, 1, 9223372036854775807)
        slice_1095: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1094, 3, 0, 256)
        copy_98: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1095, slice_1090);  slice_1095 = slice_1090 = None
        slice_scatter_179: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1094, copy_98, 3, 0, 256);  slice_1094 = copy_98 = None
        slice_scatter_180: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_32, slice_scatter_179, 1, 1, 9223372036854775807);  select_scatter_32 = slice_scatter_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_256: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_956, 1, 0);  view_956 = None
        slice_1102: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_256, 1, 0, 255);  select_256 = None
        slice_1103: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1102, 2, -255, 9223372036854775807);  slice_1102 = None
        select_260: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_180, 1, 0)
        slice_1107: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_260, 1, 1, 256)
        slice_1108: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1107, 2, 1, 256)
        copy_99: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_1108, slice_1103);  slice_1108 = slice_1103 = None
        slice_scatter_181: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1107, copy_99, 2, 1, 256);  slice_1107 = copy_99 = None
        slice_scatter_182: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_260, slice_scatter_181, 1, 1, 256);  select_260 = slice_scatter_181 = None
        select_scatter_33: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_180, slice_scatter_182, 1, 0);  slice_scatter_180 = slice_scatter_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_961: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_33, [8, 12, 1024, 513]);  select_scatter_33 = None
        permute_834: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_961, [0, 2, 1, 3]);  view_961 = None
        slice_1118: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_834, 1, 0, 256)
        slice_1119: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1118, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_65: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_1119)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_100: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1119, where_65);  slice_1119 = where_65 = None
        slice_scatter_183: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1118, copy_100, 3, 0, 257);  slice_1118 = copy_100 = None
        slice_scatter_184: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_834, slice_scatter_183, 1, 0, 256);  permute_834 = slice_scatter_183 = None
        permute_839: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_184, [0, 2, 1, 3]);  slice_scatter_184 = None
        view_965: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_839, [96, 4, 256, 513]);  permute_839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_975: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_965, [8, 12, 1024, 513]);  view_965 = None
        permute_847: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_975, [0, 2, 1, 3]);  view_975 = None
        slice_1134: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_847, 1, -256, 9223372036854775807)
        slice_1135: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1134, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_66: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_1135)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_101: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1135, where_66);  slice_1135 = where_66 = None
        slice_scatter_185: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1134, copy_101, 3, -257, 9223372036854775807);  slice_1134 = copy_101 = None
        slice_scatter_186: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_847, slice_scatter_185, 1, -256, 9223372036854775807);  permute_847 = slice_scatter_185 = None
        permute_852: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_186, [0, 2, 1, 3]);  slice_scatter_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_881: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_852, [0, 2, 1, 3]);  permute_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_125: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_881, permute_82);  permute_881 = None
        permute_883: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_125, [0, 2, 1, 3]);  add_125 = None
        permute_884: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_883, [0, 2, 1, 3]);  permute_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_121: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_884, memory_format = torch.contiguous_format);  permute_884 = None
        amax_8: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_121, [-1], True)
        sub_68: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_121, amax_8);  clone_121 = amax_8 = None
        exp_8: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
        sum_9: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_87: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_71: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_87)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_11: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_24: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 1e-30);  inductor_random_default_11 = None
        mul_113: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_24, where_71);  where_71 = None
        mul_114: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_113, 1.0);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1018: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_122, [1024, 8, 12, 64]);  add_122 = None
        permute_886: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1018, [1, 0, 2, 3]);  view_1018 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_887: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_114, [0, 2, 1, 3]);  mul_114 = None
        clone_122: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_887, memory_format = torch.contiguous_format);  permute_887 = None
        view_1019: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_122, [96, 4, 256, 513]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_888: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_886, [0, 2, 1, 3]);  permute_886 = None
        view_1020: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_888, [96, 1024, 64]);  permute_888 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_34: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_1020, [0, 0, 256, 256], -1.0);  view_1020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_80: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_34, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_34 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_35: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_1019, [0, 257], 0.0);  view_1019 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1021: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_35, [96, 4, -1]);  constant_pad_nd_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1205: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_1021, 2, 0, -256);  view_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1022: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_1205, [96, 4, 256, 769]);  slice_1205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1206: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_1022, 3, 0, -1);  view_1022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_187: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1206, 4);  slice_1206 = None
        unsqueeze_188: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_80, 4);  as_strided_80 = None
        view_1023: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_187, [384, 256, 768]);  unsqueeze_187 = None
        clone_123: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_188, memory_format = torch.contiguous_format);  unsqueeze_188 = None
        view_1024: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_123, [384, 768, 64]);  clone_123 = None
        bmm_17: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_1023, view_1024)
        view_1025: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_17, [96, 4, 256, 1, 64]);  bmm_17 = None
        permute_893: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_1025, [0, 1, 2, 4, 3]);  view_1025 = None
        view_1026: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_893, [96, 4, 256, 64]);  permute_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1027: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1026, [8, 12, 1024, 64]);  view_1026 = None
        permute_894: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1027, [0, 2, 1, 3]);  view_1027 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_895: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_894, [1, 0, 2, 3]);  permute_894 = None
        clone_124: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_895, memory_format = torch.contiguous_format);  permute_895 = None
        view_1028: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_124, [1024, 8, 768]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_896: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_1028, [1, 0, 2]);  view_1028 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_897: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0])
        clone_125: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_896, memory_format = torch.contiguous_format);  permute_896 = None
        view_1029: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_125, [8192, 768]);  clone_125 = None
        mm_35: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1029, permute_897);  permute_897 = None
        view_1030: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_35, [8, 1024, 768]);  mm_35 = None
        add_127: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1030, primals_139);  view_1030 = primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_10: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_25: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 1e-30);  inductor_random_default_10 = None
        mul_115: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_25, add_127);  add_127 = None
        mul_116: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_115, 1.0);  mul_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_128: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_116, add_119);  mul_116 = add_119 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 1024, 1]" = var_mean_16[0]
        getitem_33: "f32[8, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        add_129: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_70: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_128, getitem_33);  add_128 = getitem_33 = None
        mul_117: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_16);  sub_70 = None
        mul_118: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_117, primals_140)
        add_130: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_118, primals_141);  mul_118 = primals_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1031: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_130, [8192, 768])
        permute_898: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_142, [1, 0])
        addmm_16: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_143, view_1031, permute_898);  primals_143 = permute_898 = None
        view_1032: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_16, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_119: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1032, 0.5)
        mul_120: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1032, 0.7071067811865476);  view_1032 = None
        erf_8: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
        add_131: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_121: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_119, add_131);  mul_119 = add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1033: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_121, [8192, 3072]);  mul_121 = None
        permute_899: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0])
        addmm_17: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_145, view_1033, permute_899);  primals_145 = permute_899 = None
        view_1034: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_17, [8, 1024, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_9: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_26: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 1e-30);  inductor_random_default_9 = None
        mul_122: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_26, view_1034);  view_1034 = None
        mul_123: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_122, 1.0);  mul_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_132: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_123, add_130);  mul_123 = add_130 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_132, [2], correction = 0, keepdim = True)
        getitem_34: "f32[8, 1024, 1]" = var_mean_17[0]
        getitem_35: "f32[8, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        add_133: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_71: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_132, getitem_35);  add_132 = getitem_35 = None
        mul_124: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_17);  sub_71 = None
        mul_125: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_124, primals_146)
        add_134: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_125, primals_147);  mul_125 = primals_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_900: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_134, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_901: "f32[768, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0])
        clone_126: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_900, memory_format = torch.contiguous_format);  permute_900 = None
        view_1035: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_126, [8192, 768]);  clone_126 = None
        mm_36: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1035, permute_901);  permute_901 = None
        view_1036: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_36, [1024, 8, 768]);  mm_36 = None
        add_135: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1036, primals_149);  view_1036 = primals_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_902: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0])
        mm_37: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1035, permute_902);  permute_902 = None
        view_1038: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_37, [1024, 8, 768]);  mm_37 = None
        add_136: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1038, primals_151);  view_1038 = primals_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_903: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0])
        mm_38: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1035, permute_903);  permute_903 = None
        view_1040: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_38, [1024, 8, 768]);  mm_38 = None
        add_137: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1040, primals_153);  view_1040 = primals_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_90: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_135, 8.0);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1043: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_136, [1024, 8, 12, 64]);  add_136 = None
        permute_905: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1043, [1, 0, 2, 3]);  view_1043 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_910: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_905, [0, 2, 1, 3]);  permute_905 = None
        view_1047: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_910, [96, 1024, 64]);  permute_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1054: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_1047, [96, 2, 512, 64]);  view_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_82: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_1054, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_1054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_191: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_82, 4);  as_strided_82 = None
        permute_920: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_191, [0, 1, 4, 2, 3]);  unsqueeze_191 = None
        view_1064: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_90, [1024, 8, 12, 64]);  div_90 = None
        permute_925: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1064, [1, 0, 2, 3]);  view_1064 = None
        permute_926: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_925, [0, 2, 1, 3]);  permute_925 = None
        view_1065: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_926, [96, 1024, 64]);  permute_926 = None
        view_1066: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_1065, [96, 2, 512, 64]);  view_1065 = None
        as_strided_86: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_1066, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_1066 = None
        unsqueeze_193: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_86, 4);  as_strided_86 = None
        clone_129: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_193, memory_format = torch.contiguous_format);  unsqueeze_193 = None
        view_1067: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_129, [288, 512, 64]);  clone_129 = None
        permute_929: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_920, [0, 1, 4, 3, 2]);  permute_920 = None
        clone_130: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_929, memory_format = torch.contiguous_format);  permute_929 = None
        view_1068: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_130, [288, 64, 512]);  clone_130 = None
        bmm_18: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_1067, view_1068)
        view_1069: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_18, [96, 3, 512, 1, 512]);  bmm_18 = None
        permute_930: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_1069, [0, 1, 2, 4, 3]);  view_1069 = None
        view_1070: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_930, [96, 3, 512, 512]);  permute_930 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_36: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1070, [0, 0, 0, 1], 0.0);  view_1070 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1071: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_36, [96, 3, 512, 513]);  constant_pad_nd_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1207: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_1071, 2, 0, 256)
        slice_1208: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1207, 3, 0, 257);  slice_1207 = None
        copy_108: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_1208);  slice_1208 = None
        slice_scatter_198: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_108, 3, 256, 9223372036854775807);  copy_108 = None
        slice_scatter_199: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_198, 1, 0, -1);  slice_scatter_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_279: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_1071, 1, -1)
        slice_1217: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_279, 1, 256, 9223372036854775807);  select_279 = None
        slice_1218: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1217, 2, 0, 257);  slice_1217 = None
        select_282: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_199, 1, -1)
        slice_1220: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_282, 2, 256, 9223372036854775807)
        copy_109: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_1220, slice_1218);  slice_1220 = slice_1218 = None
        slice_scatter_200: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_282, copy_109, 2, 256, 9223372036854775807);  select_282 = copy_109 = None
        select_scatter_36: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_199, slice_scatter_200, 1, -1);  slice_scatter_199 = slice_scatter_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1223: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_1071, 2, -257, -1)
        slice_1224: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1223, 3, 257, 9223372036854775807);  slice_1223 = None
        slice_1228: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_36, 1, 1, 9223372036854775807)
        slice_1229: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1228, 3, 0, 256)
        copy_110: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1229, slice_1224);  slice_1229 = slice_1224 = None
        slice_scatter_201: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1228, copy_110, 3, 0, 256);  slice_1228 = copy_110 = None
        slice_scatter_202: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_36, slice_scatter_201, 1, 1, 9223372036854775807);  select_scatter_36 = slice_scatter_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_287: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_1071, 1, 0);  view_1071 = None
        slice_1236: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_287, 1, 0, 255);  select_287 = None
        slice_1237: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1236, 2, -255, 9223372036854775807);  slice_1236 = None
        select_291: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_202, 1, 0)
        slice_1241: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_291, 1, 1, 256)
        slice_1242: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1241, 2, 1, 256)
        copy_111: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_1242, slice_1237);  slice_1242 = slice_1237 = None
        slice_scatter_203: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1241, copy_111, 2, 1, 256);  slice_1241 = copy_111 = None
        slice_scatter_204: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_291, slice_scatter_203, 1, 1, 256);  select_291 = slice_scatter_203 = None
        select_scatter_37: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_202, slice_scatter_204, 1, 0);  slice_scatter_202 = slice_scatter_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1076: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_37, [8, 12, 1024, 513]);  select_scatter_37 = None
        permute_934: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1076, [0, 2, 1, 3]);  view_1076 = None
        slice_1252: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_934, 1, 0, 256)
        slice_1253: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1252, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_73: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_1253)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_112: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1253, where_73);  slice_1253 = where_73 = None
        slice_scatter_205: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1252, copy_112, 3, 0, 257);  slice_1252 = copy_112 = None
        slice_scatter_206: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_934, slice_scatter_205, 1, 0, 256);  permute_934 = slice_scatter_205 = None
        permute_939: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_206, [0, 2, 1, 3]);  slice_scatter_206 = None
        view_1080: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_939, [96, 4, 256, 513]);  permute_939 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1090: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1080, [8, 12, 1024, 513]);  view_1080 = None
        permute_947: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1090, [0, 2, 1, 3]);  view_1090 = None
        slice_1268: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_947, 1, -256, 9223372036854775807)
        slice_1269: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1268, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_74: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_1269)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_113: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1269, where_74);  slice_1269 = where_74 = None
        slice_scatter_207: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1268, copy_113, 3, -257, 9223372036854775807);  slice_1268 = copy_113 = None
        slice_scatter_208: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_947, slice_scatter_207, 1, -256, 9223372036854775807);  permute_947 = slice_scatter_207 = None
        permute_952: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_208, [0, 2, 1, 3]);  slice_scatter_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_981: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_952, [0, 2, 1, 3]);  permute_952 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_140: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_981, permute_82);  permute_981 = None
        permute_983: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_140, [0, 2, 1, 3]);  add_140 = None
        permute_984: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_983, [0, 2, 1, 3]);  permute_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_135: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_984, memory_format = torch.contiguous_format);  permute_984 = None
        amax_9: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_135, [-1], True)
        sub_76: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_135, amax_9);  clone_135 = amax_9 = None
        exp_9: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_10: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_97: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_79: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_97)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_8: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_27: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 1e-30);  inductor_random_default_8 = None
        mul_127: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_27, where_79);  where_79 = None
        mul_128: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_127, 1.0);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1133: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_137, [1024, 8, 12, 64]);  add_137 = None
        permute_986: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1133, [1, 0, 2, 3]);  view_1133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_987: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_128, [0, 2, 1, 3]);  mul_128 = None
        clone_136: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_987, memory_format = torch.contiguous_format);  permute_987 = None
        view_1134: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_136, [96, 4, 256, 513]);  clone_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_988: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_986, [0, 2, 1, 3]);  permute_986 = None
        view_1135: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_988, [96, 1024, 64]);  permute_988 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_38: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_1135, [0, 0, 256, 256], -1.0);  view_1135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_89: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_38, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_38 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_39: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_1134, [0, 257], 0.0);  view_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1136: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_39, [96, 4, -1]);  constant_pad_nd_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1339: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_1136, 2, 0, -256);  view_1136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1137: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_1339, [96, 4, 256, 769]);  slice_1339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1340: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_1137, 3, 0, -1);  view_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_208: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1340, 4);  slice_1340 = None
        unsqueeze_209: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_89, 4);  as_strided_89 = None
        view_1138: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_208, [384, 256, 768]);  unsqueeze_208 = None
        clone_137: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_209, memory_format = torch.contiguous_format);  unsqueeze_209 = None
        view_1139: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_137, [384, 768, 64]);  clone_137 = None
        bmm_19: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_1138, view_1139)
        view_1140: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_19, [96, 4, 256, 1, 64]);  bmm_19 = None
        permute_993: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_1140, [0, 1, 2, 4, 3]);  view_1140 = None
        view_1141: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_993, [96, 4, 256, 64]);  permute_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1142: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1141, [8, 12, 1024, 64]);  view_1141 = None
        permute_994: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1142, [0, 2, 1, 3]);  view_1142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_995: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_994, [1, 0, 2, 3]);  permute_994 = None
        clone_138: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_995, memory_format = torch.contiguous_format);  permute_995 = None
        view_1143: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_138, [1024, 8, 768]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_996: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_1143, [1, 0, 2]);  view_1143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_997: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0])
        clone_139: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_996, memory_format = torch.contiguous_format);  permute_996 = None
        view_1144: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_139, [8192, 768]);  clone_139 = None
        mm_39: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1144, permute_997);  permute_997 = None
        view_1145: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_39, [8, 1024, 768]);  mm_39 = None
        add_142: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1145, primals_155);  view_1145 = primals_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_7: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        gt_28: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 1e-30);  inductor_random_default_7 = None
        mul_129: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_28, add_142);  add_142 = None
        mul_130: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_129, 1.0);  mul_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_143: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_130, add_134);  mul_130 = add_134 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_36: "f32[8, 1024, 1]" = var_mean_18[0]
        getitem_37: "f32[8, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        add_144: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_78: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_143, getitem_37);  add_143 = getitem_37 = None
        mul_131: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_18);  sub_78 = None
        mul_132: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_131, primals_156)
        add_145: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_132, primals_157);  mul_132 = primals_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1146: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_145, [8192, 768])
        permute_998: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_158, [1, 0])
        addmm_18: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_159, view_1146, permute_998);  primals_159 = permute_998 = None
        view_1147: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_18, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_133: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1147, 0.5)
        mul_134: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1147, 0.7071067811865476);  view_1147 = None
        erf_9: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_134);  mul_134 = None
        add_146: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_135: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_133, add_146);  mul_133 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1148: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_135, [8192, 3072]);  mul_135 = None
        permute_999: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_160, [1, 0])
        addmm_19: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_161, view_1148, permute_999);  primals_161 = permute_999 = None
        view_1149: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_19, [8, 1024, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_6: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        gt_29: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 1e-30);  inductor_random_default_6 = None
        mul_136: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_29, view_1149);  view_1149 = None
        mul_137: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_136, 1.0);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_147: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_137, add_145);  mul_137 = add_145 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 1024, 1]" = var_mean_19[0]
        getitem_39: "f32[8, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        add_148: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_79: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_147, getitem_39);  add_147 = getitem_39 = None
        mul_138: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_19);  sub_79 = None
        mul_139: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_138, primals_162)
        add_149: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_139, primals_163);  mul_139 = primals_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1000: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_149, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_1001: "f32[768, 768]" = torch.ops.aten.permute.default(primals_164, [1, 0])
        clone_140: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_1000, memory_format = torch.contiguous_format);  permute_1000 = None
        view_1150: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_140, [8192, 768]);  clone_140 = None
        mm_40: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1150, permute_1001);  permute_1001 = None
        view_1151: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_40, [1024, 8, 768]);  mm_40 = None
        add_150: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1151, primals_165);  view_1151 = primals_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_1002: "f32[768, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0])
        mm_41: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1150, permute_1002);  permute_1002 = None
        view_1153: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_41, [1024, 8, 768]);  mm_41 = None
        add_151: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1153, primals_167);  view_1153 = primals_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_1003: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0])
        mm_42: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1150, permute_1003);  permute_1003 = None
        view_1155: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_42, [1024, 8, 768]);  mm_42 = None
        add_152: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1155, primals_169);  view_1155 = primals_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_100: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_150, 8.0);  add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1158: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_151, [1024, 8, 12, 64]);  add_151 = None
        permute_1005: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1158, [1, 0, 2, 3]);  view_1158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1010: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1005, [0, 2, 1, 3]);  permute_1005 = None
        view_1162: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_1010, [96, 1024, 64]);  permute_1010 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1169: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_1162, [96, 2, 512, 64]);  view_1162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_91: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_1169, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_212: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_91, 4);  as_strided_91 = None
        permute_1020: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_212, [0, 1, 4, 2, 3]);  unsqueeze_212 = None
        view_1179: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_100, [1024, 8, 12, 64]);  div_100 = None
        permute_1025: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1179, [1, 0, 2, 3]);  view_1179 = None
        permute_1026: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1025, [0, 2, 1, 3]);  permute_1025 = None
        view_1180: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_1026, [96, 1024, 64]);  permute_1026 = None
        view_1181: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_1180, [96, 2, 512, 64]);  view_1180 = None
        as_strided_95: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_1181, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_1181 = None
        unsqueeze_214: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_95, 4);  as_strided_95 = None
        clone_143: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_214, memory_format = torch.contiguous_format);  unsqueeze_214 = None
        view_1182: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_143, [288, 512, 64]);  clone_143 = None
        permute_1029: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_1020, [0, 1, 4, 3, 2]);  permute_1020 = None
        clone_144: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_1029, memory_format = torch.contiguous_format);  permute_1029 = None
        view_1183: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_144, [288, 64, 512]);  clone_144 = None
        bmm_20: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_1182, view_1183)
        view_1184: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_20, [96, 3, 512, 1, 512]);  bmm_20 = None
        permute_1030: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_1184, [0, 1, 2, 4, 3]);  view_1184 = None
        view_1185: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_1030, [96, 3, 512, 512]);  permute_1030 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_40: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1185, [0, 0, 0, 1], 0.0);  view_1185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1186: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_40, [96, 3, 512, 513]);  constant_pad_nd_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1341: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_1186, 2, 0, 256)
        slice_1342: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1341, 3, 0, 257);  slice_1341 = None
        copy_120: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_1342);  slice_1342 = None
        slice_scatter_220: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_120, 3, 256, 9223372036854775807);  copy_120 = None
        slice_scatter_221: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_220, 1, 0, -1);  slice_scatter_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_310: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_1186, 1, -1)
        slice_1351: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_310, 1, 256, 9223372036854775807);  select_310 = None
        slice_1352: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1351, 2, 0, 257);  slice_1351 = None
        select_313: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_221, 1, -1)
        slice_1354: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_313, 2, 256, 9223372036854775807)
        copy_121: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_1354, slice_1352);  slice_1354 = slice_1352 = None
        slice_scatter_222: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_313, copy_121, 2, 256, 9223372036854775807);  select_313 = copy_121 = None
        select_scatter_40: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_221, slice_scatter_222, 1, -1);  slice_scatter_221 = slice_scatter_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1357: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_1186, 2, -257, -1)
        slice_1358: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1357, 3, 257, 9223372036854775807);  slice_1357 = None
        slice_1362: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_40, 1, 1, 9223372036854775807)
        slice_1363: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1362, 3, 0, 256)
        copy_122: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1363, slice_1358);  slice_1363 = slice_1358 = None
        slice_scatter_223: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1362, copy_122, 3, 0, 256);  slice_1362 = copy_122 = None
        slice_scatter_224: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_40, slice_scatter_223, 1, 1, 9223372036854775807);  select_scatter_40 = slice_scatter_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_318: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_1186, 1, 0);  view_1186 = None
        slice_1370: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_318, 1, 0, 255);  select_318 = None
        slice_1371: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1370, 2, -255, 9223372036854775807);  slice_1370 = None
        select_322: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_224, 1, 0)
        slice_1375: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_322, 1, 1, 256)
        slice_1376: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1375, 2, 1, 256)
        copy_123: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_1376, slice_1371);  slice_1376 = slice_1371 = None
        slice_scatter_225: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1375, copy_123, 2, 1, 256);  slice_1375 = copy_123 = None
        slice_scatter_226: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_322, slice_scatter_225, 1, 1, 256);  select_322 = slice_scatter_225 = None
        select_scatter_41: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_224, slice_scatter_226, 1, 0);  slice_scatter_224 = slice_scatter_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1191: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_41, [8, 12, 1024, 513]);  select_scatter_41 = None
        permute_1034: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1191, [0, 2, 1, 3]);  view_1191 = None
        slice_1386: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1034, 1, 0, 256)
        slice_1387: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1386, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_81: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_1387)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_124: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1387, where_81);  slice_1387 = where_81 = None
        slice_scatter_227: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1386, copy_124, 3, 0, 257);  slice_1386 = copy_124 = None
        slice_scatter_228: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1034, slice_scatter_227, 1, 0, 256);  permute_1034 = slice_scatter_227 = None
        permute_1039: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_228, [0, 2, 1, 3]);  slice_scatter_228 = None
        view_1195: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1039, [96, 4, 256, 513]);  permute_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1205: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1195, [8, 12, 1024, 513]);  view_1195 = None
        permute_1047: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1205, [0, 2, 1, 3]);  view_1205 = None
        slice_1402: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1047, 1, -256, 9223372036854775807)
        slice_1403: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1402, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_82: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_1403)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_125: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1403, where_82);  slice_1403 = where_82 = None
        slice_scatter_229: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1402, copy_125, 3, -257, 9223372036854775807);  slice_1402 = copy_125 = None
        slice_scatter_230: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1047, slice_scatter_229, 1, -256, 9223372036854775807);  permute_1047 = slice_scatter_229 = None
        permute_1052: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_230, [0, 2, 1, 3]);  slice_scatter_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1081: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_1052, [0, 2, 1, 3]);  permute_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_155: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_1081, permute_82);  permute_1081 = None
        permute_1083: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_155, [0, 2, 1, 3]);  add_155 = None
        permute_1084: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_1083, [0, 2, 1, 3]);  permute_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_149: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1084, memory_format = torch.contiguous_format);  permute_1084 = None
        amax_10: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_149, [-1], True)
        sub_84: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_149, amax_10);  clone_149 = amax_10 = None
        exp_10: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
        sum_11: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_107: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_87: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_107)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_5: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_30: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 1e-30);  inductor_random_default_5 = None
        mul_141: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_30, where_87);  where_87 = None
        mul_142: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_141, 1.0);  mul_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1248: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_152, [1024, 8, 12, 64]);  add_152 = None
        permute_1086: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1248, [1, 0, 2, 3]);  view_1248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_1087: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_142, [0, 2, 1, 3]);  mul_142 = None
        clone_150: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1087, memory_format = torch.contiguous_format);  permute_1087 = None
        view_1249: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_150, [96, 4, 256, 513]);  clone_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1088: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1086, [0, 2, 1, 3]);  permute_1086 = None
        view_1250: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_1088, [96, 1024, 64]);  permute_1088 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_42: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_1250, [0, 0, 256, 256], -1.0);  view_1250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_98: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_42, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_42 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_43: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_1249, [0, 257], 0.0);  view_1249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1251: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_43, [96, 4, -1]);  constant_pad_nd_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1473: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_1251, 2, 0, -256);  view_1251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1252: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_1473, [96, 4, 256, 769]);  slice_1473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1474: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_1252, 3, 0, -1);  view_1252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_229: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1474, 4);  slice_1474 = None
        unsqueeze_230: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_98, 4);  as_strided_98 = None
        view_1253: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_229, [384, 256, 768]);  unsqueeze_229 = None
        clone_151: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_230, memory_format = torch.contiguous_format);  unsqueeze_230 = None
        view_1254: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_151, [384, 768, 64]);  clone_151 = None
        bmm_21: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_1253, view_1254)
        view_1255: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_21, [96, 4, 256, 1, 64]);  bmm_21 = None
        permute_1093: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_1255, [0, 1, 2, 4, 3]);  view_1255 = None
        view_1256: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_1093, [96, 4, 256, 64]);  permute_1093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1257: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1256, [8, 12, 1024, 64]);  view_1256 = None
        permute_1094: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1257, [0, 2, 1, 3]);  view_1257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_1095: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_1094, [1, 0, 2, 3]);  permute_1094 = None
        clone_152: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_1095, memory_format = torch.contiguous_format);  permute_1095 = None
        view_1258: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_152, [1024, 8, 768]);  clone_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1096: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_1258, [1, 0, 2]);  view_1258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_1097: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0])
        clone_153: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_1096, memory_format = torch.contiguous_format);  permute_1096 = None
        view_1259: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_153, [8192, 768]);  clone_153 = None
        mm_43: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1259, permute_1097);  permute_1097 = None
        view_1260: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_43, [8, 1024, 768]);  mm_43 = None
        add_157: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1260, primals_171);  view_1260 = primals_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_4: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_31: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 1e-30);  inductor_random_default_4 = None
        mul_143: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_31, add_157);  add_157 = None
        mul_144: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_143, 1.0);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_158: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_144, add_149);  mul_144 = add_149 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 1024, 1]" = var_mean_20[0]
        getitem_41: "f32[8, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        add_159: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_86: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_158, getitem_41);  add_158 = getitem_41 = None
        mul_145: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_20);  sub_86 = None
        mul_146: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_145, primals_172)
        add_160: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_146, primals_173);  mul_146 = primals_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1261: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_160, [8192, 768])
        permute_1098: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_174, [1, 0])
        addmm_20: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_175, view_1261, permute_1098);  primals_175 = permute_1098 = None
        view_1262: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_20, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_147: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1262, 0.5)
        mul_148: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1262, 0.7071067811865476);  view_1262 = None
        erf_10: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_161: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_149: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_147, add_161);  mul_147 = add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1263: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_149, [8192, 3072]);  mul_149 = None
        permute_1099: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_176, [1, 0])
        addmm_21: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_177, view_1263, permute_1099);  primals_177 = permute_1099 = None
        view_1264: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_21, [8, 1024, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_3: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        gt_32: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 1e-30);  inductor_random_default_3 = None
        mul_150: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_32, view_1264);  view_1264 = None
        mul_151: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_150, 1.0);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_162: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_151, add_160);  mul_151 = add_160 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_162, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 1024, 1]" = var_mean_21[0]
        getitem_43: "f32[8, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        add_163: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        sub_87: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_162, getitem_43);  add_162 = getitem_43 = None
        mul_152: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_21);  sub_87 = None
        mul_153: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_152, primals_178)
        add_164: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_153, primals_179);  mul_153 = primals_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1100: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_164, [1, 0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_1101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_180, [1, 0])
        clone_154: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_1100, memory_format = torch.contiguous_format);  permute_1100 = None
        view_1265: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_154, [8192, 768]);  clone_154 = None
        mm_44: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1265, permute_1101);  permute_1101 = None
        view_1266: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_44, [1024, 8, 768]);  mm_44 = None
        add_165: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1266, primals_181);  view_1266 = primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_1102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0])
        mm_45: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1265, permute_1102);  permute_1102 = None
        view_1268: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_45, [1024, 8, 768]);  mm_45 = None
        add_166: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1268, primals_183);  view_1268 = primals_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_1103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0])
        mm_46: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1265, permute_1103);  permute_1103 = None
        view_1270: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_46, [1024, 8, 768]);  mm_46 = None
        add_167: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1270, primals_185);  view_1270 = primals_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_110: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_165, 8.0);  add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1273: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_166, [1024, 8, 12, 64]);  add_166 = None
        permute_1105: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1273, [1, 0, 2, 3]);  view_1273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1110: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1105, [0, 2, 1, 3]);  permute_1105 = None
        view_1277: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_1110, [96, 1024, 64]);  permute_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        view_1284: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_1277, [96, 2, 512, 64]);  view_1277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_100: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_1284, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_1284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_233: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_100, 4);  as_strided_100 = None
        permute_1120: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_233, [0, 1, 4, 2, 3]);  unsqueeze_233 = None
        view_1294: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_110, [1024, 8, 12, 64]);  div_110 = None
        permute_1125: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1294, [1, 0, 2, 3]);  view_1294 = None
        permute_1126: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1125, [0, 2, 1, 3]);  permute_1125 = None
        view_1295: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_1126, [96, 1024, 64]);  permute_1126 = None
        view_1296: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(view_1295, [96, 2, 512, 64]);  view_1295 = None
        as_strided_104: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_1296, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  view_1296 = None
        unsqueeze_235: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_104, 4);  as_strided_104 = None
        clone_157: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_235, memory_format = torch.contiguous_format);  unsqueeze_235 = None
        view_1297: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_157, [288, 512, 64]);  clone_157 = None
        permute_1129: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_1120, [0, 1, 4, 3, 2]);  permute_1120 = None
        clone_158: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_1129, memory_format = torch.contiguous_format);  permute_1129 = None
        view_1298: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_158, [288, 64, 512]);  clone_158 = None
        bmm_22: "f32[288, 512, 512]" = torch.ops.aten.bmm.default(view_1297, view_1298)
        view_1299: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_22, [96, 3, 512, 1, 512]);  bmm_22 = None
        permute_1130: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_1299, [0, 1, 2, 4, 3]);  view_1299 = None
        view_1300: "f32[96, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_1130, [96, 3, 512, 512]);  permute_1130 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_44: "f32[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1300, [0, 0, 0, 1], 0.0);  view_1300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1301: "f32[96, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_44, [96, 3, 512, 513]);  constant_pad_nd_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1475: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_1301, 2, 0, 256)
        slice_1476: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1475, 3, 0, 257);  slice_1475 = None
        copy_132: "f32[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_1476);  slice_4 = slice_1476 = None
        slice_scatter_242: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_132, 3, 256, 9223372036854775807);  slice_3 = copy_132 = None
        slice_scatter_243: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_242, 1, 0, -1);  full = slice_scatter_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_341: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_1301, 1, -1)
        slice_1485: "f32[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_341, 1, 256, 9223372036854775807);  select_341 = None
        slice_1486: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1485, 2, 0, 257);  slice_1485 = None
        select_344: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_243, 1, -1)
        slice_1488: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_344, 2, 256, 9223372036854775807)
        copy_133: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_1488, slice_1486);  slice_1488 = slice_1486 = None
        slice_scatter_244: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_344, copy_133, 2, 256, 9223372036854775807);  select_344 = copy_133 = None
        select_scatter_44: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_243, slice_scatter_244, 1, -1);  slice_scatter_243 = slice_scatter_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1491: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_1301, 2, -257, -1)
        slice_1492: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1491, 3, 257, 9223372036854775807);  slice_1491 = None
        slice_1496: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_44, 1, 1, 9223372036854775807)
        slice_1497: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1496, 3, 0, 256)
        copy_134: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1497, slice_1492);  slice_1497 = slice_1492 = None
        slice_scatter_245: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1496, copy_134, 3, 0, 256);  slice_1496 = copy_134 = None
        slice_scatter_246: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_44, slice_scatter_245, 1, 1, 9223372036854775807);  select_scatter_44 = slice_scatter_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_349: "f32[96, 512, 513]" = torch.ops.aten.select.int(view_1301, 1, 0);  view_1301 = None
        slice_1504: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_349, 1, 0, 255);  select_349 = None
        slice_1505: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1504, 2, -255, 9223372036854775807);  slice_1504 = None
        select_353: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_246, 1, 0)
        slice_1509: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_353, 1, 1, 256)
        slice_1510: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1509, 2, 1, 256)
        copy_135: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_1510, slice_1505);  slice_1510 = slice_1505 = None
        slice_scatter_247: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1509, copy_135, 2, 1, 256);  slice_1509 = copy_135 = None
        slice_scatter_248: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_353, slice_scatter_247, 1, 1, 256);  select_353 = slice_scatter_247 = None
        select_scatter_45: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_246, slice_scatter_248, 1, 0);  slice_scatter_246 = slice_scatter_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        view_1306: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_45, [8, 12, 1024, 513]);  select_scatter_45 = None
        permute_1134: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1306, [0, 2, 1, 3]);  view_1306 = None
        slice_1520: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1134, 1, 0, 256)
        slice_1521: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1520, 3, 0, 257)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_89: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_35, slice_1521);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_136: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1521, where_89);  slice_1521 = where_89 = None
        slice_scatter_249: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1520, copy_136, 3, 0, 257);  slice_1520 = copy_136 = None
        slice_scatter_250: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1134, slice_scatter_249, 1, 0, 256);  permute_1134 = slice_scatter_249 = None
        permute_1139: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_250, [0, 2, 1, 3]);  slice_scatter_250 = None
        view_1310: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1139, [96, 4, 256, 513]);  permute_1139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        view_1320: "f32[8, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1310, [8, 12, 1024, 513]);  view_1310 = None
        permute_1147: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1320, [0, 2, 1, 3]);  view_1320 = None
        slice_1536: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1147, 1, -256, 9223372036854775807)
        slice_1537: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1536, 3, -257, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_90: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_35, slice_1537);  convert_element_type_1 = permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_137: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1537, where_90);  slice_1537 = where_90 = None
        slice_scatter_251: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1536, copy_137, 3, -257, 9223372036854775807);  slice_1536 = copy_137 = None
        slice_scatter_252: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1147, slice_scatter_251, 1, -256, 9223372036854775807);  permute_1147 = slice_scatter_251 = None
        permute_1152: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_252, [0, 2, 1, 3]);  slice_scatter_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1181: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_1152, [0, 2, 1, 3]);  permute_1152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_170: "f32[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_1181, permute_82);  permute_1181 = permute_82 = None
        permute_1183: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_170, [0, 2, 1, 3]);  add_170 = None
        permute_1184: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_1183, [0, 2, 1, 3]);  permute_1183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        clone_163: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1184, memory_format = torch.contiguous_format);  permute_1184 = None
        amax_11: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_163, [-1], True)
        sub_92: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_163, amax_11);  clone_163 = amax_11 = None
        exp_11: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_92);  sub_92 = None
        sum_12: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_117: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_95: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, div_117);  unsqueeze_18 = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_2: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default([8, 1024, 12, 513], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_33: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 1e-30);  inductor_random_default_2 = None
        mul_155: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_33, where_95);  where_95 = None
        mul_156: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_155, 1.0);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_1363: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_167, [1024, 8, 12, 64]);  add_167 = None
        permute_1186: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1363, [1, 0, 2, 3]);  view_1363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_1187: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_156, [0, 2, 1, 3]);  mul_156 = None
        clone_164: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1187, memory_format = torch.contiguous_format);  permute_1187 = None
        view_1364: "f32[96, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_164, [96, 4, 256, 513]);  clone_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_1188: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1186, [0, 2, 1, 3]);  permute_1186 = None
        view_1365: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_1188, [96, 1024, 64]);  permute_1188 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_46: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_1365, [0, 0, 256, 256], -1.0);  view_1365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_107: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_46, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_46 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_47: "f32[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_1364, [0, 257], 0.0);  view_1364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1366: "f32[96, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_47, [96, 4, -1]);  constant_pad_nd_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1607: "f32[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_1366, 2, 0, -256);  view_1366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1367: "f32[96, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_1607, [96, 4, 256, 769]);  slice_1607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1608: "f32[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_1367, 3, 0, -1);  view_1367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_250: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1608, 4);  slice_1608 = None
        unsqueeze_251: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_107, 4);  as_strided_107 = None
        view_1368: "f32[384, 256, 768]" = torch.ops.aten.reshape.default(unsqueeze_250, [384, 256, 768]);  unsqueeze_250 = None
        clone_165: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_251, memory_format = torch.contiguous_format);  unsqueeze_251 = None
        view_1369: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_165, [384, 768, 64]);  clone_165 = None
        bmm_23: "f32[384, 256, 64]" = torch.ops.aten.bmm.default(view_1368, view_1369)
        view_1370: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_23, [96, 4, 256, 1, 64]);  bmm_23 = None
        permute_1193: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_1370, [0, 1, 2, 4, 3]);  view_1370 = None
        view_1371: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_1193, [96, 4, 256, 64]);  permute_1193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_1372: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1371, [8, 12, 1024, 64]);  view_1371 = None
        permute_1194: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1372, [0, 2, 1, 3]);  view_1372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_1195: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_1194, [1, 0, 2, 3]);  permute_1194 = None
        clone_166: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_1195, memory_format = torch.contiguous_format);  permute_1195 = None
        view_1373: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_166, [1024, 8, 768]);  clone_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1196: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(view_1373, [1, 0, 2]);  view_1373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        permute_1197: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0])
        clone_167: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_1196, memory_format = torch.contiguous_format);  permute_1196 = None
        view_1374: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_167, [8192, 768]);  clone_167 = None
        mm_47: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1374, permute_1197);  permute_1197 = None
        view_1375: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_47, [8, 1024, 768]);  mm_47 = None
        add_172: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1375, primals_187);  view_1375 = primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_1: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_34: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 1e-30);  inductor_random_default_1 = None
        mul_157: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_34, add_172);  add_172 = None
        mul_158: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_157, 1.0);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_173: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_158, add_164);  mul_158 = add_164 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 1024, 1]" = var_mean_22[0]
        getitem_45: "f32[8, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        add_174: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_94: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_173, getitem_45);  add_173 = getitem_45 = None
        mul_159: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_22);  sub_94 = None
        mul_160: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_159, primals_188)
        add_175: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_160, primals_189);  mul_160 = primals_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1376: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_175, [8192, 768])
        permute_1198: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_190, [1, 0])
        addmm_22: "f32[8192, 3072]" = torch.ops.aten.addmm.default(primals_191, view_1376, permute_1198);  primals_191 = permute_1198 = None
        view_1377: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 1024, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_161: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1377, 0.5)
        mul_162: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1377, 0.7071067811865476);  view_1377 = None
        erf_11: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_162);  mul_162 = None
        add_176: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_163: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_161, add_176);  mul_161 = add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1378: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_163, [8192, 3072]);  mul_163 = None
        permute_1199: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_192, [1, 0])
        addmm_23: "f32[8192, 768]" = torch.ops.aten.addmm.default(primals_193, view_1378, permute_1199);  primals_193 = permute_1199 = None
        view_1379: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_23, [8, 1024, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_35: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 1e-30);  inductor_random_default = None
        mul_164: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_35, view_1379);  view_1379 = None
        mul_165: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_164, 1.0);  mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_177: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_165, add_175);  mul_165 = add_175 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 1024, 1]" = var_mean_23[0]
        getitem_47: "f32[8, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        add_178: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_95: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_177, getitem_47);  add_177 = getitem_47 = None
        mul_166: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_23);  sub_95 = None
        mul_167: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_166, primals_194)
        add_179: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_167, primals_195);  mul_167 = primals_195 = None
        div_120: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_121: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_1216: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_1368, [0, 2, 1]);  view_1368 = None
        permute_1217: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_1369, [0, 2, 1]);  view_1369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_1247: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_1297, [0, 2, 1]);  view_1297 = None
        permute_1248: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_1298, [0, 2, 1]);  view_1298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_123: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_124: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_1301: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_1253, [0, 2, 1]);  view_1253 = None
        permute_1302: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_1254, [0, 2, 1]);  view_1254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_1332: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_1182, [0, 2, 1]);  view_1182 = None
        permute_1333: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_1183, [0, 2, 1]);  view_1183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_126: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_127: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_1386: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_1138, [0, 2, 1]);  view_1138 = None
        permute_1387: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_1139, [0, 2, 1]);  view_1139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_1417: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_1067, [0, 2, 1]);  view_1067 = None
        permute_1418: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_1068, [0, 2, 1]);  view_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_129: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_130: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_1471: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_1023, [0, 2, 1]);  view_1023 = None
        permute_1472: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_1024, [0, 2, 1]);  view_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_1502: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_952, [0, 2, 1]);  view_952 = None
        permute_1503: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_953, [0, 2, 1]);  view_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_132: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_133: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_1556: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_908, [0, 2, 1]);  view_908 = None
        permute_1557: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_909, [0, 2, 1]);  view_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_1587: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_837, [0, 2, 1]);  view_837 = None
        permute_1588: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_838, [0, 2, 1]);  view_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_135: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_136: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_1641: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_793, [0, 2, 1]);  view_793 = None
        permute_1642: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_794, [0, 2, 1]);  view_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_1672: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_722, [0, 2, 1]);  view_722 = None
        permute_1673: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_723, [0, 2, 1]);  view_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_138: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_139: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_1726: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_678, [0, 2, 1]);  view_678 = None
        permute_1727: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_679, [0, 2, 1]);  view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_1757: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_607, [0, 2, 1]);  view_607 = None
        permute_1758: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1]);  view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_141: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_142: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_1811: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_563, [0, 2, 1]);  view_563 = None
        permute_1812: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_564, [0, 2, 1]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_1842: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_492, [0, 2, 1]);  view_492 = None
        permute_1843: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_493, [0, 2, 1]);  view_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_144: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_145: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_1896: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
        permute_1897: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_1927: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
        permute_1928: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_147: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_148: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_1981: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
        permute_1982: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_2012: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_262, [0, 2, 1]);  view_262 = None
        permute_2013: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_150: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_151: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_2066: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
        permute_2067: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_2097: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_2098: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_153: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_154: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_2151: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
        permute_2152: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_104, [0, 2, 1]);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_2182: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        permute_2183: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        return (add_179, primals_2, primals_4, primals_6, primals_9, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, primals_156, primals_158, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, view, unsqueeze_8, rev_1, div_7, gt, view_109, gt_1, mul_5, view_111, addmm, view_113, gt_2, mul_12, view_115, div_17, gt_3, view_224, gt_4, mul_19, view_226, addmm_2, view_228, gt_5, mul_26, view_230, div_27, gt_6, view_339, gt_7, mul_33, view_341, addmm_4, view_343, gt_8, mul_40, view_345, div_37, gt_9, view_454, gt_10, mul_47, view_456, addmm_6, view_458, gt_11, mul_54, view_460, div_47, gt_12, view_569, gt_13, mul_61, view_571, addmm_8, view_573, gt_14, mul_68, view_575, div_57, gt_15, view_684, gt_16, mul_75, view_686, addmm_10, view_688, gt_17, mul_82, view_690, div_67, gt_18, view_799, gt_19, mul_89, view_801, addmm_12, view_803, gt_20, mul_96, view_805, div_77, gt_21, view_914, gt_22, mul_103, view_916, addmm_14, view_918, gt_23, mul_110, view_920, div_87, gt_24, view_1029, gt_25, mul_117, view_1031, addmm_16, view_1033, gt_26, mul_124, view_1035, div_97, gt_27, view_1144, gt_28, mul_131, view_1146, addmm_18, view_1148, gt_29, mul_138, view_1150, div_107, gt_30, view_1259, gt_31, mul_145, view_1261, addmm_20, view_1263, gt_32, mul_152, view_1265, div_117, gt_33, view_1374, gt_34, mul_159, view_1376, addmm_22, view_1378, gt_35, mul_166, div_120, div_121, permute_1216, permute_1217, permute_1247, permute_1248, div_123, div_124, permute_1301, permute_1302, permute_1332, permute_1333, div_126, div_127, permute_1386, permute_1387, permute_1417, permute_1418, div_129, div_130, permute_1471, permute_1472, permute_1502, permute_1503, div_132, div_133, permute_1556, permute_1557, permute_1587, permute_1588, div_135, div_136, permute_1641, permute_1642, permute_1672, permute_1673, div_138, div_139, permute_1726, permute_1727, permute_1757, permute_1758, div_141, div_142, permute_1811, permute_1812, permute_1842, permute_1843, div_144, div_145, permute_1896, permute_1897, permute_1927, permute_1928, div_147, div_148, permute_1981, permute_1982, permute_2012, permute_2013, div_150, div_151, permute_2066, permute_2067, permute_2097, permute_2098, div_153, div_154, permute_2151, permute_2152, permute_2182, permute_2183)

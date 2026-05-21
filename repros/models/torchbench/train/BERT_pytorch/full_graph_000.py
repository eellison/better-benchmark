class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[128, 128]", primals_2: "f32[20005, 768]", primals_3: "f32[1, 512, 768]", primals_4: "f32[3, 768]", primals_5: "i64[128, 128]", primals_6: "f32[768]", primals_7: "f32[768]", primals_8: "f32[768, 768]", primals_9: "f32[768]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_13: "f32[768]", primals_14: "f32[768, 768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[3072, 768]", primals_19: "f32[3072]", primals_20: "f32[768, 3072]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768, 768]", primals_25: "f32[768]", primals_26: "f32[768, 768]", primals_27: "f32[768]", primals_28: "f32[768, 768]", primals_29: "f32[768]", primals_30: "f32[768, 768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[3072, 768]", primals_35: "f32[3072]", primals_36: "f32[768, 3072]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768, 768]", primals_41: "f32[768]", primals_42: "f32[768, 768]", primals_43: "f32[768]", primals_44: "f32[768, 768]", primals_45: "f32[768]", primals_46: "f32[768, 768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[3072, 768]", primals_51: "f32[3072]", primals_52: "f32[768, 3072]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768, 768]", primals_57: "f32[768]", primals_58: "f32[768, 768]", primals_59: "f32[768]", primals_60: "f32[768, 768]", primals_61: "f32[768]", primals_62: "f32[768, 768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[3072, 768]", primals_67: "f32[3072]", primals_68: "f32[768, 3072]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768, 768]", primals_73: "f32[768]", primals_74: "f32[768, 768]", primals_75: "f32[768]", primals_76: "f32[768, 768]", primals_77: "f32[768]", primals_78: "f32[768, 768]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[3072, 768]", primals_83: "f32[3072]", primals_84: "f32[768, 3072]", primals_85: "f32[768]", primals_86: "f32[768]", primals_87: "f32[768]", primals_88: "f32[768, 768]", primals_89: "f32[768]", primals_90: "f32[768, 768]", primals_91: "f32[768]", primals_92: "f32[768, 768]", primals_93: "f32[768]", primals_94: "f32[768, 768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[768]", primals_98: "f32[3072, 768]", primals_99: "f32[3072]", primals_100: "f32[768, 3072]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[768, 768]", primals_105: "f32[768]", primals_106: "f32[768, 768]", primals_107: "f32[768]", primals_108: "f32[768, 768]", primals_109: "f32[768]", primals_110: "f32[768, 768]", primals_111: "f32[768]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[3072, 768]", primals_115: "f32[3072]", primals_116: "f32[768, 3072]", primals_117: "f32[768]", primals_118: "f32[768]", primals_119: "f32[768]", primals_120: "f32[768, 768]", primals_121: "f32[768]", primals_122: "f32[768, 768]", primals_123: "f32[768]", primals_124: "f32[768, 768]", primals_125: "f32[768]", primals_126: "f32[768, 768]", primals_127: "f32[768]", primals_128: "f32[768]", primals_129: "f32[768]", primals_130: "f32[3072, 768]", primals_131: "f32[3072]", primals_132: "f32[768, 3072]", primals_133: "f32[768]", primals_134: "f32[768]", primals_135: "f32[768]", primals_136: "f32[768, 768]", primals_137: "f32[768]", primals_138: "f32[768, 768]", primals_139: "f32[768]", primals_140: "f32[768, 768]", primals_141: "f32[768]", primals_142: "f32[768, 768]", primals_143: "f32[768]", primals_144: "f32[768]", primals_145: "f32[768]", primals_146: "f32[3072, 768]", primals_147: "f32[3072]", primals_148: "f32[768, 3072]", primals_149: "f32[768]", primals_150: "f32[768]", primals_151: "f32[768]", primals_152: "f32[768, 768]", primals_153: "f32[768]", primals_154: "f32[768, 768]", primals_155: "f32[768]", primals_156: "f32[768, 768]", primals_157: "f32[768]", primals_158: "f32[768, 768]", primals_159: "f32[768]", primals_160: "f32[768]", primals_161: "f32[768]", primals_162: "f32[3072, 768]", primals_163: "f32[3072]", primals_164: "f32[768, 3072]", primals_165: "f32[768]", primals_166: "f32[768]", primals_167: "f32[768]", primals_168: "f32[768, 768]", primals_169: "f32[768]", primals_170: "f32[768, 768]", primals_171: "f32[768]", primals_172: "f32[768, 768]", primals_173: "f32[768]", primals_174: "f32[768, 768]", primals_175: "f32[768]", primals_176: "f32[768]", primals_177: "f32[768]", primals_178: "f32[3072, 768]", primals_179: "f32[3072]", primals_180: "f32[768, 3072]", primals_181: "f32[768]", primals_182: "f32[768]", primals_183: "f32[768]", primals_184: "f32[768, 768]", primals_185: "f32[768]", primals_186: "f32[768, 768]", primals_187: "f32[768]", primals_188: "f32[768, 768]", primals_189: "f32[768]", primals_190: "f32[768, 768]", primals_191: "f32[768]", primals_192: "f32[768]", primals_193: "f32[768]", primals_194: "f32[3072, 768]", primals_195: "f32[3072]", primals_196: "f32[768, 3072]", primals_197: "f32[768]", primals_198: "f32[2, 768]", primals_199: "f32[2]", primals_200: "f32[20005, 768]", primals_201: "f32[20005]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/bert.py:43 in forward, code: mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        gt: "b8[128, 128]" = torch.ops.aten.gt.Scalar(primals_1, 0)
        unsqueeze: "b8[128, 1, 128]" = torch.ops.aten.unsqueeze.default(gt, 1);  gt = None
        repeat: "b8[128, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze, [1, 128, 1]);  unsqueeze = None
        unsqueeze_1: "b8[128, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(repeat, 1);  repeat = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        embedding: "f32[128, 128, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1, 0);  primals_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/position.py:28 in forward, code: return self.pe[:, : x.size(1)]
        slice_1: "f32[1, 128, 768]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128);  primals_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        add: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(embedding, slice_1);  embedding = slice_1 = None
        embedding_1: "f32[128, 128, 768]" = torch.ops.aten.embedding.default(primals_4, primals_5, 0);  primals_4 = None
        add_1: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add, embedding_1);  add = embedding_1 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[61]" = torch.ops.prims.inductor_seeds.default(61, device(type='cuda', index=0))

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:33 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_60: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_1: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_60, 0.1);  inductor_random_default_60 = None
        mul: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_1, add_1);  add_1 = None
        mul_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_1, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_1, [-1], correction = 1.0, keepdim = True)
        sqrt: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var);  var = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_1, mean);  mean = None
        mul_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_6, sub)
        add_2: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt, 1e-06)
        div: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        add_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div, primals_7);  div = primals_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_3, [16384, 768]);  add_3 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_9, view, permute);  primals_9 = permute = None
        view_1: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm, [128, 128, 768]);  addmm = None
        view_2: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_1, [128, -1, 12, 64]);  view_1 = None
        permute_1: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm_1: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_11, view, permute_2);  primals_11 = permute_2 = None
        view_4: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_1, [128, 128, 768]);  addmm_1 = None
        view_5: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_4, [128, -1, 12, 64]);  view_4 = None
        permute_3: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm_2: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_13, view, permute_4);  primals_13 = permute_4 = None
        view_7: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_2, [128, 128, 768]);  addmm_2 = None
        view_8: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_7, [128, -1, 12, 64]);  view_7 = None
        permute_5: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_6: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_1, [128, 12, 128, 64]);  permute_1 = None
        clone: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_9: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone, [1536, 128, 64]);  clone = None
        expand_1: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_6, [128, 12, 64, 128]);  permute_6 = None
        clone_1: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_1, [1536, 64, 128]);  clone_1 = None
        bmm: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_10)
        view_11: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm, [128, 12, 128, 128])
        div_1: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        eq: "b8[128, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_default: "f32[]" = torch.ops.aten.full.default([], -1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_1);  div_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where, [-1], True)
        sub_1: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = None
        exp: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_59: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_2: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_59, 0.1);  inductor_random_default_59 = None
        mul_3: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_2, div_2);  div_2 = None
        mul_4: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_2: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_4, [128, 12, 128, 128]);  mul_4 = None
        view_12: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_2, [1536, 128, 128]);  expand_2 = None
        expand_3: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_5, [128, 12, 128, 64]);  permute_5 = None
        clone_2: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_13: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_2, [1536, 128, 64]);  clone_2 = None
        bmm_1: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
        view_14: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_1, [128, 12, 128, 64]);  bmm_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_7: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_3, [128, -1, 768]);  clone_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_16: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_15, [16384, 768]);  view_15 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_3: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_15, view_16, permute_8);  primals_15 = permute_8 = None
        view_17: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_3, [128, 128, 768]);  addmm_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_58: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_3: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_58, 0.1);  inductor_random_default_58 = None
        mul_5: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_3, view_17);  view_17 = None
        mul_6: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        add_4: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, mul_6);  mul_1 = mul_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_1: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_4, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_1: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_4, [-1], correction = 1.0, keepdim = True)
        sqrt_1: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_1);  var_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_2: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_4, mean_1);  mean_1 = None
        mul_7: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_16, sub_2)
        add_5: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_1, 1e-06)
        div_3: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_7, add_5);  mul_7 = add_5 = None
        add_6: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_3, primals_17);  div_3 = primals_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_18: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_6, [16384, 768]);  add_6 = None
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
        addmm_4: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_19, view_18, permute_9);  primals_19 = permute_9 = None
        view_19: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_4, [128, 128, 3072])
        mul_8: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_9: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_7: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_10: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_8, add_7);  mul_8 = add_7 = None
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_57: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_4: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_57, 0.1);  inductor_random_default_57 = None
        mul_11: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_4, mul_10);  mul_10 = None
        mul_12: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_11, 1.1111111111111112);  mul_11 = None
        view_20: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_12, [16384, 3072]);  mul_12 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
        addmm_5: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_21, view_20, permute_10);  primals_21 = permute_10 = None
        view_21: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_5, [128, 128, 768]);  addmm_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_56: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_5: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_56, 0.1);  inductor_random_default_56 = None
        mul_13: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_5, view_21);  view_21 = None
        mul_14: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_13, 1.1111111111111112);  mul_13 = None
        add_8: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_4, mul_14);  add_4 = mul_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_55: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_6: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_55, 0.1);  inductor_random_default_55 = None
        mul_15: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_6, add_8);  add_8 = None
        mul_16: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_15, 1.1111111111111112);  mul_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_2: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_16, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_2: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_16, [-1], correction = 1.0, keepdim = True)
        sqrt_2: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_2);  var_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_3: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_16, mean_2);  mean_2 = None
        mul_17: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_22, sub_3)
        add_9: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_2, 1e-06)
        div_4: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_17, add_9);  mul_17 = add_9 = None
        add_10: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_4, primals_23);  div_4 = primals_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_22: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_10, [16384, 768]);  add_10 = None
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0])
        addmm_6: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_25, view_22, permute_11);  primals_25 = permute_11 = None
        view_23: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_6, [128, 128, 768]);  addmm_6 = None
        view_24: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_23, [128, -1, 12, 64]);  view_23 = None
        permute_12: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0])
        addmm_7: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_27, view_22, permute_13);  primals_27 = permute_13 = None
        view_26: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_7, [128, 128, 768]);  addmm_7 = None
        view_27: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_26, [128, -1, 12, 64]);  view_26 = None
        permute_14: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0])
        addmm_8: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_29, view_22, permute_15);  primals_29 = permute_15 = None
        view_29: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_8, [128, 128, 768]);  addmm_8 = None
        view_30: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_29, [128, -1, 12, 64]);  view_29 = None
        permute_16: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_17: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        expand_4: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_12, [128, 12, 128, 64]);  permute_12 = None
        clone_4: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_31: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_4, [1536, 128, 64]);  clone_4 = None
        expand_5: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_17, [128, 12, 64, 128]);  permute_17 = None
        clone_5: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_32: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_5, [1536, 64, 128]);  clone_5 = None
        bmm_2: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_31, view_32)
        view_33: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [128, 12, 128, 128]);  bmm_2 = None
        div_5: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_1: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_5);  div_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_1: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
        sub_4: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
        exp_1: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_6: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_54: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_7: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_54, 0.1);  inductor_random_default_54 = None
        mul_18: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_7, div_6)
        mul_19: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_18, 1.1111111111111112);  mul_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_6: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_19, [128, 12, 128, 128]);  mul_19 = None
        view_34: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_6, [1536, 128, 128]);  expand_6 = None
        expand_7: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_16, [128, 12, 128, 64]);  permute_16 = None
        clone_6: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_35: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_6, [1536, 128, 64]);  clone_6 = None
        bmm_3: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_34, view_35)
        view_36: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_3, [128, 12, 128, 64]);  bmm_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_18: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_7, [128, -1, 768]);  clone_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_38: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_37, [16384, 768]);  view_37 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        addmm_9: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_31, view_38, permute_19);  primals_31 = permute_19 = None
        view_39: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_9, [128, 128, 768]);  addmm_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_53: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_8: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_53, 0.1);  inductor_random_default_53 = None
        mul_20: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_8, view_39);  view_39 = None
        mul_21: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        add_11: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_16, mul_21);  mul_16 = mul_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_3: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_11, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_3: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_11, [-1], correction = 1.0, keepdim = True)
        sqrt_3: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_3);  var_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_5: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, mean_3);  mean_3 = None
        mul_22: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_32, sub_5)
        add_12: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_3, 1e-06)
        div_7: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_22, add_12);  mul_22 = add_12 = None
        add_13: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_7, primals_33);  div_7 = primals_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_40: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_13, [16384, 768]);  add_13 = None
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_34, [1, 0])
        addmm_10: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_35, view_40, permute_20);  primals_35 = permute_20 = None
        view_41: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_10, [128, 128, 3072])
        mul_23: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_24: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_14: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_25: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_23, add_14);  mul_23 = add_14 = None
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_52: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_9: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_52, 0.1);  inductor_random_default_52 = None
        mul_26: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_9, mul_25);  mul_25 = None
        mul_27: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_26, 1.1111111111111112);  mul_26 = None
        view_42: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_27, [16384, 3072]);  mul_27 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0])
        addmm_11: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_37, view_42, permute_21);  primals_37 = permute_21 = None
        view_43: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_11, [128, 128, 768]);  addmm_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_51: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_10: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_51, 0.1);  inductor_random_default_51 = None
        mul_28: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_10, view_43);  view_43 = None
        mul_29: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_28, 1.1111111111111112);  mul_28 = None
        add_15: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_11, mul_29);  add_11 = mul_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_50: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_11: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_50, 0.1);  inductor_random_default_50 = None
        mul_30: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_11, add_15);  add_15 = None
        mul_31: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_4: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_31, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_4: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_31, [-1], correction = 1.0, keepdim = True)
        sqrt_4: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_4);  var_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_6: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_31, mean_4);  mean_4 = None
        mul_32: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_38, sub_6)
        add_16: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_4, 1e-06)
        div_8: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_32, add_16);  mul_32 = add_16 = None
        add_17: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_8, primals_39);  div_8 = primals_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_44: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_17, [16384, 768]);  add_17 = None
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0])
        addmm_12: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_41, view_44, permute_22);  primals_41 = permute_22 = None
        view_45: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_12, [128, 128, 768]);  addmm_12 = None
        view_46: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_45, [128, -1, 12, 64]);  view_45 = None
        permute_23: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0])
        addmm_13: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_43, view_44, permute_24);  primals_43 = permute_24 = None
        view_48: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_13, [128, 128, 768]);  addmm_13 = None
        view_49: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_48, [128, -1, 12, 64]);  view_48 = None
        permute_25: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0])
        addmm_14: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_45, view_44, permute_26);  primals_45 = permute_26 = None
        view_51: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_14, [128, 128, 768]);  addmm_14 = None
        view_52: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_51, [128, -1, 12, 64]);  view_51 = None
        permute_27: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_28: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        expand_8: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_23, [128, 12, 128, 64]);  permute_23 = None
        clone_8: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_53: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_8, [1536, 128, 64]);  clone_8 = None
        expand_9: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_28, [128, 12, 64, 128]);  permute_28 = None
        clone_9: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_54: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_9, [1536, 64, 128]);  clone_9 = None
        bmm_4: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_53, view_54)
        view_55: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [128, 12, 128, 128]);  bmm_4 = None
        div_9: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_2: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_9);  div_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_2: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
        sub_7: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
        exp_2: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_10: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_49: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_12: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_49, 0.1);  inductor_random_default_49 = None
        mul_33: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_12, div_10)
        mul_34: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_33, 1.1111111111111112);  mul_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_10: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_34, [128, 12, 128, 128]);  mul_34 = None
        view_56: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_10, [1536, 128, 128]);  expand_10 = None
        expand_11: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_27, [128, 12, 128, 64]);  permute_27 = None
        clone_10: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_57: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_10, [1536, 128, 64]);  clone_10 = None
        bmm_5: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_56, view_57)
        view_58: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [128, 12, 128, 64]);  bmm_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_29: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_11: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_11, [128, -1, 768]);  clone_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_60: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_59, [16384, 768]);  view_59 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0])
        addmm_15: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_47, view_60, permute_30);  primals_47 = permute_30 = None
        view_61: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_15, [128, 128, 768]);  addmm_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_48: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_13: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_48, 0.1);  inductor_random_default_48 = None
        mul_35: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_13, view_61);  view_61 = None
        mul_36: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_35, 1.1111111111111112);  mul_35 = None
        add_18: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_31, mul_36);  mul_31 = mul_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_5: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_18, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_5: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_18, [-1], correction = 1.0, keepdim = True)
        sqrt_5: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_5);  var_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_8: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_18, mean_5);  mean_5 = None
        mul_37: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_48, sub_8)
        add_19: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_5, 1e-06)
        div_11: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_37, add_19);  mul_37 = add_19 = None
        add_20: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_11, primals_49);  div_11 = primals_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_62: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_20, [16384, 768]);  add_20 = None
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_50, [1, 0])
        addmm_16: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_51, view_62, permute_31);  primals_51 = permute_31 = None
        view_63: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_16, [128, 128, 3072])
        mul_38: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_39: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_21: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_40: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_38, add_21);  mul_38 = add_21 = None
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_47: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_14: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_47, 0.1);  inductor_random_default_47 = None
        mul_41: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_14, mul_40);  mul_40 = None
        mul_42: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_41, 1.1111111111111112);  mul_41 = None
        view_64: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_42, [16384, 3072]);  mul_42 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0])
        addmm_17: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_53, view_64, permute_32);  primals_53 = permute_32 = None
        view_65: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_17, [128, 128, 768]);  addmm_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_46: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_15: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_46, 0.1);  inductor_random_default_46 = None
        mul_43: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_15, view_65);  view_65 = None
        mul_44: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_43, 1.1111111111111112);  mul_43 = None
        add_22: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_18, mul_44);  add_18 = mul_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_45: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_16: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_45, 0.1);  inductor_random_default_45 = None
        mul_45: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_16, add_22);  add_22 = None
        mul_46: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_45, 1.1111111111111112);  mul_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_6: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_46, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_6: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_46, [-1], correction = 1.0, keepdim = True)
        sqrt_6: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_6);  var_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_9: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_46, mean_6);  mean_6 = None
        mul_47: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_54, sub_9)
        add_23: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_6, 1e-06)
        div_12: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_47, add_23);  mul_47 = add_23 = None
        add_24: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_12, primals_55);  div_12 = primals_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_66: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_24, [16384, 768]);  add_24 = None
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0])
        addmm_18: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_57, view_66, permute_33);  primals_57 = permute_33 = None
        view_67: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_18, [128, 128, 768]);  addmm_18 = None
        view_68: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_67, [128, -1, 12, 64]);  view_67 = None
        permute_34: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0])
        addmm_19: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_59, view_66, permute_35);  primals_59 = permute_35 = None
        view_70: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_19, [128, 128, 768]);  addmm_19 = None
        view_71: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_70, [128, -1, 12, 64]);  view_70 = None
        permute_36: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0])
        addmm_20: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_61, view_66, permute_37);  primals_61 = permute_37 = None
        view_73: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_20, [128, 128, 768]);  addmm_20 = None
        view_74: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_73, [128, -1, 12, 64]);  view_73 = None
        permute_38: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_39: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        expand_12: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_34, [128, 12, 128, 64]);  permute_34 = None
        clone_12: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_75: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_12, [1536, 128, 64]);  clone_12 = None
        expand_13: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_39, [128, 12, 64, 128]);  permute_39 = None
        clone_13: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_76: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_13, [1536, 64, 128]);  clone_13 = None
        bmm_6: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_75, view_76)
        view_77: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [128, 12, 128, 128]);  bmm_6 = None
        div_13: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_3: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_13);  div_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_3: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
        sub_10: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
        exp_3: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_14: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_44: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_17: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_44, 0.1);  inductor_random_default_44 = None
        mul_48: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_17, div_14)
        mul_49: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_14: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_49, [128, 12, 128, 128]);  mul_49 = None
        view_78: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_14, [1536, 128, 128]);  expand_14 = None
        expand_15: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_38, [128, 12, 128, 64]);  permute_38 = None
        clone_14: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_79: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_14, [1536, 128, 64]);  clone_14 = None
        bmm_7: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_78, view_79)
        view_80: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_7, [128, 12, 128, 64]);  bmm_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_40: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_15: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_81: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_15, [128, -1, 768]);  clone_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_82: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_81, [16384, 768]);  view_81 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0])
        addmm_21: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_63, view_82, permute_41);  primals_63 = permute_41 = None
        view_83: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_21, [128, 128, 768]);  addmm_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_43: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_18: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_43, 0.1);  inductor_random_default_43 = None
        mul_50: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_18, view_83);  view_83 = None
        mul_51: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_50, 1.1111111111111112);  mul_50 = None
        add_25: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_46, mul_51);  mul_46 = mul_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_7: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_25, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_7: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_25, [-1], correction = 1.0, keepdim = True)
        sqrt_7: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_7);  var_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_11: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_25, mean_7);  mean_7 = None
        mul_52: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_64, sub_11)
        add_26: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_7, 1e-06)
        div_15: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_52, add_26);  mul_52 = add_26 = None
        add_27: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_15, primals_65);  div_15 = primals_65 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_84: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_27, [16384, 768]);  add_27 = None
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_66, [1, 0])
        addmm_22: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_67, view_84, permute_42);  primals_67 = permute_42 = None
        view_85: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_22, [128, 128, 3072])
        mul_53: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_54: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_28: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_55: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_28);  mul_53 = add_28 = None
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_42: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_19: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_42, 0.1);  inductor_random_default_42 = None
        mul_56: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_19, mul_55);  mul_55 = None
        mul_57: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_56, 1.1111111111111112);  mul_56 = None
        view_86: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_57, [16384, 3072]);  mul_57 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0])
        addmm_23: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_69, view_86, permute_43);  primals_69 = permute_43 = None
        view_87: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_23, [128, 128, 768]);  addmm_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_41: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_20: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_41, 0.1);  inductor_random_default_41 = None
        mul_58: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_20, view_87);  view_87 = None
        mul_59: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_58, 1.1111111111111112);  mul_58 = None
        add_29: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_25, mul_59);  add_25 = mul_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_40: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_21: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_40, 0.1);  inductor_random_default_40 = None
        mul_60: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_21, add_29);  add_29 = None
        mul_61: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_8: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_61, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_8: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_61, [-1], correction = 1.0, keepdim = True)
        sqrt_8: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_8);  var_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_12: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_61, mean_8);  mean_8 = None
        mul_62: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_70, sub_12)
        add_30: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_8, 1e-06)
        div_16: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_62, add_30);  mul_62 = add_30 = None
        add_31: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_16, primals_71);  div_16 = primals_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_88: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_31, [16384, 768]);  add_31 = None
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0])
        addmm_24: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_73, view_88, permute_44);  primals_73 = permute_44 = None
        view_89: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_24, [128, 128, 768]);  addmm_24 = None
        view_90: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_89, [128, -1, 12, 64]);  view_89 = None
        permute_45: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0])
        addmm_25: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_75, view_88, permute_46);  primals_75 = permute_46 = None
        view_92: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_25, [128, 128, 768]);  addmm_25 = None
        view_93: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_92, [128, -1, 12, 64]);  view_92 = None
        permute_47: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0])
        addmm_26: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_77, view_88, permute_48);  primals_77 = permute_48 = None
        view_95: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_26, [128, 128, 768]);  addmm_26 = None
        view_96: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_95, [128, -1, 12, 64]);  view_95 = None
        permute_49: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_50: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        expand_16: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_45, [128, 12, 128, 64]);  permute_45 = None
        clone_16: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_97: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_16, [1536, 128, 64]);  clone_16 = None
        expand_17: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_50, [128, 12, 64, 128]);  permute_50 = None
        clone_17: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_98: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_17, [1536, 64, 128]);  clone_17 = None
        bmm_8: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_97, view_98)
        view_99: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [128, 12, 128, 128]);  bmm_8 = None
        div_17: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_4: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_17);  div_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_4: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
        sub_13: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
        exp_4: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_18: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_39: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_22: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_39, 0.1);  inductor_random_default_39 = None
        mul_63: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_22, div_18)
        mul_64: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_63, 1.1111111111111112);  mul_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_18: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_64, [128, 12, 128, 128]);  mul_64 = None
        view_100: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_18, [1536, 128, 128]);  expand_18 = None
        expand_19: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_49, [128, 12, 128, 64]);  permute_49 = None
        clone_18: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_101: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_18, [1536, 128, 64]);  clone_18 = None
        bmm_9: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
        view_102: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_9, [128, 12, 128, 64]);  bmm_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_51: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_19: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_103: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_19, [128, -1, 768]);  clone_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_104: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_103, [16384, 768]);  view_103 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0])
        addmm_27: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_79, view_104, permute_52);  primals_79 = permute_52 = None
        view_105: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_27, [128, 128, 768]);  addmm_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_38: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_23: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_38, 0.1);  inductor_random_default_38 = None
        mul_65: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_23, view_105);  view_105 = None
        mul_66: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_65, 1.1111111111111112);  mul_65 = None
        add_32: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_61, mul_66);  mul_61 = mul_66 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_9: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_32, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_9: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_32, [-1], correction = 1.0, keepdim = True)
        sqrt_9: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_9);  var_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_14: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, mean_9);  mean_9 = None
        mul_67: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_80, sub_14)
        add_33: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_9, 1e-06)
        div_19: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_67, add_33);  mul_67 = add_33 = None
        add_34: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_19, primals_81);  div_19 = primals_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_106: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_34, [16384, 768]);  add_34 = None
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_82, [1, 0])
        addmm_28: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_83, view_106, permute_53);  primals_83 = permute_53 = None
        view_107: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_28, [128, 128, 3072])
        mul_68: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_69: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_35: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_70: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_35);  mul_68 = add_35 = None
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_37: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_24: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_37, 0.1);  inductor_random_default_37 = None
        mul_71: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_24, mul_70);  mul_70 = None
        mul_72: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_71, 1.1111111111111112);  mul_71 = None
        view_108: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_72, [16384, 3072]);  mul_72 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_84, [1, 0])
        addmm_29: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_85, view_108, permute_54);  primals_85 = permute_54 = None
        view_109: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_29, [128, 128, 768]);  addmm_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_36: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_25: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_73: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_25, view_109);  view_109 = None
        mul_74: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_73, 1.1111111111111112);  mul_73 = None
        add_36: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_32, mul_74);  add_32 = mul_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_35: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_26: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_35, 0.1);  inductor_random_default_35 = None
        mul_75: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_26, add_36);  add_36 = None
        mul_76: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_75, 1.1111111111111112);  mul_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_10: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_76, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_10: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_76, [-1], correction = 1.0, keepdim = True)
        sqrt_10: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_10);  var_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_15: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_76, mean_10);  mean_10 = None
        mul_77: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_86, sub_15)
        add_37: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_10, 1e-06)
        div_20: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_77, add_37);  mul_77 = add_37 = None
        add_38: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_20, primals_87);  div_20 = primals_87 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_110: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_38, [16384, 768]);  add_38 = None
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0])
        addmm_30: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_89, view_110, permute_55);  primals_89 = permute_55 = None
        view_111: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_30, [128, 128, 768]);  addmm_30 = None
        view_112: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_111, [128, -1, 12, 64]);  view_111 = None
        permute_56: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0])
        addmm_31: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_91, view_110, permute_57);  primals_91 = permute_57 = None
        view_114: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_31, [128, 128, 768]);  addmm_31 = None
        view_115: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_114, [128, -1, 12, 64]);  view_114 = None
        permute_58: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0])
        addmm_32: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_93, view_110, permute_59);  primals_93 = permute_59 = None
        view_117: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_32, [128, 128, 768]);  addmm_32 = None
        view_118: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_117, [128, -1, 12, 64]);  view_117 = None
        permute_60: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_61: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        expand_20: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_56, [128, 12, 128, 64]);  permute_56 = None
        clone_20: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_119: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_20, [1536, 128, 64]);  clone_20 = None
        expand_21: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_61, [128, 12, 64, 128]);  permute_61 = None
        clone_21: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_120: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_21, [1536, 64, 128]);  clone_21 = None
        bmm_10: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_119, view_120)
        view_121: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [128, 12, 128, 128]);  bmm_10 = None
        div_21: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_5: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_21);  div_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_5: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
        sub_16: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
        exp_5: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_22: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_34: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_27: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_34, 0.1);  inductor_random_default_34 = None
        mul_78: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_27, div_22)
        mul_79: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_22: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_79, [128, 12, 128, 128]);  mul_79 = None
        view_122: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_22, [1536, 128, 128]);  expand_22 = None
        expand_23: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_60, [128, 12, 128, 64]);  permute_60 = None
        clone_22: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_123: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_22, [1536, 128, 64]);  clone_22 = None
        bmm_11: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_122, view_123)
        view_124: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_11, [128, 12, 128, 64]);  bmm_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_62: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_23: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_125: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_23, [128, -1, 768]);  clone_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_126: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_125, [16384, 768]);  view_125 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0])
        addmm_33: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_95, view_126, permute_63);  primals_95 = permute_63 = None
        view_127: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_33, [128, 128, 768]);  addmm_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_33: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_28: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_33, 0.1);  inductor_random_default_33 = None
        mul_80: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_28, view_127);  view_127 = None
        mul_81: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_80, 1.1111111111111112);  mul_80 = None
        add_39: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_76, mul_81);  mul_76 = mul_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_11: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_39, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_11: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_39, [-1], correction = 1.0, keepdim = True)
        sqrt_11: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_11);  var_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_17: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_39, mean_11);  mean_11 = None
        mul_82: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_96, sub_17)
        add_40: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_11, 1e-06)
        div_23: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_82, add_40);  mul_82 = add_40 = None
        add_41: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_23, primals_97);  div_23 = primals_97 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_128: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_41, [16384, 768]);  add_41 = None
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_98, [1, 0])
        addmm_34: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_99, view_128, permute_64);  primals_99 = permute_64 = None
        view_129: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_34, [128, 128, 3072])
        mul_83: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_84: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_42: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_85: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_83, add_42);  mul_83 = add_42 = None
        inductor_lookup_seed_default_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_32: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        gt_29: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_86: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_29, mul_85);  mul_85 = None
        mul_87: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_86, 1.1111111111111112);  mul_86 = None
        view_130: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_87, [16384, 3072]);  mul_87 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_100, [1, 0])
        addmm_35: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_101, view_130, permute_65);  primals_101 = permute_65 = None
        view_131: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_35, [128, 128, 768]);  addmm_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_31: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        gt_30: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_31, 0.1);  inductor_random_default_31 = None
        mul_88: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_30, view_131);  view_131 = None
        mul_89: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_88, 1.1111111111111112);  mul_88 = None
        add_43: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_39, mul_89);  add_39 = mul_89 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_30: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_31: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_30, 0.1);  inductor_random_default_30 = None
        mul_90: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_31, add_43);  add_43 = None
        mul_91: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_90, 1.1111111111111112);  mul_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_12: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_91, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_12: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_91, [-1], correction = 1.0, keepdim = True)
        sqrt_12: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_12);  var_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_18: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_91, mean_12);  mean_12 = None
        mul_92: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_102, sub_18)
        add_44: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_12, 1e-06)
        div_24: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_92, add_44);  mul_92 = add_44 = None
        add_45: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_24, primals_103);  div_24 = primals_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_132: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_45, [16384, 768]);  add_45 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0])
        addmm_36: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_105, view_132, permute_66);  primals_105 = permute_66 = None
        view_133: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_36, [128, 128, 768]);  addmm_36 = None
        view_134: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_133, [128, -1, 12, 64]);  view_133 = None
        permute_67: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0])
        addmm_37: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_107, view_132, permute_68);  primals_107 = permute_68 = None
        view_136: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_37, [128, 128, 768]);  addmm_37 = None
        view_137: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_136, [128, -1, 12, 64]);  view_136 = None
        permute_69: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0])
        addmm_38: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_109, view_132, permute_70);  primals_109 = permute_70 = None
        view_139: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_38, [128, 128, 768]);  addmm_38 = None
        view_140: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_139, [128, -1, 12, 64]);  view_139 = None
        permute_71: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_72: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
        expand_24: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_67, [128, 12, 128, 64]);  permute_67 = None
        clone_24: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_141: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_24, [1536, 128, 64]);  clone_24 = None
        expand_25: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_72, [128, 12, 64, 128]);  permute_72 = None
        clone_25: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_142: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_25, [1536, 64, 128]);  clone_25 = None
        bmm_12: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_141, view_142)
        view_143: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [128, 12, 128, 128]);  bmm_12 = None
        div_25: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_6: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_25);  div_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_6: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
        sub_19: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
        exp_6: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_26: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_29: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_32: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_29, 0.1);  inductor_random_default_29 = None
        mul_93: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_32, div_26)
        mul_94: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_93, 1.1111111111111112);  mul_93 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_26: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_94, [128, 12, 128, 128]);  mul_94 = None
        view_144: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_26, [1536, 128, 128]);  expand_26 = None
        expand_27: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_71, [128, 12, 128, 64]);  permute_71 = None
        clone_26: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_145: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_26, [1536, 128, 64]);  clone_26 = None
        bmm_13: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_144, view_145)
        view_146: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_13, [128, 12, 128, 64]);  bmm_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_73: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_27: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_147: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_27, [128, -1, 768]);  clone_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_148: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_147, [16384, 768]);  view_147 = None
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0])
        addmm_39: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_111, view_148, permute_74);  primals_111 = permute_74 = None
        view_149: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_39, [128, 128, 768]);  addmm_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_28: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        gt_33: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_28, 0.1);  inductor_random_default_28 = None
        mul_95: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_33, view_149);  view_149 = None
        mul_96: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_95, 1.1111111111111112);  mul_95 = None
        add_46: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_91, mul_96);  mul_91 = mul_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_13: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_46, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_13: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_46, [-1], correction = 1.0, keepdim = True)
        sqrt_13: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_13);  var_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_20: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_46, mean_13);  mean_13 = None
        mul_97: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_112, sub_20)
        add_47: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_13, 1e-06)
        div_27: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_97, add_47);  mul_97 = add_47 = None
        add_48: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_27, primals_113);  div_27 = primals_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_150: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_48, [16384, 768]);  add_48 = None
        permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_114, [1, 0])
        addmm_40: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_115, view_150, permute_75);  primals_115 = permute_75 = None
        view_151: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_40, [128, 128, 3072])
        mul_98: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_99: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
        add_49: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_100: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_98, add_49);  mul_98 = add_49 = None
        inductor_lookup_seed_default_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_27: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_34: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_27, 0.1);  inductor_random_default_27 = None
        mul_101: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_34, mul_100);  mul_100 = None
        mul_102: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_101, 1.1111111111111112);  mul_101 = None
        view_152: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_102, [16384, 3072]);  mul_102 = None
        permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_116, [1, 0])
        addmm_41: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_117, view_152, permute_76);  primals_117 = permute_76 = None
        view_153: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_41, [128, 128, 768]);  addmm_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_26: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_35: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_103: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_35, view_153);  view_153 = None
        mul_104: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_103, 1.1111111111111112);  mul_103 = None
        add_50: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_46, mul_104);  add_46 = mul_104 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_25: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_36: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_105: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_36, add_50);  add_50 = None
        mul_106: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_14: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_106, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_14: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_106, [-1], correction = 1.0, keepdim = True)
        sqrt_14: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_14);  var_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_21: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_106, mean_14);  mean_14 = None
        mul_107: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_118, sub_21)
        add_51: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_14, 1e-06)
        div_28: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_107, add_51);  mul_107 = add_51 = None
        add_52: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_28, primals_119);  div_28 = primals_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_154: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_52, [16384, 768]);  add_52 = None
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0])
        addmm_42: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_121, view_154, permute_77);  primals_121 = permute_77 = None
        view_155: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_42, [128, 128, 768]);  addmm_42 = None
        view_156: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_155, [128, -1, 12, 64]);  view_155 = None
        permute_78: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0])
        addmm_43: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_123, view_154, permute_79);  primals_123 = permute_79 = None
        view_158: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_43, [128, 128, 768]);  addmm_43 = None
        view_159: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_158, [128, -1, 12, 64]);  view_158 = None
        permute_80: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0])
        addmm_44: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_125, view_154, permute_81);  primals_125 = permute_81 = None
        view_161: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_44, [128, 128, 768]);  addmm_44 = None
        view_162: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_161, [128, -1, 12, 64]);  view_161 = None
        permute_82: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_83: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        expand_28: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_78, [128, 12, 128, 64]);  permute_78 = None
        clone_28: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_163: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_28, [1536, 128, 64]);  clone_28 = None
        expand_29: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_83, [128, 12, 64, 128]);  permute_83 = None
        clone_29: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_164: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_29, [1536, 64, 128]);  clone_29 = None
        bmm_14: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_163, view_164)
        view_165: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [128, 12, 128, 128]);  bmm_14 = None
        div_29: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_7: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_29);  div_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_7: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
        sub_22: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
        exp_7: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_30: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_24: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        gt_37: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_108: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_37, div_30)
        mul_109: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_108, 1.1111111111111112);  mul_108 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_30: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_109, [128, 12, 128, 128]);  mul_109 = None
        view_166: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_30, [1536, 128, 128]);  expand_30 = None
        expand_31: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_82, [128, 12, 128, 64]);  permute_82 = None
        clone_30: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_167: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_30, [1536, 128, 64]);  clone_30 = None
        bmm_15: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_166, view_167)
        view_168: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_15, [128, 12, 128, 64]);  bmm_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_84: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_31: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_169: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_31, [128, -1, 768]);  clone_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_170: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_169, [16384, 768]);  view_169 = None
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0])
        addmm_45: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_127, view_170, permute_85);  primals_127 = permute_85 = None
        view_171: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_45, [128, 128, 768]);  addmm_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_23: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        gt_38: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_110: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_38, view_171);  view_171 = None
        mul_111: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_110, 1.1111111111111112);  mul_110 = None
        add_53: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_106, mul_111);  mul_106 = mul_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_15: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_53, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_15: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_53, [-1], correction = 1.0, keepdim = True)
        sqrt_15: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_15);  var_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_23: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_53, mean_15);  mean_15 = None
        mul_112: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_128, sub_23)
        add_54: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_15, 1e-06)
        div_31: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_112, add_54);  mul_112 = add_54 = None
        add_55: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_31, primals_129);  div_31 = primals_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_172: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_55, [16384, 768]);  add_55 = None
        permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_130, [1, 0])
        addmm_46: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_131, view_172, permute_86);  primals_131 = permute_86 = None
        view_173: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_46, [128, 128, 3072])
        mul_113: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_114: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
        add_56: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_115: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_113, add_56);  mul_113 = add_56 = None
        inductor_lookup_seed_default_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_22: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        gt_39: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_116: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_39, mul_115);  mul_115 = None
        mul_117: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_116, 1.1111111111111112);  mul_116 = None
        view_174: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_117, [16384, 3072]);  mul_117 = None
        permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_132, [1, 0])
        addmm_47: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_133, view_174, permute_87);  primals_133 = permute_87 = None
        view_175: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_47, [128, 128, 768]);  addmm_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_21: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        gt_40: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_118: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_40, view_175);  view_175 = None
        mul_119: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_118, 1.1111111111111112);  mul_118 = None
        add_57: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_53, mul_119);  add_53 = mul_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_20: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        gt_41: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_120: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_41, add_57);  add_57 = None
        mul_121: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_16: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_121, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_16: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_121, [-1], correction = 1.0, keepdim = True)
        sqrt_16: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_16);  var_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_24: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_121, mean_16);  mean_16 = None
        mul_122: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_134, sub_24)
        add_58: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_16, 1e-06)
        div_32: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_122, add_58);  mul_122 = add_58 = None
        add_59: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_32, primals_135);  div_32 = primals_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_176: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_59, [16384, 768]);  add_59 = None
        permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0])
        addmm_48: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_137, view_176, permute_88);  primals_137 = permute_88 = None
        view_177: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_48, [128, 128, 768]);  addmm_48 = None
        view_178: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_177, [128, -1, 12, 64]);  view_177 = None
        permute_89: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0])
        addmm_49: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_139, view_176, permute_90);  primals_139 = permute_90 = None
        view_180: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_49, [128, 128, 768]);  addmm_49 = None
        view_181: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_180, [128, -1, 12, 64]);  view_180 = None
        permute_91: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0])
        addmm_50: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_141, view_176, permute_92);  primals_141 = permute_92 = None
        view_183: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_50, [128, 128, 768]);  addmm_50 = None
        view_184: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_183, [128, -1, 12, 64]);  view_183 = None
        permute_93: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_94: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        expand_32: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_89, [128, 12, 128, 64]);  permute_89 = None
        clone_32: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_185: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_32, [1536, 128, 64]);  clone_32 = None
        expand_33: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_94, [128, 12, 64, 128]);  permute_94 = None
        clone_33: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_186: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_33, [1536, 64, 128]);  clone_33 = None
        bmm_16: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_185, view_186)
        view_187: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [128, 12, 128, 128]);  bmm_16 = None
        div_33: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_8: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_33);  div_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_8: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
        sub_25: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
        exp_8: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_34: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_19: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        gt_42: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_123: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_42, div_34)
        mul_124: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_123, 1.1111111111111112);  mul_123 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_34: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_124, [128, 12, 128, 128]);  mul_124 = None
        view_188: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_34, [1536, 128, 128]);  expand_34 = None
        expand_35: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_93, [128, 12, 128, 64]);  permute_93 = None
        clone_34: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_189: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_34, [1536, 128, 64]);  clone_34 = None
        bmm_17: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_188, view_189)
        view_190: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_17, [128, 12, 128, 64]);  bmm_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_95: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_35: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_191: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_35, [128, -1, 768]);  clone_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_192: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_191, [16384, 768]);  view_191 = None
        permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0])
        addmm_51: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_143, view_192, permute_96);  primals_143 = permute_96 = None
        view_193: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_51, [128, 128, 768]);  addmm_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_18: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        gt_43: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_125: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_43, view_193);  view_193 = None
        mul_126: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_125, 1.1111111111111112);  mul_125 = None
        add_60: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_121, mul_126);  mul_121 = mul_126 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_17: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_60, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_17: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_60, [-1], correction = 1.0, keepdim = True)
        sqrt_17: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_17);  var_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_26: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_60, mean_17);  mean_17 = None
        mul_127: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_144, sub_26)
        add_61: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_17, 1e-06)
        div_35: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_127, add_61);  mul_127 = add_61 = None
        add_62: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_35, primals_145);  div_35 = primals_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_194: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_62, [16384, 768]);  add_62 = None
        permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_146, [1, 0])
        addmm_52: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_147, view_194, permute_97);  primals_147 = permute_97 = None
        view_195: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_52, [128, 128, 3072])
        mul_128: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_129: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_63: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_130: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_128, add_63);  mul_128 = add_63 = None
        inductor_lookup_seed_default_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_17: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        gt_44: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_131: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_44, mul_130);  mul_130 = None
        mul_132: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_131, 1.1111111111111112);  mul_131 = None
        view_196: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_132, [16384, 3072]);  mul_132 = None
        permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0])
        addmm_53: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_149, view_196, permute_98);  primals_149 = permute_98 = None
        view_197: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_53, [128, 128, 768]);  addmm_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_16: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        gt_45: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_133: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_45, view_197);  view_197 = None
        mul_134: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_133, 1.1111111111111112);  mul_133 = None
        add_64: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_60, mul_134);  add_60 = mul_134 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_15: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        gt_46: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_135: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_46, add_64);  add_64 = None
        mul_136: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_18: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_136, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_18: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_136, [-1], correction = 1.0, keepdim = True)
        sqrt_18: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_18);  var_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_27: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_136, mean_18);  mean_18 = None
        mul_137: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_150, sub_27)
        add_65: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_18, 1e-06)
        div_36: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_137, add_65);  mul_137 = add_65 = None
        add_66: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_36, primals_151);  div_36 = primals_151 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_198: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_66, [16384, 768]);  add_66 = None
        permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0])
        addmm_54: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_153, view_198, permute_99);  primals_153 = permute_99 = None
        view_199: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_54, [128, 128, 768]);  addmm_54 = None
        view_200: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_199, [128, -1, 12, 64]);  view_199 = None
        permute_100: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0])
        addmm_55: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_155, view_198, permute_101);  primals_155 = permute_101 = None
        view_202: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_55, [128, 128, 768]);  addmm_55 = None
        view_203: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_202, [128, -1, 12, 64]);  view_202 = None
        permute_102: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
        permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0])
        addmm_56: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_157, view_198, permute_103);  primals_157 = permute_103 = None
        view_205: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_56, [128, 128, 768]);  addmm_56 = None
        view_206: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_205, [128, -1, 12, 64]);  view_205 = None
        permute_104: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_105: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
        expand_36: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_100, [128, 12, 128, 64]);  permute_100 = None
        clone_36: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_207: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_36, [1536, 128, 64]);  clone_36 = None
        expand_37: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_105, [128, 12, 64, 128]);  permute_105 = None
        clone_37: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_208: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_37, [1536, 64, 128]);  clone_37 = None
        bmm_18: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_207, view_208)
        view_209: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [128, 12, 128, 128]);  bmm_18 = None
        div_37: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_9: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_37);  div_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_9: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
        sub_28: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
        exp_9: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_38: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_46: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_14: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        gt_47: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_138: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_47, div_38)
        mul_139: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_138, 1.1111111111111112);  mul_138 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_38: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_139, [128, 12, 128, 128]);  mul_139 = None
        view_210: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_38, [1536, 128, 128]);  expand_38 = None
        expand_39: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_104, [128, 12, 128, 64]);  permute_104 = None
        clone_38: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_211: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_38, [1536, 128, 64]);  clone_38 = None
        bmm_19: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_210, view_211)
        view_212: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_19, [128, 12, 128, 64]);  bmm_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_106: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_39: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_213: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_39, [128, -1, 768]);  clone_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_214: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_213, [16384, 768]);  view_213 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_158, [1, 0])
        addmm_57: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_159, view_214, permute_107);  primals_159 = permute_107 = None
        view_215: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_57, [128, 128, 768]);  addmm_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_47: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_13: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        gt_48: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_140: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_48, view_215);  view_215 = None
        mul_141: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_140, 1.1111111111111112);  mul_140 = None
        add_67: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_136, mul_141);  mul_136 = mul_141 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_19: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_67, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_19: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_67, [-1], correction = 1.0, keepdim = True)
        sqrt_19: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_19);  var_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_29: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_67, mean_19);  mean_19 = None
        mul_142: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_160, sub_29)
        add_68: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_19, 1e-06)
        div_39: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_142, add_68);  mul_142 = add_68 = None
        add_69: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_39, primals_161);  div_39 = primals_161 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_216: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_69, [16384, 768]);  add_69 = None
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_162, [1, 0])
        addmm_58: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_163, view_216, permute_108);  primals_163 = permute_108 = None
        view_217: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_58, [128, 128, 3072])
        mul_143: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_144: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_144);  mul_144 = None
        add_70: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_145: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_143, add_70);  mul_143 = add_70 = None
        inductor_lookup_seed_default_48: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default_12: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        gt_49: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_146: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_49, mul_145);  mul_145 = None
        mul_147: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_146, 1.1111111111111112);  mul_146 = None
        view_218: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_147, [16384, 3072]);  mul_147 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_164, [1, 0])
        addmm_59: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_165, view_218, permute_109);  primals_165 = permute_109 = None
        view_219: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_59, [128, 128, 768]);  addmm_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_49: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49)
        inductor_random_default_11: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_49, 'rand');  inductor_lookup_seed_default_49 = None
        gt_50: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_148: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_50, view_219);  view_219 = None
        mul_149: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_148, 1.1111111111111112);  mul_148 = None
        add_71: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_67, mul_149);  add_67 = mul_149 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_50: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50)
        inductor_random_default_10: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_50, 'rand');  inductor_lookup_seed_default_50 = None
        gt_51: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_150: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_51, add_71);  add_71 = None
        mul_151: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_150, 1.1111111111111112);  mul_150 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_20: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_151, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_20: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_151, [-1], correction = 1.0, keepdim = True)
        sqrt_20: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_20);  var_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_30: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_151, mean_20);  mean_20 = None
        mul_152: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_166, sub_30)
        add_72: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_20, 1e-06)
        div_40: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_152, add_72);  mul_152 = add_72 = None
        add_73: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_40, primals_167);  div_40 = primals_167 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_220: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_73, [16384, 768]);  add_73 = None
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0])
        addmm_60: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_169, view_220, permute_110);  primals_169 = permute_110 = None
        view_221: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_60, [128, 128, 768]);  addmm_60 = None
        view_222: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_221, [128, -1, 12, 64]);  view_221 = None
        permute_111: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0])
        addmm_61: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_171, view_220, permute_112);  primals_171 = permute_112 = None
        view_224: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_61, [128, 128, 768]);  addmm_61 = None
        view_225: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_224, [128, -1, 12, 64]);  view_224 = None
        permute_113: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0])
        addmm_62: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_173, view_220, permute_114);  primals_173 = permute_114 = None
        view_227: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_62, [128, 128, 768]);  addmm_62 = None
        view_228: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_227, [128, -1, 12, 64]);  view_227 = None
        permute_115: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_116: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
        expand_40: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_111, [128, 12, 128, 64]);  permute_111 = None
        clone_40: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_229: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_40, [1536, 128, 64]);  clone_40 = None
        expand_41: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_116, [128, 12, 64, 128]);  permute_116 = None
        clone_41: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_230: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_41, [1536, 64, 128]);  clone_41 = None
        bmm_20: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_229, view_230)
        view_231: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [128, 12, 128, 128]);  bmm_20 = None
        div_41: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_10: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_41);  div_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_10: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
        sub_31: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
        exp_10: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_42: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_51: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51)
        inductor_random_default_9: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_51, 'rand');  inductor_lookup_seed_default_51 = None
        gt_52: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_153: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_52, div_42)
        mul_154: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_153, 1.1111111111111112);  mul_153 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_42: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_154, [128, 12, 128, 128]);  mul_154 = None
        view_232: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_42, [1536, 128, 128]);  expand_42 = None
        expand_43: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_115, [128, 12, 128, 64]);  permute_115 = None
        clone_42: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_233: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_42, [1536, 128, 64]);  clone_42 = None
        bmm_21: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_232, view_233)
        view_234: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_21, [128, 12, 128, 64]);  bmm_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_117: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_43: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_235: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_43, [128, -1, 768]);  clone_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_236: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_235, [16384, 768]);  view_235 = None
        permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0])
        addmm_63: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_175, view_236, permute_118);  primals_175 = permute_118 = None
        view_237: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_63, [128, 128, 768]);  addmm_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_52: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 52)
        inductor_random_default_8: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_52, 'rand');  inductor_lookup_seed_default_52 = None
        gt_53: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_155: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_53, view_237);  view_237 = None
        mul_156: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_155, 1.1111111111111112);  mul_155 = None
        add_74: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_151, mul_156);  mul_151 = mul_156 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_21: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_74, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_21: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_74, [-1], correction = 1.0, keepdim = True)
        sqrt_21: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_21);  var_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_32: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_74, mean_21);  mean_21 = None
        mul_157: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_176, sub_32)
        add_75: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_21, 1e-06)
        div_43: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_157, add_75);  mul_157 = add_75 = None
        add_76: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_43, primals_177);  div_43 = primals_177 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_238: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_76, [16384, 768]);  add_76 = None
        permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_178, [1, 0])
        addmm_64: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_179, view_238, permute_119);  primals_179 = permute_119 = None
        view_239: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_64, [128, 128, 3072])
        mul_158: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_159: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
        add_77: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_160: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_158, add_77);  mul_158 = add_77 = None
        inductor_lookup_seed_default_53: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 53)
        inductor_random_default_7: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_53, 'rand');  inductor_lookup_seed_default_53 = None
        gt_54: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_161: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_54, mul_160);  mul_160 = None
        mul_162: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_161, 1.1111111111111112);  mul_161 = None
        view_240: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_162, [16384, 3072]);  mul_162 = None
        permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_180, [1, 0])
        addmm_65: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_181, view_240, permute_120);  primals_181 = permute_120 = None
        view_241: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_65, [128, 128, 768]);  addmm_65 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_54: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 54)
        inductor_random_default_6: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_54, 'rand');  inductor_lookup_seed_default_54 = None
        gt_55: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_163: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_55, view_241);  view_241 = None
        mul_164: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_163, 1.1111111111111112);  mul_163 = None
        add_78: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_74, mul_164);  add_74 = mul_164 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_55: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 55)
        inductor_random_default_5: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_55, 'rand');  inductor_lookup_seed_default_55 = None
        gt_56: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_165: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_56, add_78);  add_78 = None
        mul_166: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_165, 1.1111111111111112);  mul_165 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_22: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_166, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_22: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_166, [-1], correction = 1.0, keepdim = True)
        sqrt_22: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_22);  var_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_33: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_166, mean_22);  mean_22 = None
        mul_167: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_182, sub_33)
        add_79: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_22, 1e-06)
        div_44: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_167, add_79);  mul_167 = add_79 = None
        add_80: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_44, primals_183);  div_44 = primals_183 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        view_242: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_80, [16384, 768]);  add_80 = None
        permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0])
        addmm_66: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_185, view_242, permute_121);  primals_185 = permute_121 = None
        view_243: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_66, [128, 128, 768]);  addmm_66 = None
        view_244: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_243, [128, -1, 12, 64]);  view_243 = None
        permute_122: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0])
        addmm_67: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_187, view_242, permute_123);  primals_187 = permute_123 = None
        view_246: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_67, [128, 128, 768]);  addmm_67 = None
        view_247: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_246, [128, -1, 12, 64]);  view_246 = None
        permute_124: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0])
        addmm_68: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_189, view_242, permute_125);  primals_189 = permute_125 = None
        view_249: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_68, [128, 128, 768]);  addmm_68 = None
        view_250: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(view_249, [128, -1, 12, 64]);  view_249 = None
        permute_126: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_127: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
        expand_44: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_122, [128, 12, 128, 64]);  permute_122 = None
        clone_44: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_251: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_44, [1536, 128, 64]);  clone_44 = None
        expand_45: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_127, [128, 12, 64, 128]);  permute_127 = None
        clone_45: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_252: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_45, [1536, 64, 128]);  clone_45 = None
        bmm_22: "f32[1536, 128, 128]" = torch.ops.aten.bmm.default(view_251, view_252)
        view_253: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [128, 12, 128, 128]);  bmm_22 = None
        div_45: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_11: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq, full_default, div_45);  eq = full_default = div_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        amax_11: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
        sub_34: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
        exp_11: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_46: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_56: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56)
        inductor_random_default_4: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default_56, 'rand');  inductor_lookup_seed_default_56 = None
        gt_57: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_168: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_57, div_46)
        mul_169: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_168, 1.1111111111111112);  mul_168 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        expand_46: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(mul_169, [128, 12, 128, 128]);  mul_169 = None
        view_254: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(expand_46, [1536, 128, 128]);  expand_46 = None
        expand_47: "f32[128, 12, 128, 64]" = torch.ops.aten.expand.default(permute_126, [128, 12, 128, 64]);  permute_126 = None
        clone_46: "f32[128, 12, 128, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_255: "f32[1536, 128, 64]" = torch.ops.aten.reshape.default(clone_46, [1536, 128, 64]);  clone_46 = None
        bmm_23: "f32[1536, 128, 64]" = torch.ops.aten.bmm.default(view_254, view_255)
        view_256: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_23, [128, 12, 128, 64]);  bmm_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_128: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_47: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_257: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_47, [128, -1, 768]);  clone_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_258: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_257, [16384, 768]);  view_257 = None
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0])
        addmm_69: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_191, view_258, permute_129);  primals_191 = permute_129 = None
        view_259: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_69, [128, 128, 768]);  addmm_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_57: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57)
        inductor_random_default_3: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_57, 'rand');  inductor_lookup_seed_default_57 = None
        gt_58: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_170: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_58, view_259);  view_259 = None
        mul_171: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_170, 1.1111111111111112);  mul_170 = None
        add_81: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_166, mul_171);  mul_166 = mul_171 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_23: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_81, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_23: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_81, [-1], correction = 1.0, keepdim = True)
        sqrt_23: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_23);  var_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_35: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_81, mean_23);  mean_23 = None
        mul_172: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_192, sub_35)
        add_82: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_23, 1e-06)
        div_47: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_172, add_82);  mul_172 = add_82 = None
        add_83: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_47, primals_193);  div_47 = primals_193 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_260: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_83, [16384, 768]);  add_83 = None
        permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_194, [1, 0])
        addmm_70: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_195, view_260, permute_130);  primals_195 = permute_130 = None
        view_261: "f32[128, 128, 3072]" = torch.ops.aten.reshape.default(addmm_70, [128, 128, 3072])
        mul_173: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_174: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_84: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_175: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_173, add_84);  mul_173 = add_84 = None
        inductor_lookup_seed_default_58: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58)
        inductor_random_default_2: "f32[128, 128, 3072]" = torch.ops.prims.inductor_random.default([128, 128, 3072], inductor_lookup_seed_default_58, 'rand');  inductor_lookup_seed_default_58 = None
        gt_59: "b8[128, 128, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_176: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(gt_59, mul_175);  mul_175 = None
        mul_177: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_176, 1.1111111111111112);  mul_176 = None
        view_262: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_177, [16384, 3072]);  mul_177 = None
        permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_196, [1, 0])
        addmm_71: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_197, view_262, permute_131);  primals_197 = permute_131 = None
        view_263: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_71, [128, 128, 768]);  addmm_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_59: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 59)
        inductor_random_default_1: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_59, 'rand');  inductor_lookup_seed_default_59 = None
        gt_60: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_178: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_60, view_263);  view_263 = None
        mul_179: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_178, 1.1111111111111112);  mul_178 = None
        add_85: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_81, mul_179);  add_81 = mul_179 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_60: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_60, 'rand');  inductor_lookup_seed_default_60 = None
        gt_61: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_180: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_61, add_85);  add_85 = None
        mul_181: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_180, 1.1111111111111112);  mul_180 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:42 in forward, code: return self.softmax(self.linear(x[:, 0]))
        select: "f32[128, 768]" = torch.ops.aten.select.int(mul_181, 1, 0)
        permute_132: "f32[768, 2]" = torch.ops.aten.permute.default(primals_198, [1, 0])
        addmm_72: "f32[128, 2]" = torch.ops.aten.addmm.default(primals_199, select, permute_132);  primals_199 = permute_132 = None
        amax_12: "f32[128, 1]" = torch.ops.aten.amax.default(addmm_72, [-1], True)
        sub_36: "f32[128, 2]" = torch.ops.aten.sub.Tensor(addmm_72, amax_12);  addmm_72 = amax_12 = None
        exp_12: "f32[128, 2]" = torch.ops.aten.exp.default(sub_36)
        sum_13: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True);  exp_12 = None
        log: "f32[128, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_37: "f32[128, 2]" = torch.ops.aten.sub.Tensor(sub_36, log);  sub_36 = log = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:61 in forward, code: return self.softmax(self.linear(x))
        view_264: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_181, [16384, 768]);  mul_181 = None
        permute_133: "f32[768, 20005]" = torch.ops.aten.permute.default(primals_200, [1, 0])
        addmm_73: "f32[16384, 20005]" = torch.ops.aten.addmm.default(primals_201, view_264, permute_133);  primals_201 = permute_133 = None
        view_265: "f32[128, 128, 20005]" = torch.ops.aten.reshape.default(addmm_73, [128, 128, 20005]);  addmm_73 = None
        amax_13: "f32[128, 128, 1]" = torch.ops.aten.amax.default(view_265, [-1], True)
        sub_38: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(view_265, amax_13);  view_265 = amax_13 = None
        exp_13: "f32[128, 128, 20005]" = torch.ops.aten.exp.default(sub_38)
        sum_14: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True);  exp_13 = None
        log_1: "f32[128, 128, 1]" = torch.ops.aten.log.default(sum_14);  sum_14 = None
        sub_39: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(sub_38, log_1);  sub_38 = log_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_155: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_156: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_157: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_158: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_188: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        permute_189: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_190: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_191: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_221: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_222: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_223: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_224: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_254: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_255: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_256: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_257: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_287: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_288: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_289: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_290: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_320: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_321: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_322: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_323: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_353: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_354: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_355: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_356: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_386: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_387: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_388: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_389: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_419: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_420: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_421: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_422: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_452: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_453: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_454: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_455: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_485: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_486: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_487: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_488: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_518: "f32[1536, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_519: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_520: "f32[1536, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_521: "f32[1536, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        return (sub_37, sub_39, unsqueeze_1, primals_1, primals_5, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, primals_156, primals_158, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, unsqueeze_1, gt_1, sqrt, sub, view, bmm, amax, sum_1, gt_2, view_16, gt_3, sqrt_1, sub_2, view_18, addmm_4, gt_4, view_20, gt_5, gt_6, sqrt_2, sub_3, view_22, div_6, gt_7, view_38, gt_8, sqrt_3, sub_5, view_40, addmm_10, gt_9, view_42, gt_10, gt_11, sqrt_4, sub_6, view_44, div_10, gt_12, view_60, gt_13, sqrt_5, sub_8, view_62, addmm_16, gt_14, view_64, gt_15, gt_16, sqrt_6, sub_9, view_66, div_14, gt_17, view_82, gt_18, sqrt_7, sub_11, view_84, addmm_22, gt_19, view_86, gt_20, gt_21, sqrt_8, sub_12, view_88, div_18, gt_22, view_104, gt_23, sqrt_9, sub_14, view_106, addmm_28, gt_24, view_108, gt_25, gt_26, sqrt_10, sub_15, view_110, div_22, gt_27, view_126, gt_28, sqrt_11, sub_17, view_128, addmm_34, gt_29, view_130, gt_30, gt_31, sqrt_12, sub_18, view_132, div_26, gt_32, view_148, gt_33, sqrt_13, sub_20, view_150, addmm_40, gt_34, view_152, gt_35, gt_36, sqrt_14, sub_21, view_154, div_30, gt_37, view_170, gt_38, sqrt_15, sub_23, view_172, addmm_46, gt_39, view_174, gt_40, gt_41, sqrt_16, sub_24, view_176, div_34, gt_42, view_192, gt_43, sqrt_17, sub_26, view_194, addmm_52, gt_44, view_196, gt_45, gt_46, sqrt_18, sub_27, view_198, div_38, gt_47, view_214, gt_48, sqrt_19, sub_29, view_216, addmm_58, gt_49, view_218, gt_50, gt_51, sqrt_20, sub_30, view_220, div_42, gt_52, view_236, gt_53, sqrt_21, sub_32, view_238, addmm_64, gt_54, view_240, gt_55, gt_56, sqrt_22, sub_33, view_242, div_46, gt_57, view_258, gt_58, sqrt_23, sub_35, view_260, addmm_70, gt_59, view_262, gt_60, gt_61, select, sub_37, view_264, sub_39, permute_155, permute_156, permute_157, permute_158, permute_188, permute_189, permute_190, permute_191, permute_221, permute_222, permute_223, permute_224, permute_254, permute_255, permute_256, permute_257, permute_287, permute_288, permute_289, permute_290, permute_320, permute_321, permute_322, permute_323, permute_353, permute_354, permute_355, permute_356, permute_386, permute_387, permute_388, permute_389, permute_419, permute_420, permute_421, permute_422, permute_452, permute_453, permute_454, permute_455, permute_485, permute_486, permute_487, permute_488, permute_518, permute_519, permute_520, permute_521)

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[16, 128][128, 1]cuda:0", primals_2: "f32[20005, 768][768, 1]cuda:0", primals_3: "f32[1, 512, 768][393216, 768, 1]cuda:0", primals_4: "f32[3, 768][768, 1]cuda:0", primals_5: "i64[16, 128][128, 1]cuda:0", primals_6: "f32[768][1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_8: "f32[768, 768][768, 1]cuda:0", primals_9: "f32[768][1]cuda:0", primals_10: "f32[768, 768][768, 1]cuda:0", primals_11: "f32[768][1]cuda:0", primals_12: "f32[768, 768][768, 1]cuda:0", primals_13: "f32[768][1]cuda:0", primals_14: "f32[768, 768][768, 1]cuda:0", primals_15: "f32[768][1]cuda:0", primals_16: "f32[768][1]cuda:0", primals_17: "f32[768][1]cuda:0", primals_18: "f32[3072, 768][768, 1]cuda:0", primals_19: "f32[3072][1]cuda:0", primals_20: "f32[768, 3072][3072, 1]cuda:0", primals_21: "f32[768][1]cuda:0", primals_22: "f32[768][1]cuda:0", primals_23: "f32[768][1]cuda:0", primals_24: "f32[768, 768][768, 1]cuda:0", primals_25: "f32[768][1]cuda:0", primals_26: "f32[768, 768][768, 1]cuda:0", primals_27: "f32[768][1]cuda:0", primals_28: "f32[768, 768][768, 1]cuda:0", primals_29: "f32[768][1]cuda:0", primals_30: "f32[768, 768][768, 1]cuda:0", primals_31: "f32[768][1]cuda:0", primals_32: "f32[768][1]cuda:0", primals_33: "f32[768][1]cuda:0", primals_34: "f32[3072, 768][768, 1]cuda:0", primals_35: "f32[3072][1]cuda:0", primals_36: "f32[768, 3072][3072, 1]cuda:0", primals_37: "f32[768][1]cuda:0", primals_38: "f32[768][1]cuda:0", primals_39: "f32[768][1]cuda:0", primals_40: "f32[768, 768][768, 1]cuda:0", primals_41: "f32[768][1]cuda:0", primals_42: "f32[768, 768][768, 1]cuda:0", primals_43: "f32[768][1]cuda:0", primals_44: "f32[768, 768][768, 1]cuda:0", primals_45: "f32[768][1]cuda:0", primals_46: "f32[768, 768][768, 1]cuda:0", primals_47: "f32[768][1]cuda:0", primals_48: "f32[768][1]cuda:0", primals_49: "f32[768][1]cuda:0", primals_50: "f32[3072, 768][768, 1]cuda:0", primals_51: "f32[3072][1]cuda:0", primals_52: "f32[768, 3072][3072, 1]cuda:0", primals_53: "f32[768][1]cuda:0", primals_54: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_56: "f32[768, 768][768, 1]cuda:0", primals_57: "f32[768][1]cuda:0", primals_58: "f32[768, 768][768, 1]cuda:0", primals_59: "f32[768][1]cuda:0", primals_60: "f32[768, 768][768, 1]cuda:0", primals_61: "f32[768][1]cuda:0", primals_62: "f32[768, 768][768, 1]cuda:0", primals_63: "f32[768][1]cuda:0", primals_64: "f32[768][1]cuda:0", primals_65: "f32[768][1]cuda:0", primals_66: "f32[3072, 768][768, 1]cuda:0", primals_67: "f32[3072][1]cuda:0", primals_68: "f32[768, 3072][3072, 1]cuda:0", primals_69: "f32[768][1]cuda:0", primals_70: "f32[768][1]cuda:0", primals_71: "f32[768][1]cuda:0", primals_72: "f32[768, 768][768, 1]cuda:0", primals_73: "f32[768][1]cuda:0", primals_74: "f32[768, 768][768, 1]cuda:0", primals_75: "f32[768][1]cuda:0", primals_76: "f32[768, 768][768, 1]cuda:0", primals_77: "f32[768][1]cuda:0", primals_78: "f32[768, 768][768, 1]cuda:0", primals_79: "f32[768][1]cuda:0", primals_80: "f32[768][1]cuda:0", primals_81: "f32[768][1]cuda:0", primals_82: "f32[3072, 768][768, 1]cuda:0", primals_83: "f32[3072][1]cuda:0", primals_84: "f32[768, 3072][3072, 1]cuda:0", primals_85: "f32[768][1]cuda:0", primals_86: "f32[768][1]cuda:0", primals_87: "f32[768][1]cuda:0", primals_88: "f32[768, 768][768, 1]cuda:0", primals_89: "f32[768][1]cuda:0", primals_90: "f32[768, 768][768, 1]cuda:0", primals_91: "f32[768][1]cuda:0", primals_92: "f32[768, 768][768, 1]cuda:0", primals_93: "f32[768][1]cuda:0", primals_94: "f32[768, 768][768, 1]cuda:0", primals_95: "f32[768][1]cuda:0", primals_96: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_98: "f32[3072, 768][768, 1]cuda:0", primals_99: "f32[3072][1]cuda:0", primals_100: "f32[768, 3072][3072, 1]cuda:0", primals_101: "f32[768][1]cuda:0", primals_102: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_104: "f32[768, 768][768, 1]cuda:0", primals_105: "f32[768][1]cuda:0", primals_106: "f32[768, 768][768, 1]cuda:0", primals_107: "f32[768][1]cuda:0", primals_108: "f32[768, 768][768, 1]cuda:0", primals_109: "f32[768][1]cuda:0", primals_110: "f32[768, 768][768, 1]cuda:0", primals_111: "f32[768][1]cuda:0", primals_112: "f32[768][1]cuda:0", primals_113: "f32[768][1]cuda:0", primals_114: "f32[3072, 768][768, 1]cuda:0", primals_115: "f32[3072][1]cuda:0", primals_116: "f32[768, 3072][3072, 1]cuda:0", primals_117: "f32[768][1]cuda:0", primals_118: "f32[768][1]cuda:0", primals_119: "f32[768][1]cuda:0", primals_120: "f32[768, 768][768, 1]cuda:0", primals_121: "f32[768][1]cuda:0", primals_122: "f32[768, 768][768, 1]cuda:0", primals_123: "f32[768][1]cuda:0", primals_124: "f32[768, 768][768, 1]cuda:0", primals_125: "f32[768][1]cuda:0", primals_126: "f32[768, 768][768, 1]cuda:0", primals_127: "f32[768][1]cuda:0", primals_128: "f32[768][1]cuda:0", primals_129: "f32[768][1]cuda:0", primals_130: "f32[3072, 768][768, 1]cuda:0", primals_131: "f32[3072][1]cuda:0", primals_132: "f32[768, 3072][3072, 1]cuda:0", primals_133: "f32[768][1]cuda:0", primals_134: "f32[768][1]cuda:0", primals_135: "f32[768][1]cuda:0", primals_136: "f32[768, 768][768, 1]cuda:0", primals_137: "f32[768][1]cuda:0", primals_138: "f32[768, 768][768, 1]cuda:0", primals_139: "f32[768][1]cuda:0", primals_140: "f32[768, 768][768, 1]cuda:0", primals_141: "f32[768][1]cuda:0", primals_142: "f32[768, 768][768, 1]cuda:0", primals_143: "f32[768][1]cuda:0", primals_144: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_146: "f32[3072, 768][768, 1]cuda:0", primals_147: "f32[3072][1]cuda:0", primals_148: "f32[768, 3072][3072, 1]cuda:0", primals_149: "f32[768][1]cuda:0", primals_150: "f32[768][1]cuda:0", primals_151: "f32[768][1]cuda:0", primals_152: "f32[768, 768][768, 1]cuda:0", primals_153: "f32[768][1]cuda:0", primals_154: "f32[768, 768][768, 1]cuda:0", primals_155: "f32[768][1]cuda:0", primals_156: "f32[768, 768][768, 1]cuda:0", primals_157: "f32[768][1]cuda:0", primals_158: "f32[768, 768][768, 1]cuda:0", primals_159: "f32[768][1]cuda:0", primals_160: "f32[768][1]cuda:0", primals_161: "f32[768][1]cuda:0", primals_162: "f32[3072, 768][768, 1]cuda:0", primals_163: "f32[3072][1]cuda:0", primals_164: "f32[768, 3072][3072, 1]cuda:0", primals_165: "f32[768][1]cuda:0", primals_166: "f32[768][1]cuda:0", primals_167: "f32[768][1]cuda:0", primals_168: "f32[768, 768][768, 1]cuda:0", primals_169: "f32[768][1]cuda:0", primals_170: "f32[768, 768][768, 1]cuda:0", primals_171: "f32[768][1]cuda:0", primals_172: "f32[768, 768][768, 1]cuda:0", primals_173: "f32[768][1]cuda:0", primals_174: "f32[768, 768][768, 1]cuda:0", primals_175: "f32[768][1]cuda:0", primals_176: "f32[768][1]cuda:0", primals_177: "f32[768][1]cuda:0", primals_178: "f32[3072, 768][768, 1]cuda:0", primals_179: "f32[3072][1]cuda:0", primals_180: "f32[768, 3072][3072, 1]cuda:0", primals_181: "f32[768][1]cuda:0", primals_182: "f32[768][1]cuda:0", primals_183: "f32[768][1]cuda:0", primals_184: "f32[768, 768][768, 1]cuda:0", primals_185: "f32[768][1]cuda:0", primals_186: "f32[768, 768][768, 1]cuda:0", primals_187: "f32[768][1]cuda:0", primals_188: "f32[768, 768][768, 1]cuda:0", primals_189: "f32[768][1]cuda:0", primals_190: "f32[768, 768][768, 1]cuda:0", primals_191: "f32[768][1]cuda:0", primals_192: "f32[768][1]cuda:0", primals_193: "f32[768][1]cuda:0", primals_194: "f32[3072, 768][768, 1]cuda:0", primals_195: "f32[3072][1]cuda:0", primals_196: "f32[768, 3072][3072, 1]cuda:0", primals_197: "f32[768][1]cuda:0", primals_198: "f32[2, 768][768, 1]cuda:0", primals_199: "f32[2][1]cuda:0", primals_200: "f32[20005, 768][768, 1]cuda:0", primals_201: "f32[20005][1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/bert.py:43 in forward, code: mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        gt: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.gt.Scalar(primals_1, 0)
        unsqueeze: "b8[16, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(gt, 1);  gt = None
        repeat: "b8[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.repeat.default(unsqueeze, [1, 128, 1]);  unsqueeze = None
        unsqueeze_1: "b8[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(repeat, 1);  repeat = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        embedding: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_2, primals_1, 0);  primals_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/position.py:28 in forward, code: return self.pe[:, : x.size(1)]
        slice_1: "f32[1, 128, 768][393216, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128);  primals_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        add: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, slice_1);  embedding = slice_1 = None
        embedding_1: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_4, primals_5, 0);  primals_4 = None
        add_1: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add, embedding_1);  add = embedding_1 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[61][1]cuda:0" = torch.ops.prims.inductor_seeds.default(61, device(type='cuda', index=0))

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:33 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_60: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_1: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_60, 0.1);  inductor_random_default_60 = None
        mul: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, add_1);  add_1 = None
        mul_1: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_1, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_1, [-1], correction = 1.0, keepdim = True)
        sqrt: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var);  var = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_1, mean);  mean = None
        mul_2: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_6, sub)
        add_2: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt, 1e-06)
        div: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        add_3: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div, primals_7);  div = primals_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_1: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convert_element_type_2: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [2048, 768]);  convert_element_type_2 = None
        permute: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [16, 128, 768]);  addmm = None
        view_2: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [16, -1, 12, 64]);  view_1 = None
        permute_1: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        convert_element_type_6: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_7: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        permute_2: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, view, permute_2);  convert_element_type_6 = None
        view_4: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [16, 128, 768]);  addmm_1 = None
        view_5: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [16, -1, 12, 64]);  view_4 = None
        permute_3: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        convert_element_type_12: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_13: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        permute_4: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_2: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_12, view, permute_4);  convert_element_type_12 = None
        view_7: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [16, 128, 768]);  addmm_2 = None
        view_8: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_7, [16, -1, 12, 64]);  view_7 = None
        permute_5: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_6: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_1, [16, 12, 128, 64]);  permute_1 = None
        clone: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_9: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [192, 128, 64]);  clone = None
        expand_1: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_6, [16, 12, 64, 128]);  permute_6 = None
        clone_1: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [192, 64, 128]);  clone_1 = None
        bmm: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_9, view_10)
        view_11: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [16, 12, 128, 128])
        div_1: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        eq: "b8[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_1);  div_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_20: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_20, [-1], True)
        sub_1: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, amax);  convert_element_type_20 = None
        exp: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_59: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_2: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_59, 0.1);  inductor_random_default_59 = None
        mul_3: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, div_2);  div_2 = None
        mul_4: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_21: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        expand_2: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_21, [16, 12, 128, 128]);  convert_element_type_21 = None
        view_12: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_2, [192, 128, 128]);  expand_2 = None
        expand_3: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [16, 12, 128, 64]);  permute_5 = None
        clone_2: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_13: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [192, 128, 64]);  clone_2 = None
        bmm_1: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_12, view_13)
        view_14: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [16, 12, 128, 64]);  bmm_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_7: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [16, -1, 768]);  clone_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_24: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_25: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        view_16: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [2048, 768]);  view_15 = None
        permute_8: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None
        addmm_3: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_24, view_16, permute_8);  convert_element_type_24 = None
        view_17: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [16, 128, 768]);  addmm_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_58: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_35: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_58, torch.bfloat16);  inductor_random_default_58 = None
        gt_3: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_35, 0.1);  convert_element_type_default_35 = None
        mul_5: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_3, view_17);  view_17 = None
        mul_6: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        add_4: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_6);  mul_1 = mul_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_1: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_4, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_1: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_4, [-1], correction = 1.0, keepdim = True)
        sqrt_1: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_1);  var_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_2: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_4, mean_1);  mean_1 = None
        mul_7: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, sub_2)
        add_5: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_1, 1e-06)
        div_3: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_7, add_5);  mul_7 = add_5 = None
        add_6: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_3, primals_17);  div_3 = primals_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_29: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_30: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convert_element_type_31: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_18: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_31, [2048, 768]);  convert_element_type_31 = None
        permute_9: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        addmm_4: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_29, view_18, permute_9);  convert_element_type_29 = None
        view_19: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [16, 128, 3072])
        convert_element_type_35: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_8: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.5)
        mul_9: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.7071067811865476);  convert_element_type_35 = None
        erf: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_7: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_10: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, add_7);  mul_8 = add_7 = None
        convert_element_type_36: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_57: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_34: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_57, torch.bfloat16);  inductor_random_default_57 = None
        gt_4: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_34, 0.1);  convert_element_type_default_34 = None
        mul_11: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_4, convert_element_type_36);  convert_element_type_36 = None
        mul_12: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, 1.1111111111111112);  mul_11 = None
        convert_element_type_37: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_38: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        view_20: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_12, [2048, 3072]);  mul_12 = None
        permute_10: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_38, [1, 0]);  convert_element_type_38 = None
        addmm_5: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_37, view_20, permute_10);  convert_element_type_37 = None
        view_21: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [16, 128, 768]);  addmm_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_4: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_56: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_33: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_56, torch.bfloat16);  inductor_random_default_56 = None
        gt_5: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_33, 0.1);  convert_element_type_default_33 = None
        mul_13: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_5, view_21);  view_21 = None
        mul_14: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, 1.1111111111111112);  mul_13 = None
        add_8: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4, mul_14);  add_4 = mul_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_5: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_55: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_6: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_55, 0.1);  inductor_random_default_55 = None
        mul_15: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_6, add_8);  add_8 = None
        mul_16: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, 1.1111111111111112);  mul_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_2: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_16, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_2: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_16, [-1], correction = 1.0, keepdim = True)
        sqrt_2: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_2);  var_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_3: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_16, mean_2);  mean_2 = None
        mul_17: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, sub_3)
        add_9: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_2, 1e-06)
        div_4: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_17, add_9);  mul_17 = add_9 = None
        add_10: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_4, primals_23);  div_4 = primals_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_42: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_43: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convert_element_type_44: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None
        view_22: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_44, [2048, 768]);  convert_element_type_44 = None
        permute_11: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_43, [1, 0]);  convert_element_type_43 = None
        addmm_6: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_42, view_22, permute_11);  convert_element_type_42 = None
        view_23: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [16, 128, 768]);  addmm_6 = None
        view_24: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [16, -1, 12, 64]);  view_23 = None
        permute_12: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        convert_element_type_48: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        convert_element_type_49: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        permute_13: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_49, [1, 0]);  convert_element_type_49 = None
        addmm_7: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_48, view_22, permute_13);  convert_element_type_48 = None
        view_26: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [16, 128, 768]);  addmm_7 = None
        view_27: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [16, -1, 12, 64]);  view_26 = None
        permute_14: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        convert_element_type_54: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_55: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        permute_15: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_55, [1, 0]);  convert_element_type_55 = None
        addmm_8: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_54, view_22, permute_15);  convert_element_type_54 = None
        view_29: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [16, 128, 768]);  addmm_8 = None
        view_30: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [16, -1, 12, 64]);  view_29 = None
        permute_16: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_17: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        expand_4: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_12, [16, 12, 128, 64]);  permute_12 = None
        clone_4: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_31: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [192, 128, 64]);  clone_4 = None
        expand_5: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_17, [16, 12, 64, 128]);  permute_17 = None
        clone_5: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_32: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [192, 64, 128]);  clone_5 = None
        bmm_2: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_31, view_32)
        view_33: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [16, 12, 128, 128]);  bmm_2 = None
        div_5: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_1: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_5);  div_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_62: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32)
        amax_1: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_62, [-1], True)
        sub_4: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, amax_1);  convert_element_type_62 = None
        exp_1: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_6: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_6: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_54: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_7: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_54, 0.1);  inductor_random_default_54 = None
        mul_18: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_7, div_6);  div_6 = None
        mul_19: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, 1.1111111111111112);  mul_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_63: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        expand_6: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_63, [16, 12, 128, 128]);  convert_element_type_63 = None
        view_34: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_6, [192, 128, 128]);  expand_6 = None
        expand_7: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_16, [16, 12, 128, 64]);  permute_16 = None
        clone_6: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_35: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [192, 128, 64]);  clone_6 = None
        bmm_3: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_34, view_35)
        view_36: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [16, 12, 128, 64]);  bmm_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_18: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [16, -1, 768]);  clone_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_66: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convert_element_type_67: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        view_38: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [2048, 768]);  view_37 = None
        permute_19: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_67, [1, 0]);  convert_element_type_67 = None
        addmm_9: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_66, view_38, permute_19);  convert_element_type_66 = None
        view_39: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [16, 128, 768]);  addmm_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_7: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_53: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_32: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_53, torch.bfloat16);  inductor_random_default_53 = None
        gt_8: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_32, 0.1);  convert_element_type_default_32 = None
        mul_20: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_8, view_39);  view_39 = None
        mul_21: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        add_11: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, mul_21);  mul_16 = mul_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_3: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_11, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_3: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_11, [-1], correction = 1.0, keepdim = True)
        sqrt_3: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_3);  var_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_5: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_11, mean_3);  mean_3 = None
        mul_22: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_32, sub_5)
        add_12: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_3, 1e-06)
        div_7: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_22, add_12);  mul_22 = add_12 = None
        add_13: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_7, primals_33);  div_7 = primals_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_71: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_72: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        convert_element_type_73: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        view_40: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_73, [2048, 768]);  convert_element_type_73 = None
        permute_20: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_72, [1, 0]);  convert_element_type_72 = None
        addmm_10: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_71, view_40, permute_20);  convert_element_type_71 = None
        view_41: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [16, 128, 3072])
        convert_element_type_77: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_23: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, 0.5)
        mul_24: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, 0.7071067811865476);  convert_element_type_77 = None
        erf_1: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_14: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_25: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, add_14);  mul_23 = add_14 = None
        convert_element_type_78: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        inductor_lookup_seed_default_8: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_52: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_31: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_52, torch.bfloat16);  inductor_random_default_52 = None
        gt_9: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_31, 0.1);  convert_element_type_default_31 = None
        mul_26: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_9, convert_element_type_78);  convert_element_type_78 = None
        mul_27: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, 1.1111111111111112);  mul_26 = None
        convert_element_type_79: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convert_element_type_80: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        view_42: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [2048, 3072]);  mul_27 = None
        permute_21: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        addmm_11: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_79, view_42, permute_21);  convert_element_type_79 = None
        view_43: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [16, 128, 768]);  addmm_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_9: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_51: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_30: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_51, torch.bfloat16);  inductor_random_default_51 = None
        gt_10: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_30, 0.1);  convert_element_type_default_30 = None
        mul_28: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_10, view_43);  view_43 = None
        mul_29: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, 1.1111111111111112);  mul_28 = None
        add_15: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_11, mul_29);  add_11 = mul_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_10: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_50: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_11: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_50, 0.1);  inductor_random_default_50 = None
        mul_30: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_11, add_15);  add_15 = None
        mul_31: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_4: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_31, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_4: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_31, [-1], correction = 1.0, keepdim = True)
        sqrt_4: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_4);  var_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_6: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_31, mean_4);  mean_4 = None
        mul_32: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_38, sub_6)
        add_16: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_4, 1e-06)
        div_8: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_32, add_16);  mul_32 = add_16 = None
        add_17: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_8, primals_39);  div_8 = primals_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_84: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_85: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        convert_element_type_86: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None
        view_44: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_86, [2048, 768]);  convert_element_type_86 = None
        permute_22: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        addmm_12: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_84, view_44, permute_22);  convert_element_type_84 = None
        view_45: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [16, 128, 768]);  addmm_12 = None
        view_46: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [16, -1, 12, 64]);  view_45 = None
        permute_23: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        convert_element_type_90: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        convert_element_type_91: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        permute_24: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_91, [1, 0]);  convert_element_type_91 = None
        addmm_13: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_90, view_44, permute_24);  convert_element_type_90 = None
        view_48: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [16, 128, 768]);  addmm_13 = None
        view_49: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_48, [16, -1, 12, 64]);  view_48 = None
        permute_25: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        convert_element_type_96: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_97: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        permute_26: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_97, [1, 0]);  convert_element_type_97 = None
        addmm_14: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_96, view_44, permute_26);  convert_element_type_96 = None
        view_51: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [16, 128, 768]);  addmm_14 = None
        view_52: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_51, [16, -1, 12, 64]);  view_51 = None
        permute_27: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_28: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        expand_8: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_23, [16, 12, 128, 64]);  permute_23 = None
        clone_8: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_53: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [192, 128, 64]);  clone_8 = None
        expand_9: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_28, [16, 12, 64, 128]);  permute_28 = None
        clone_9: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_54: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [192, 64, 128]);  clone_9 = None
        bmm_4: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_53, view_54)
        view_55: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [16, 12, 128, 128]);  bmm_4 = None
        div_9: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_2: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_9);  div_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_104: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32)
        amax_2: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_104, [-1], True)
        sub_7: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_104, amax_2);  convert_element_type_104 = None
        exp_2: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_10: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_11: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_49: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_12: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_49, 0.1);  inductor_random_default_49 = None
        mul_33: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_12, div_10);  div_10 = None
        mul_34: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, 1.1111111111111112);  mul_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_105: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None
        expand_10: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_105, [16, 12, 128, 128]);  convert_element_type_105 = None
        view_56: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_10, [192, 128, 128]);  expand_10 = None
        expand_11: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_27, [16, 12, 128, 64]);  permute_27 = None
        clone_10: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_57: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [192, 128, 64]);  clone_10 = None
        bmm_5: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_56, view_57)
        view_58: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [16, 12, 128, 64]);  bmm_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_29: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_11: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [16, -1, 768]);  clone_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_108: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        convert_element_type_109: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        view_60: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [2048, 768]);  view_59 = None
        permute_30: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        addmm_15: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_108, view_60, permute_30);  convert_element_type_108 = None
        view_61: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [16, 128, 768]);  addmm_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_12: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_48: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        convert_element_type_default_29: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_48, torch.bfloat16);  inductor_random_default_48 = None
        gt_13: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_29, 0.1);  convert_element_type_default_29 = None
        mul_35: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_13, view_61);  view_61 = None
        mul_36: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, 1.1111111111111112);  mul_35 = None
        add_18: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, mul_36);  mul_31 = mul_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_5: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_18, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_5: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_18, [-1], correction = 1.0, keepdim = True)
        sqrt_5: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_5);  var_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_8: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_18, mean_5);  mean_5 = None
        mul_37: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_48, sub_8)
        add_19: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_5, 1e-06)
        div_11: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_37, add_19);  mul_37 = add_19 = None
        add_20: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_11, primals_49);  div_11 = primals_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_113: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        convert_element_type_114: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convert_element_type_115: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None
        view_62: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_115, [2048, 768]);  convert_element_type_115 = None
        permute_31: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        addmm_16: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_113, view_62, permute_31);  convert_element_type_113 = None
        view_63: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [16, 128, 3072])
        convert_element_type_119: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_38: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, 0.5)
        mul_39: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, 0.7071067811865476);  convert_element_type_119 = None
        erf_2: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_21: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_40: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, add_21);  mul_38 = add_21 = None
        convert_element_type_120: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None
        inductor_lookup_seed_default_13: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_47: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        convert_element_type_default_28: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_47, torch.bfloat16);  inductor_random_default_47 = None
        gt_14: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_28, 0.1);  convert_element_type_default_28 = None
        mul_41: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_14, convert_element_type_120);  convert_element_type_120 = None
        mul_42: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, 1.1111111111111112);  mul_41 = None
        convert_element_type_121: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        convert_element_type_122: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        view_64: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_42, [2048, 3072]);  mul_42 = None
        permute_32: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_122, [1, 0]);  convert_element_type_122 = None
        addmm_17: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_121, view_64, permute_32);  convert_element_type_121 = None
        view_65: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [16, 128, 768]);  addmm_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_14: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_46: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        convert_element_type_default_27: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_46, torch.bfloat16);  inductor_random_default_46 = None
        gt_15: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_27, 0.1);  convert_element_type_default_27 = None
        mul_43: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_15, view_65);  view_65 = None
        mul_44: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, 1.1111111111111112);  mul_43 = None
        add_22: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_18, mul_44);  add_18 = mul_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_15: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_45: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_16: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_45, 0.1);  inductor_random_default_45 = None
        mul_45: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_16, add_22);  add_22 = None
        mul_46: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, 1.1111111111111112);  mul_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_6: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_46, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_6: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_46, [-1], correction = 1.0, keepdim = True)
        sqrt_6: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_6);  var_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_9: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_46, mean_6);  mean_6 = None
        mul_47: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_54, sub_9)
        add_23: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_6, 1e-06)
        div_12: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_47, add_23);  mul_47 = add_23 = None
        add_24: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_12, primals_55);  div_12 = primals_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_126: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_127: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convert_element_type_128: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None
        view_66: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_128, [2048, 768]);  convert_element_type_128 = None
        permute_33: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_127, [1, 0]);  convert_element_type_127 = None
        addmm_18: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_126, view_66, permute_33);  convert_element_type_126 = None
        view_67: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [16, 128, 768]);  addmm_18 = None
        view_68: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [16, -1, 12, 64]);  view_67 = None
        permute_34: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        convert_element_type_132: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        convert_element_type_133: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        permute_35: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_133, [1, 0]);  convert_element_type_133 = None
        addmm_19: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_132, view_66, permute_35);  convert_element_type_132 = None
        view_70: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [16, 128, 768]);  addmm_19 = None
        view_71: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [16, -1, 12, 64]);  view_70 = None
        permute_36: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        convert_element_type_138: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        convert_element_type_139: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        permute_37: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        addmm_20: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_138, view_66, permute_37);  convert_element_type_138 = None
        view_73: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [16, 128, 768]);  addmm_20 = None
        view_74: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [16, -1, 12, 64]);  view_73 = None
        permute_38: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_39: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        expand_12: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_34, [16, 12, 128, 64]);  permute_34 = None
        clone_12: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_75: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [192, 128, 64]);  clone_12 = None
        expand_13: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_39, [16, 12, 64, 128]);  permute_39 = None
        clone_13: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_76: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [192, 64, 128]);  clone_13 = None
        bmm_6: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_75, view_76)
        view_77: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [16, 12, 128, 128]);  bmm_6 = None
        div_13: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_3: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_13);  div_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_146: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32)
        amax_3: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_146, [-1], True)
        sub_10: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_146, amax_3);  convert_element_type_146 = None
        exp_3: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_14: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_16: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_44: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_17: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_44, 0.1);  inductor_random_default_44 = None
        mul_48: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_17, div_14);  div_14 = None
        mul_49: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_147: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        expand_14: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_147, [16, 12, 128, 128]);  convert_element_type_147 = None
        view_78: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_14, [192, 128, 128]);  expand_14 = None
        expand_15: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_38, [16, 12, 128, 64]);  permute_38 = None
        clone_14: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_79: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [192, 128, 64]);  clone_14 = None
        bmm_7: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_78, view_79)
        view_80: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [16, 12, 128, 64]);  bmm_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_40: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_15: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_81: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [16, -1, 768]);  clone_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_150: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        convert_element_type_151: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        view_82: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [2048, 768]);  view_81 = None
        permute_41: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        addmm_21: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_150, view_82, permute_41);  convert_element_type_150 = None
        view_83: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [16, 128, 768]);  addmm_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_17: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_43: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        convert_element_type_default_26: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_43, torch.bfloat16);  inductor_random_default_43 = None
        gt_18: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_26, 0.1);  convert_element_type_default_26 = None
        mul_50: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_18, view_83);  view_83 = None
        mul_51: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, 1.1111111111111112);  mul_50 = None
        add_25: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, mul_51);  mul_46 = mul_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_7: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_25, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_7: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_25, [-1], correction = 1.0, keepdim = True)
        sqrt_7: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_7);  var_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_11: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_25, mean_7);  mean_7 = None
        mul_52: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, sub_11)
        add_26: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_7, 1e-06)
        div_15: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_52, add_26);  mul_52 = add_26 = None
        add_27: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_15, primals_65);  div_15 = primals_65 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_155: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_156: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        convert_element_type_157: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None
        view_84: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_157, [2048, 768]);  convert_element_type_157 = None
        permute_42: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_156, [1, 0]);  convert_element_type_156 = None
        addmm_22: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_155, view_84, permute_42);  convert_element_type_155 = None
        view_85: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [16, 128, 3072])
        convert_element_type_161: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_53: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, 0.5)
        mul_54: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, 0.7071067811865476);  convert_element_type_161 = None
        erf_3: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_28: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_55: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, add_28);  mul_53 = add_28 = None
        convert_element_type_162: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        inductor_lookup_seed_default_18: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_42: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        convert_element_type_default_25: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_42, torch.bfloat16);  inductor_random_default_42 = None
        gt_19: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_25, 0.1);  convert_element_type_default_25 = None
        mul_56: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_19, convert_element_type_162);  convert_element_type_162 = None
        mul_57: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, 1.1111111111111112);  mul_56 = None
        convert_element_type_163: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        convert_element_type_164: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        view_86: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_57, [2048, 3072]);  mul_57 = None
        permute_43: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        addmm_23: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_163, view_86, permute_43);  convert_element_type_163 = None
        view_87: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [16, 128, 768]);  addmm_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_19: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_41: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        convert_element_type_default_24: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_41, torch.bfloat16);  inductor_random_default_41 = None
        gt_20: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_24, 0.1);  convert_element_type_default_24 = None
        mul_58: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_20, view_87);  view_87 = None
        mul_59: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, 1.1111111111111112);  mul_58 = None
        add_29: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_25, mul_59);  add_25 = mul_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_20: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_40: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_21: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_40, 0.1);  inductor_random_default_40 = None
        mul_60: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_21, add_29);  add_29 = None
        mul_61: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_8: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_61, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_8: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_61, [-1], correction = 1.0, keepdim = True)
        sqrt_8: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_8);  var_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_12: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_61, mean_8);  mean_8 = None
        mul_62: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, sub_12)
        add_30: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_8, 1e-06)
        div_16: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_62, add_30);  mul_62 = add_30 = None
        add_31: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_16, primals_71);  div_16 = primals_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_168: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_169: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        convert_element_type_170: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        view_88: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_170, [2048, 768]);  convert_element_type_170 = None
        permute_44: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_169, [1, 0]);  convert_element_type_169 = None
        addmm_24: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_168, view_88, permute_44);  convert_element_type_168 = None
        view_89: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [16, 128, 768]);  addmm_24 = None
        view_90: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [16, -1, 12, 64]);  view_89 = None
        permute_45: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        convert_element_type_174: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        convert_element_type_175: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        permute_46: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_175, [1, 0]);  convert_element_type_175 = None
        addmm_25: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_174, view_88, permute_46);  convert_element_type_174 = None
        view_92: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [16, 128, 768]);  addmm_25 = None
        view_93: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_92, [16, -1, 12, 64]);  view_92 = None
        permute_47: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        convert_element_type_180: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        convert_element_type_181: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        permute_48: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_181, [1, 0]);  convert_element_type_181 = None
        addmm_26: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_180, view_88, permute_48);  convert_element_type_180 = None
        view_95: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [16, 128, 768]);  addmm_26 = None
        view_96: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_95, [16, -1, 12, 64]);  view_95 = None
        permute_49: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_50: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        expand_16: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_45, [16, 12, 128, 64]);  permute_45 = None
        clone_16: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_97: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [192, 128, 64]);  clone_16 = None
        expand_17: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_50, [16, 12, 64, 128]);  permute_50 = None
        clone_17: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_98: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [192, 64, 128]);  clone_17 = None
        bmm_8: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_97, view_98)
        view_99: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [16, 12, 128, 128]);  bmm_8 = None
        div_17: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_4: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_17);  div_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_188: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32)
        amax_4: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_188, [-1], True)
        sub_13: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_188, amax_4);  convert_element_type_188 = None
        exp_4: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_18: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_21: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_39: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_22: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_39, 0.1);  inductor_random_default_39 = None
        mul_63: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_22, div_18);  div_18 = None
        mul_64: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, 1.1111111111111112);  mul_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_189: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        expand_18: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_189, [16, 12, 128, 128]);  convert_element_type_189 = None
        view_100: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_18, [192, 128, 128]);  expand_18 = None
        expand_19: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_49, [16, 12, 128, 64]);  permute_49 = None
        clone_18: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_101: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [192, 128, 64]);  clone_18 = None
        bmm_9: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_100, view_101)
        view_102: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [16, 12, 128, 64]);  bmm_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_51: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_19: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_103: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [16, -1, 768]);  clone_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_192: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_193: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        view_104: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [2048, 768]);  view_103 = None
        permute_52: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_193, [1, 0]);  convert_element_type_193 = None
        addmm_27: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_192, view_104, permute_52);  convert_element_type_192 = None
        view_105: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [16, 128, 768]);  addmm_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_22: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_38: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        convert_element_type_default_23: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_38, torch.bfloat16);  inductor_random_default_38 = None
        gt_23: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_23, 0.1);  convert_element_type_default_23 = None
        mul_65: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_23, view_105);  view_105 = None
        mul_66: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, 1.1111111111111112);  mul_65 = None
        add_32: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, mul_66);  mul_61 = mul_66 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_9: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_32, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_9: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_32, [-1], correction = 1.0, keepdim = True)
        sqrt_9: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_9);  var_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_14: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_32, mean_9);  mean_9 = None
        mul_67: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_80, sub_14)
        add_33: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_9, 1e-06)
        div_19: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_67, add_33);  mul_67 = add_33 = None
        add_34: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_19, primals_81);  div_19 = primals_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_197: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        convert_element_type_198: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        convert_element_type_199: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        view_106: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_199, [2048, 768]);  convert_element_type_199 = None
        permute_53: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_198, [1, 0]);  convert_element_type_198 = None
        addmm_28: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_197, view_106, permute_53);  convert_element_type_197 = None
        view_107: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [16, 128, 3072])
        convert_element_type_203: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_68: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, 0.5)
        mul_69: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, 0.7071067811865476);  convert_element_type_203 = None
        erf_4: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_35: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_70: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_35);  mul_68 = add_35 = None
        convert_element_type_204: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        inductor_lookup_seed_default_23: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_37: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        convert_element_type_default_22: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_37, torch.bfloat16);  inductor_random_default_37 = None
        gt_24: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_22, 0.1);  convert_element_type_default_22 = None
        mul_71: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_24, convert_element_type_204);  convert_element_type_204 = None
        mul_72: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, 1.1111111111111112);  mul_71 = None
        convert_element_type_205: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        convert_element_type_206: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        view_108: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_72, [2048, 3072]);  mul_72 = None
        permute_54: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        addmm_29: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_205, view_108, permute_54);  convert_element_type_205 = None
        view_109: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [16, 128, 768]);  addmm_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_24: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_36: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        convert_element_type_default_21: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_36, torch.bfloat16);  inductor_random_default_36 = None
        gt_25: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_21, 0.1);  convert_element_type_default_21 = None
        mul_73: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_25, view_109);  view_109 = None
        mul_74: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, 1.1111111111111112);  mul_73 = None
        add_36: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_32, mul_74);  add_32 = mul_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_25: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_35: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_26: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_35, 0.1);  inductor_random_default_35 = None
        mul_75: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_26, add_36);  add_36 = None
        mul_76: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, 1.1111111111111112);  mul_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_10: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_76, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_10: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_76, [-1], correction = 1.0, keepdim = True)
        sqrt_10: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_10);  var_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_15: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_76, mean_10);  mean_10 = None
        mul_77: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_86, sub_15)
        add_37: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_10, 1e-06)
        div_20: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_77, add_37);  mul_77 = add_37 = None
        add_38: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_20, primals_87);  div_20 = primals_87 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_210: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        convert_element_type_211: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        convert_element_type_212: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None
        view_110: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_212, [2048, 768]);  convert_element_type_212 = None
        permute_55: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_211, [1, 0]);  convert_element_type_211 = None
        addmm_30: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_210, view_110, permute_55);  convert_element_type_210 = None
        view_111: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [16, 128, 768]);  addmm_30 = None
        view_112: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [16, -1, 12, 64]);  view_111 = None
        permute_56: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        convert_element_type_216: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        convert_element_type_217: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        permute_57: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_217, [1, 0]);  convert_element_type_217 = None
        addmm_31: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_216, view_110, permute_57);  convert_element_type_216 = None
        view_114: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [16, 128, 768]);  addmm_31 = None
        view_115: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [16, -1, 12, 64]);  view_114 = None
        permute_58: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        convert_element_type_222: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convert_element_type_223: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        permute_59: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        addmm_32: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_222, view_110, permute_59);  convert_element_type_222 = None
        view_117: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [16, 128, 768]);  addmm_32 = None
        view_118: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_117, [16, -1, 12, 64]);  view_117 = None
        permute_60: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_61: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        expand_20: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_56, [16, 12, 128, 64]);  permute_56 = None
        clone_20: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_119: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [192, 128, 64]);  clone_20 = None
        expand_21: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_61, [16, 12, 64, 128]);  permute_61 = None
        clone_21: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_120: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [192, 64, 128]);  clone_21 = None
        bmm_10: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_119, view_120)
        view_121: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [16, 12, 128, 128]);  bmm_10 = None
        div_21: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_5: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_21);  div_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_230: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32)
        amax_5: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_230, [-1], True)
        sub_16: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, amax_5);  convert_element_type_230 = None
        exp_5: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_22: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_26: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_34: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_27: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_34, 0.1);  inductor_random_default_34 = None
        mul_78: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_27, div_22);  div_22 = None
        mul_79: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_231: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        expand_22: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_231, [16, 12, 128, 128]);  convert_element_type_231 = None
        view_122: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_22, [192, 128, 128]);  expand_22 = None
        expand_23: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_60, [16, 12, 128, 64]);  permute_60 = None
        clone_22: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_123: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [192, 128, 64]);  clone_22 = None
        bmm_11: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_122, view_123)
        view_124: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [16, 12, 128, 64]);  bmm_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_62: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_23: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_125: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [16, -1, 768]);  clone_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_234: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        convert_element_type_235: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        view_126: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [2048, 768]);  view_125 = None
        permute_63: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_235, [1, 0]);  convert_element_type_235 = None
        addmm_33: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_234, view_126, permute_63);  convert_element_type_234 = None
        view_127: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [16, 128, 768]);  addmm_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_27: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_33: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        convert_element_type_default_20: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_33, torch.bfloat16);  inductor_random_default_33 = None
        gt_28: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_20, 0.1);  convert_element_type_default_20 = None
        mul_80: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_28, view_127);  view_127 = None
        mul_81: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 1.1111111111111112);  mul_80 = None
        add_39: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, mul_81);  mul_76 = mul_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_11: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_39, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_11: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_39, [-1], correction = 1.0, keepdim = True)
        sqrt_11: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_11);  var_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_17: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_39, mean_11);  mean_11 = None
        mul_82: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_96, sub_17)
        add_40: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_11, 1e-06)
        div_23: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_82, add_40);  mul_82 = add_40 = None
        add_41: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_23, primals_97);  div_23 = primals_97 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_239: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convert_element_type_240: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convert_element_type_241: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None
        view_128: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [2048, 768]);  convert_element_type_241 = None
        permute_64: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        addmm_34: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_239, view_128, permute_64);  convert_element_type_239 = None
        view_129: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [16, 128, 3072])
        convert_element_type_245: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_83: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, 0.5)
        mul_84: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, 0.7071067811865476);  convert_element_type_245 = None
        erf_5: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_42: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_85: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, add_42);  mul_83 = add_42 = None
        convert_element_type_246: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        inductor_lookup_seed_default_28: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_32: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        convert_element_type_default_19: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_32, torch.bfloat16);  inductor_random_default_32 = None
        gt_29: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_19, 0.1);  convert_element_type_default_19 = None
        mul_86: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_29, convert_element_type_246);  convert_element_type_246 = None
        mul_87: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, 1.1111111111111112);  mul_86 = None
        convert_element_type_247: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        convert_element_type_248: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        view_130: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_87, [2048, 3072]);  mul_87 = None
        permute_65: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_248, [1, 0]);  convert_element_type_248 = None
        addmm_35: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_247, view_130, permute_65);  convert_element_type_247 = None
        view_131: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [16, 128, 768]);  addmm_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_29: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_31: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        convert_element_type_default_18: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_31, torch.bfloat16);  inductor_random_default_31 = None
        gt_30: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_18, 0.1);  convert_element_type_default_18 = None
        mul_88: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_30, view_131);  view_131 = None
        mul_89: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, 1.1111111111111112);  mul_88 = None
        add_43: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_39, mul_89);  add_39 = mul_89 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_30: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_30: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_31: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_30, 0.1);  inductor_random_default_30 = None
        mul_90: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_31, add_43);  add_43 = None
        mul_91: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, 1.1111111111111112);  mul_90 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_12: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_91, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_12: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_91, [-1], correction = 1.0, keepdim = True)
        sqrt_12: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_12);  var_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_18: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_91, mean_12);  mean_12 = None
        mul_92: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_102, sub_18)
        add_44: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_12, 1e-06)
        div_24: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_92, add_44);  mul_92 = add_44 = None
        add_45: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_24, primals_103);  div_24 = primals_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_252: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_253: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convert_element_type_254: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None
        view_132: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_254, [2048, 768]);  convert_element_type_254 = None
        permute_66: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_253, [1, 0]);  convert_element_type_253 = None
        addmm_36: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_252, view_132, permute_66);  convert_element_type_252 = None
        view_133: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [16, 128, 768]);  addmm_36 = None
        view_134: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [16, -1, 12, 64]);  view_133 = None
        permute_67: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        convert_element_type_258: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16);  primals_107 = None
        convert_element_type_259: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        permute_68: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        addmm_37: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_258, view_132, permute_68);  convert_element_type_258 = None
        view_136: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [16, 128, 768]);  addmm_37 = None
        view_137: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_136, [16, -1, 12, 64]);  view_136 = None
        permute_69: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        convert_element_type_264: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        convert_element_type_265: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        permute_70: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_265, [1, 0]);  convert_element_type_265 = None
        addmm_38: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_264, view_132, permute_70);  convert_element_type_264 = None
        view_139: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [16, 128, 768]);  addmm_38 = None
        view_140: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_139, [16, -1, 12, 64]);  view_139 = None
        permute_71: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_72: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
        expand_24: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_67, [16, 12, 128, 64]);  permute_67 = None
        clone_24: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_141: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [192, 128, 64]);  clone_24 = None
        expand_25: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_72, [16, 12, 64, 128]);  permute_72 = None
        clone_25: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_142: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [192, 64, 128]);  clone_25 = None
        bmm_12: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_141, view_142)
        view_143: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [16, 12, 128, 128]);  bmm_12 = None
        div_25: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_6: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_25);  div_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_272: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32)
        amax_6: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_272, [-1], True)
        sub_19: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, amax_6);  convert_element_type_272 = None
        exp_6: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_26: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_31: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_29: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_32: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_29, 0.1);  inductor_random_default_29 = None
        mul_93: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_32, div_26);  div_26 = None
        mul_94: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, 1.1111111111111112);  mul_93 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_273: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        expand_26: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_273, [16, 12, 128, 128]);  convert_element_type_273 = None
        view_144: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_26, [192, 128, 128]);  expand_26 = None
        expand_27: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_71, [16, 12, 128, 64]);  permute_71 = None
        clone_26: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_145: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [192, 128, 64]);  clone_26 = None
        bmm_13: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_144, view_145)
        view_146: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [16, 12, 128, 64]);  bmm_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_73: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_27: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_147: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [16, -1, 768]);  clone_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_276: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        convert_element_type_277: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        view_148: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [2048, 768]);  view_147 = None
        permute_74: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_277, [1, 0]);  convert_element_type_277 = None
        addmm_39: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_276, view_148, permute_74);  convert_element_type_276 = None
        view_149: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [16, 128, 768]);  addmm_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_32: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_28: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        convert_element_type_default_17: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_28, torch.bfloat16);  inductor_random_default_28 = None
        gt_33: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_17, 0.1);  convert_element_type_default_17 = None
        mul_95: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_33, view_149);  view_149 = None
        mul_96: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, 1.1111111111111112);  mul_95 = None
        add_46: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, mul_96);  mul_91 = mul_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_13: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_46, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_13: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_46, [-1], correction = 1.0, keepdim = True)
        sqrt_13: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_13);  var_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_20: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_46, mean_13);  mean_13 = None
        mul_97: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, sub_20)
        add_47: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_13, 1e-06)
        div_27: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_97, add_47);  mul_97 = add_47 = None
        add_48: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_27, primals_113);  div_27 = primals_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_281: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convert_element_type_282: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        convert_element_type_283: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None
        view_150: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_283, [2048, 768]);  convert_element_type_283 = None
        permute_75: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_282, [1, 0]);  convert_element_type_282 = None
        addmm_40: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_281, view_150, permute_75);  convert_element_type_281 = None
        view_151: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [16, 128, 3072])
        convert_element_type_287: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_98: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, 0.5)
        mul_99: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, 0.7071067811865476);  convert_element_type_287 = None
        erf_6: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
        add_49: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_100: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, add_49);  mul_98 = add_49 = None
        convert_element_type_288: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_100, torch.bfloat16);  mul_100 = None
        inductor_lookup_seed_default_33: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_27: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        convert_element_type_default_16: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_27, torch.bfloat16);  inductor_random_default_27 = None
        gt_34: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_16, 0.1);  convert_element_type_default_16 = None
        mul_101: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_34, convert_element_type_288);  convert_element_type_288 = None
        mul_102: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, 1.1111111111111112);  mul_101 = None
        convert_element_type_289: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        convert_element_type_290: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        view_152: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_102, [2048, 3072]);  mul_102 = None
        permute_76: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_290, [1, 0]);  convert_element_type_290 = None
        addmm_41: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_289, view_152, permute_76);  convert_element_type_289 = None
        view_153: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [16, 128, 768]);  addmm_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_34: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_26: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        convert_element_type_default_15: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_26, torch.bfloat16);  inductor_random_default_26 = None
        gt_35: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_15, 0.1);  convert_element_type_default_15 = None
        mul_103: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_35, view_153);  view_153 = None
        mul_104: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, 1.1111111111111112);  mul_103 = None
        add_50: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_46, mul_104);  add_46 = mul_104 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_35: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_25: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_36: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_105: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_36, add_50);  add_50 = None
        mul_106: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_14: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_106, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_14: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_106, [-1], correction = 1.0, keepdim = True)
        sqrt_14: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_14);  var_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_21: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_106, mean_14);  mean_14 = None
        mul_107: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_118, sub_21)
        add_51: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_14, 1e-06)
        div_28: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_107, add_51);  mul_107 = add_51 = None
        add_52: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_28, primals_119);  div_28 = primals_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_294: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_295: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        convert_element_type_296: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None
        view_154: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_296, [2048, 768]);  convert_element_type_296 = None
        permute_77: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_295, [1, 0]);  convert_element_type_295 = None
        addmm_42: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_294, view_154, permute_77);  convert_element_type_294 = None
        view_155: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [16, 128, 768]);  addmm_42 = None
        view_156: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [16, -1, 12, 64]);  view_155 = None
        permute_78: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        convert_element_type_300: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        convert_element_type_301: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        permute_79: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_301, [1, 0]);  convert_element_type_301 = None
        addmm_43: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_300, view_154, permute_79);  convert_element_type_300 = None
        view_158: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [16, 128, 768]);  addmm_43 = None
        view_159: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_158, [16, -1, 12, 64]);  view_158 = None
        permute_80: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        convert_element_type_306: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        convert_element_type_307: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        permute_81: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_307, [1, 0]);  convert_element_type_307 = None
        addmm_44: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_306, view_154, permute_81);  convert_element_type_306 = None
        view_161: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [16, 128, 768]);  addmm_44 = None
        view_162: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [16, -1, 12, 64]);  view_161 = None
        permute_82: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_83: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        expand_28: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_78, [16, 12, 128, 64]);  permute_78 = None
        clone_28: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_163: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [192, 128, 64]);  clone_28 = None
        expand_29: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_83, [16, 12, 64, 128]);  permute_83 = None
        clone_29: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_164: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [192, 64, 128]);  clone_29 = None
        bmm_14: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_163, view_164)
        view_165: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [16, 12, 128, 128]);  bmm_14 = None
        div_29: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_7: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_29);  div_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_314: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32)
        amax_7: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_314, [-1], True)
        sub_22: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_314, amax_7);  convert_element_type_314 = None
        exp_7: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_30: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_36: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_24: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        gt_37: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_108: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_37, div_30);  div_30 = None
        mul_109: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 1.1111111111111112);  mul_108 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_315: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        expand_30: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_315, [16, 12, 128, 128]);  convert_element_type_315 = None
        view_166: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_30, [192, 128, 128]);  expand_30 = None
        expand_31: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_82, [16, 12, 128, 64]);  permute_82 = None
        clone_30: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_167: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [192, 128, 64]);  clone_30 = None
        bmm_15: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_166, view_167)
        view_168: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [16, 12, 128, 64]);  bmm_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_84: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_31: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_169: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [16, -1, 768]);  clone_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_318: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convert_element_type_319: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        view_170: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [2048, 768]);  view_169 = None
        permute_85: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_319, [1, 0]);  convert_element_type_319 = None
        addmm_45: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_318, view_170, permute_85);  convert_element_type_318 = None
        view_171: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [16, 128, 768]);  addmm_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_37: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_23: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        convert_element_type_default_14: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_23, torch.bfloat16);  inductor_random_default_23 = None
        gt_38: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_14, 0.1);  convert_element_type_default_14 = None
        mul_110: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_38, view_171);  view_171 = None
        mul_111: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, 1.1111111111111112);  mul_110 = None
        add_53: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, mul_111);  mul_106 = mul_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_15: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_53, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_15: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_53, [-1], correction = 1.0, keepdim = True)
        sqrt_15: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_15);  var_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_23: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_53, mean_15);  mean_15 = None
        mul_112: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_128, sub_23)
        add_54: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_15, 1e-06)
        div_31: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_112, add_54);  mul_112 = add_54 = None
        add_55: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_31, primals_129);  div_31 = primals_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_323: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        convert_element_type_324: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        convert_element_type_325: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        view_172: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_325, [2048, 768]);  convert_element_type_325 = None
        permute_86: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_324, [1, 0]);  convert_element_type_324 = None
        addmm_46: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_323, view_172, permute_86);  convert_element_type_323 = None
        view_173: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [16, 128, 3072])
        convert_element_type_329: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_113: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, 0.5)
        mul_114: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, 0.7071067811865476);  convert_element_type_329 = None
        erf_7: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
        add_56: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_115: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, add_56);  mul_113 = add_56 = None
        convert_element_type_330: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_115, torch.bfloat16);  mul_115 = None
        inductor_lookup_seed_default_38: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_22: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        convert_element_type_default_13: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_22, torch.bfloat16);  inductor_random_default_22 = None
        gt_39: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_13, 0.1);  convert_element_type_default_13 = None
        mul_116: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_39, convert_element_type_330);  convert_element_type_330 = None
        mul_117: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, 1.1111111111111112);  mul_116 = None
        convert_element_type_331: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        convert_element_type_332: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        view_174: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_117, [2048, 3072]);  mul_117 = None
        permute_87: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_332, [1, 0]);  convert_element_type_332 = None
        addmm_47: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_331, view_174, permute_87);  convert_element_type_331 = None
        view_175: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [16, 128, 768]);  addmm_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_39: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_21: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        convert_element_type_default_12: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_21, torch.bfloat16);  inductor_random_default_21 = None
        gt_40: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_12, 0.1);  convert_element_type_default_12 = None
        mul_118: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_40, view_175);  view_175 = None
        mul_119: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, 1.1111111111111112);  mul_118 = None
        add_57: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_53, mul_119);  add_53 = mul_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_40: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_20: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        gt_41: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_120: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_41, add_57);  add_57 = None
        mul_121: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_16: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_121, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_16: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_121, [-1], correction = 1.0, keepdim = True)
        sqrt_16: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_16);  var_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_24: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_121, mean_16);  mean_16 = None
        mul_122: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_134, sub_24)
        add_58: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_16, 1e-06)
        div_32: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_122, add_58);  mul_122 = add_58 = None
        add_59: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_32, primals_135);  div_32 = primals_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_336: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_337: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        convert_element_type_338: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        view_176: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_338, [2048, 768]);  convert_element_type_338 = None
        permute_88: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_337, [1, 0]);  convert_element_type_337 = None
        addmm_48: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_336, view_176, permute_88);  convert_element_type_336 = None
        view_177: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [16, 128, 768]);  addmm_48 = None
        view_178: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [16, -1, 12, 64]);  view_177 = None
        permute_89: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        convert_element_type_342: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        convert_element_type_343: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        permute_90: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_343, [1, 0]);  convert_element_type_343 = None
        addmm_49: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_342, view_176, permute_90);  convert_element_type_342 = None
        view_180: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [16, 128, 768]);  addmm_49 = None
        view_181: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_180, [16, -1, 12, 64]);  view_180 = None
        permute_91: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        convert_element_type_348: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        convert_element_type_349: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        permute_92: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_349, [1, 0]);  convert_element_type_349 = None
        addmm_50: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_348, view_176, permute_92);  convert_element_type_348 = None
        view_183: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [16, 128, 768]);  addmm_50 = None
        view_184: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_183, [16, -1, 12, 64]);  view_183 = None
        permute_93: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_94: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        expand_32: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_89, [16, 12, 128, 64]);  permute_89 = None
        clone_32: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_185: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [192, 128, 64]);  clone_32 = None
        expand_33: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_94, [16, 12, 64, 128]);  permute_94 = None
        clone_33: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_186: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [192, 64, 128]);  clone_33 = None
        bmm_16: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_185, view_186)
        view_187: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [16, 12, 128, 128]);  bmm_16 = None
        div_33: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_8: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_33);  div_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_356: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32)
        amax_8: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_356, [-1], True)
        sub_25: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_356, amax_8);  convert_element_type_356 = None
        exp_8: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_34: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_41: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_19: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        gt_42: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_123: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_42, div_34);  div_34 = None
        mul_124: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, 1.1111111111111112);  mul_123 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_357: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_124, torch.bfloat16);  mul_124 = None
        expand_34: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_357, [16, 12, 128, 128]);  convert_element_type_357 = None
        view_188: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_34, [192, 128, 128]);  expand_34 = None
        expand_35: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_93, [16, 12, 128, 64]);  permute_93 = None
        clone_34: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_189: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [192, 128, 64]);  clone_34 = None
        bmm_17: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_188, view_189)
        view_190: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [16, 12, 128, 64]);  bmm_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_95: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_35: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_191: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [16, -1, 768]);  clone_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_360: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        convert_element_type_361: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        view_192: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [2048, 768]);  view_191 = None
        permute_96: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_361, [1, 0]);  convert_element_type_361 = None
        addmm_51: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_360, view_192, permute_96);  convert_element_type_360 = None
        view_193: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [16, 128, 768]);  addmm_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_42: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_18: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        convert_element_type_default_11: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_18, torch.bfloat16);  inductor_random_default_18 = None
        gt_43: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_11, 0.1);  convert_element_type_default_11 = None
        mul_125: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_43, view_193);  view_193 = None
        mul_126: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, 1.1111111111111112);  mul_125 = None
        add_60: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, mul_126);  mul_121 = mul_126 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_17: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_60, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_17: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_60, [-1], correction = 1.0, keepdim = True)
        sqrt_17: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_17);  var_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_26: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_60, mean_17);  mean_17 = None
        mul_127: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_144, sub_26)
        add_61: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_17, 1e-06)
        div_35: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_127, add_61);  mul_127 = add_61 = None
        add_62: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_35, primals_145);  div_35 = primals_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_365: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        convert_element_type_366: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convert_element_type_367: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None
        view_194: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_367, [2048, 768]);  convert_element_type_367 = None
        permute_97: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_366, [1, 0]);  convert_element_type_366 = None
        addmm_52: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_365, view_194, permute_97);  convert_element_type_365 = None
        view_195: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [16, 128, 3072])
        convert_element_type_371: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_128: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, 0.5)
        mul_129: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, 0.7071067811865476);  convert_element_type_371 = None
        erf_8: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_63: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_130: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, add_63);  mul_128 = add_63 = None
        convert_element_type_372: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_130, torch.bfloat16);  mul_130 = None
        inductor_lookup_seed_default_43: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_17: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        convert_element_type_default_10: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_17, torch.bfloat16);  inductor_random_default_17 = None
        gt_44: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_10, 0.1);  convert_element_type_default_10 = None
        mul_131: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_44, convert_element_type_372);  convert_element_type_372 = None
        mul_132: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, 1.1111111111111112);  mul_131 = None
        convert_element_type_373: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        convert_element_type_374: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        view_196: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_132, [2048, 3072]);  mul_132 = None
        permute_98: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_374, [1, 0]);  convert_element_type_374 = None
        addmm_53: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_373, view_196, permute_98);  convert_element_type_373 = None
        view_197: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [16, 128, 768]);  addmm_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_44: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_16: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        convert_element_type_default_9: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_16, torch.bfloat16);  inductor_random_default_16 = None
        gt_45: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_9, 0.1);  convert_element_type_default_9 = None
        mul_133: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_45, view_197);  view_197 = None
        mul_134: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, 1.1111111111111112);  mul_133 = None
        add_64: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_60, mul_134);  add_60 = mul_134 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_45: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_15: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        gt_46: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_135: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_46, add_64);  add_64 = None
        mul_136: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_18: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_136, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_18: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_136, [-1], correction = 1.0, keepdim = True)
        sqrt_18: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_18);  var_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_27: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_136, mean_18);  mean_18 = None
        mul_137: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_150, sub_27)
        add_65: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_18, 1e-06)
        div_36: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_137, add_65);  mul_137 = add_65 = None
        add_66: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_36, primals_151);  div_36 = primals_151 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_378: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        convert_element_type_379: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        convert_element_type_380: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None
        view_198: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_380, [2048, 768]);  convert_element_type_380 = None
        permute_99: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_379, [1, 0]);  convert_element_type_379 = None
        addmm_54: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_378, view_198, permute_99);  convert_element_type_378 = None
        view_199: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [16, 128, 768]);  addmm_54 = None
        view_200: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [16, -1, 12, 64]);  view_199 = None
        permute_100: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        convert_element_type_384: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        convert_element_type_385: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16);  primals_154 = None
        permute_101: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_385, [1, 0]);  convert_element_type_385 = None
        addmm_55: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_384, view_198, permute_101);  convert_element_type_384 = None
        view_202: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [16, 128, 768]);  addmm_55 = None
        view_203: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_202, [16, -1, 12, 64]);  view_202 = None
        permute_102: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
        convert_element_type_390: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        convert_element_type_391: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        permute_103: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_391, [1, 0]);  convert_element_type_391 = None
        addmm_56: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_390, view_198, permute_103);  convert_element_type_390 = None
        view_205: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [16, 128, 768]);  addmm_56 = None
        view_206: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_205, [16, -1, 12, 64]);  view_205 = None
        permute_104: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_105: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
        expand_36: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_100, [16, 12, 128, 64]);  permute_100 = None
        clone_36: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_207: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [192, 128, 64]);  clone_36 = None
        expand_37: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_105, [16, 12, 64, 128]);  permute_105 = None
        clone_37: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_208: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [192, 64, 128]);  clone_37 = None
        bmm_18: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_207, view_208)
        view_209: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [16, 12, 128, 128]);  bmm_18 = None
        div_37: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_9: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_37);  div_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_398: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32)
        amax_9: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_398, [-1], True)
        sub_28: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_398, amax_9);  convert_element_type_398 = None
        exp_9: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_38: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_46: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_14: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        gt_47: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_138: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_47, div_38);  div_38 = None
        mul_139: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, 1.1111111111111112);  mul_138 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_399: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_139, torch.bfloat16);  mul_139 = None
        expand_38: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_399, [16, 12, 128, 128]);  convert_element_type_399 = None
        view_210: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_38, [192, 128, 128]);  expand_38 = None
        expand_39: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_104, [16, 12, 128, 64]);  permute_104 = None
        clone_38: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_211: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [192, 128, 64]);  clone_38 = None
        bmm_19: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_210, view_211)
        view_212: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [16, 12, 128, 64]);  bmm_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_106: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_39: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_213: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [16, -1, 768]);  clone_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_402: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        convert_element_type_403: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        view_214: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_213, [2048, 768]);  view_213 = None
        permute_107: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_403, [1, 0]);  convert_element_type_403 = None
        addmm_57: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_402, view_214, permute_107);  convert_element_type_402 = None
        view_215: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [16, 128, 768]);  addmm_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_47: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_13: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        convert_element_type_default_8: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_13, torch.bfloat16);  inductor_random_default_13 = None
        gt_48: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_8, 0.1);  convert_element_type_default_8 = None
        mul_140: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_48, view_215);  view_215 = None
        mul_141: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, 1.1111111111111112);  mul_140 = None
        add_67: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_136, mul_141);  mul_136 = mul_141 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_19: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_67, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_19: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_67, [-1], correction = 1.0, keepdim = True)
        sqrt_19: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_19);  var_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_29: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_67, mean_19);  mean_19 = None
        mul_142: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_160, sub_29)
        add_68: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_19, 1e-06)
        div_39: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_142, add_68);  mul_142 = add_68 = None
        add_69: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_39, primals_161);  div_39 = primals_161 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_407: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        convert_element_type_408: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16);  primals_162 = None
        convert_element_type_409: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None
        view_216: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_409, [2048, 768]);  convert_element_type_409 = None
        permute_108: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_408, [1, 0]);  convert_element_type_408 = None
        addmm_58: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_407, view_216, permute_108);  convert_element_type_407 = None
        view_217: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [16, 128, 3072])
        convert_element_type_413: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_143: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, 0.5)
        mul_144: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, 0.7071067811865476);  convert_element_type_413 = None
        erf_9: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_144);  mul_144 = None
        add_70: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_145: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, add_70);  mul_143 = add_70 = None
        convert_element_type_414: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_145, torch.bfloat16);  mul_145 = None
        inductor_lookup_seed_default_48: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default_12: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        convert_element_type_default_7: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_12, torch.bfloat16);  inductor_random_default_12 = None
        gt_49: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_7, 0.1);  convert_element_type_default_7 = None
        mul_146: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_49, convert_element_type_414);  convert_element_type_414 = None
        mul_147: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, 1.1111111111111112);  mul_146 = None
        convert_element_type_415: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16);  primals_165 = None
        convert_element_type_416: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        view_218: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_147, [2048, 3072]);  mul_147 = None
        permute_109: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_416, [1, 0]);  convert_element_type_416 = None
        addmm_59: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_415, view_218, permute_109);  convert_element_type_415 = None
        view_219: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [16, 128, 768]);  addmm_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_49: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49)
        inductor_random_default_11: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_49, 'rand');  inductor_lookup_seed_default_49 = None
        convert_element_type_default_6: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.bfloat16);  inductor_random_default_11 = None
        gt_50: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_6, 0.1);  convert_element_type_default_6 = None
        mul_148: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_50, view_219);  view_219 = None
        mul_149: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, 1.1111111111111112);  mul_148 = None
        add_71: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_67, mul_149);  add_67 = mul_149 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_50: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50)
        inductor_random_default_10: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_50, 'rand');  inductor_lookup_seed_default_50 = None
        gt_51: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_150: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_51, add_71);  add_71 = None
        mul_151: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 1.1111111111111112);  mul_150 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_20: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_151, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_20: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_151, [-1], correction = 1.0, keepdim = True)
        sqrt_20: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_20);  var_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_30: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_151, mean_20);  mean_20 = None
        mul_152: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_166, sub_30)
        add_72: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_20, 1e-06)
        div_40: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_152, add_72);  mul_152 = add_72 = None
        add_73: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_40, primals_167);  div_40 = primals_167 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_420: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        convert_element_type_421: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_168, torch.bfloat16);  primals_168 = None
        convert_element_type_422: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None
        view_220: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_422, [2048, 768]);  convert_element_type_422 = None
        permute_110: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_421, [1, 0]);  convert_element_type_421 = None
        addmm_60: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_420, view_220, permute_110);  convert_element_type_420 = None
        view_221: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [16, 128, 768]);  addmm_60 = None
        view_222: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [16, -1, 12, 64]);  view_221 = None
        permute_111: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        convert_element_type_426: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        convert_element_type_427: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        permute_112: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_427, [1, 0]);  convert_element_type_427 = None
        addmm_61: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_426, view_220, permute_112);  convert_element_type_426 = None
        view_224: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [16, 128, 768]);  addmm_61 = None
        view_225: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_224, [16, -1, 12, 64]);  view_224 = None
        permute_113: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        convert_element_type_432: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_173, torch.bfloat16);  primals_173 = None
        convert_element_type_433: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        permute_114: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_433, [1, 0]);  convert_element_type_433 = None
        addmm_62: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_432, view_220, permute_114);  convert_element_type_432 = None
        view_227: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [16, 128, 768]);  addmm_62 = None
        view_228: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_227, [16, -1, 12, 64]);  view_227 = None
        permute_115: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_116: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
        expand_40: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_111, [16, 12, 128, 64]);  permute_111 = None
        clone_40: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_229: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [192, 128, 64]);  clone_40 = None
        expand_41: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_116, [16, 12, 64, 128]);  permute_116 = None
        clone_41: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_230: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [192, 64, 128]);  clone_41 = None
        bmm_20: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_229, view_230)
        view_231: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [16, 12, 128, 128]);  bmm_20 = None
        div_41: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_10: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_41);  div_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_440: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32)
        amax_10: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_440, [-1], True)
        sub_31: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_440, amax_10);  convert_element_type_440 = None
        exp_10: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_42: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_51: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51)
        inductor_random_default_9: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_51, 'rand');  inductor_lookup_seed_default_51 = None
        gt_52: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_153: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_52, div_42);  div_42 = None
        mul_154: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_153, 1.1111111111111112);  mul_153 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_441: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None
        expand_42: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_441, [16, 12, 128, 128]);  convert_element_type_441 = None
        view_232: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_42, [192, 128, 128]);  expand_42 = None
        expand_43: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_115, [16, 12, 128, 64]);  permute_115 = None
        clone_42: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_233: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [192, 128, 64]);  clone_42 = None
        bmm_21: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_232, view_233)
        view_234: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [16, 12, 128, 64]);  bmm_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_117: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_43: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_235: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [16, -1, 768]);  clone_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_444: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        convert_element_type_445: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16);  primals_174 = None
        view_236: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [2048, 768]);  view_235 = None
        permute_118: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_445, [1, 0]);  convert_element_type_445 = None
        addmm_63: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_444, view_236, permute_118);  convert_element_type_444 = None
        view_237: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [16, 128, 768]);  addmm_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_52: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 52)
        inductor_random_default_8: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_52, 'rand');  inductor_lookup_seed_default_52 = None
        convert_element_type_default_5: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.bfloat16);  inductor_random_default_8 = None
        gt_53: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_5, 0.1);  convert_element_type_default_5 = None
        mul_155: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_53, view_237);  view_237 = None
        mul_156: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, 1.1111111111111112);  mul_155 = None
        add_74: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, mul_156);  mul_151 = mul_156 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_21: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_74, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_21: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_74, [-1], correction = 1.0, keepdim = True)
        sqrt_21: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_21);  var_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_32: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_74, mean_21);  mean_21 = None
        mul_157: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_176, sub_32)
        add_75: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_21, 1e-06)
        div_43: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_157, add_75);  mul_157 = add_75 = None
        add_76: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_43, primals_177);  div_43 = primals_177 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_449: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        convert_element_type_450: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        convert_element_type_451: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None
        view_238: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_451, [2048, 768]);  convert_element_type_451 = None
        permute_119: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_450, [1, 0]);  convert_element_type_450 = None
        addmm_64: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_449, view_238, permute_119);  convert_element_type_449 = None
        view_239: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [16, 128, 3072])
        convert_element_type_455: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_158: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, 0.5)
        mul_159: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, 0.7071067811865476);  convert_element_type_455 = None
        erf_10: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
        add_77: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_160: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, add_77);  mul_158 = add_77 = None
        convert_element_type_456: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_160, torch.bfloat16);  mul_160 = None
        inductor_lookup_seed_default_53: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 53)
        inductor_random_default_7: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_53, 'rand');  inductor_lookup_seed_default_53 = None
        convert_element_type_default_4: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.bfloat16);  inductor_random_default_7 = None
        gt_54: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_4, 0.1);  convert_element_type_default_4 = None
        mul_161: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_54, convert_element_type_456);  convert_element_type_456 = None
        mul_162: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, 1.1111111111111112);  mul_161 = None
        convert_element_type_457: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        convert_element_type_458: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16);  primals_180 = None
        view_240: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_162, [2048, 3072]);  mul_162 = None
        permute_120: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_458, [1, 0]);  convert_element_type_458 = None
        addmm_65: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_457, view_240, permute_120);  convert_element_type_457 = None
        view_241: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [16, 128, 768]);  addmm_65 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_54: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 54)
        inductor_random_default_6: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_54, 'rand');  inductor_lookup_seed_default_54 = None
        convert_element_type_default_3: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.bfloat16);  inductor_random_default_6 = None
        gt_55: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_3, 0.1);  convert_element_type_default_3 = None
        mul_163: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_55, view_241);  view_241 = None
        mul_164: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, 1.1111111111111112);  mul_163 = None
        add_78: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_74, mul_164);  add_74 = mul_164 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_55: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 55)
        inductor_random_default_5: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_55, 'rand');  inductor_lookup_seed_default_55 = None
        gt_56: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_165: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_56, add_78);  add_78 = None
        mul_166: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, 1.1111111111111112);  mul_165 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_22: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(mul_166, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_22: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(mul_166, [-1], correction = 1.0, keepdim = True)
        sqrt_22: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_22);  var_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_33: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_166, mean_22);  mean_22 = None
        mul_167: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_182, sub_33)
        add_79: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_22, 1e-06)
        div_44: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_167, add_79);  mul_167 = add_79 = None
        add_80: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_44, primals_183);  div_44 = primals_183 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        convert_element_type_462: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        convert_element_type_463: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        convert_element_type_464: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None
        view_242: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_464, [2048, 768]);  convert_element_type_464 = None
        permute_121: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_463, [1, 0]);  convert_element_type_463 = None
        addmm_66: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_462, view_242, permute_121);  convert_element_type_462 = None
        view_243: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [16, 128, 768]);  addmm_66 = None
        view_244: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_243, [16, -1, 12, 64]);  view_243 = None
        permute_122: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        convert_element_type_468: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_187, torch.bfloat16);  primals_187 = None
        convert_element_type_469: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_186, torch.bfloat16);  primals_186 = None
        permute_123: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_469, [1, 0]);  convert_element_type_469 = None
        addmm_67: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_468, view_242, permute_123);  convert_element_type_468 = None
        view_246: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [16, 128, 768]);  addmm_67 = None
        view_247: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_246, [16, -1, 12, 64]);  view_246 = None
        permute_124: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        convert_element_type_474: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        convert_element_type_475: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        permute_125: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_475, [1, 0]);  convert_element_type_475 = None
        addmm_68: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_474, view_242, permute_125);  convert_element_type_474 = None
        view_249: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [16, 128, 768]);  addmm_68 = None
        view_250: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_249, [16, -1, 12, 64]);  view_249 = None
        permute_126: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_127: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
        expand_44: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_122, [16, 12, 128, 64]);  permute_122 = None
        clone_44: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_251: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [192, 128, 64]);  clone_44 = None
        expand_45: "bf16[16, 12, 64, 128][98304, 64, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_127, [16, 12, 64, 128]);  permute_127 = None
        clone_45: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_252: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [192, 64, 128]);  clone_45 = None
        bmm_22: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_251, view_252)
        view_253: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [16, 12, 128, 128]);  bmm_22 = None
        div_45: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_11: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_45);  eq = full_default = div_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_482: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32)
        amax_11: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_482, [-1], True)
        sub_34: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_482, amax_11);  convert_element_type_482 = None
        exp_11: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_46: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_56: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56)
        inductor_random_default_4: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 12, 128, 128], inductor_lookup_seed_default_56, 'rand');  inductor_lookup_seed_default_56 = None
        gt_57: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_168: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_57, div_46);  div_46 = None
        mul_169: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, 1.1111111111111112);  mul_168 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        convert_element_type_483: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_169, torch.bfloat16);  mul_169 = None
        expand_46: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_483, [16, 12, 128, 128]);  convert_element_type_483 = None
        view_254: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_46, [192, 128, 128]);  expand_46 = None
        expand_47: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.expand.default(permute_126, [16, 12, 128, 64]);  permute_126 = None
        clone_46: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_255: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [192, 128, 64]);  clone_46 = None
        bmm_23: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_254, view_255)
        view_256: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [16, 12, 128, 64]);  bmm_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_128: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_47: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_257: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [16, -1, 768]);  clone_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        convert_element_type_486: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16);  primals_191 = None
        convert_element_type_487: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16);  primals_190 = None
        view_258: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [2048, 768]);  view_257 = None
        permute_129: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_487, [1, 0]);  convert_element_type_487 = None
        addmm_69: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_486, view_258, permute_129);  convert_element_type_486 = None
        view_259: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [16, 128, 768]);  addmm_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_57: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57)
        inductor_random_default_3: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_57, 'rand');  inductor_lookup_seed_default_57 = None
        convert_element_type_default_2: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.bfloat16);  inductor_random_default_3 = None
        gt_58: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_170: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_58, view_259);  view_259 = None
        mul_171: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, 1.1111111111111112);  mul_170 = None
        add_81: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_166, mul_171);  mul_166 = mul_171 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_23: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_81, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_23: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.var.correction(add_81, [-1], correction = 1.0, keepdim = True)
        sqrt_23: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sqrt.default(var_23);  var_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_35: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_81, mean_23);  mean_23 = None
        mul_172: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_192, sub_35)
        add_82: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_23, 1e-06)
        div_47: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_172, add_82);  mul_172 = add_82 = None
        add_83: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(div_47, primals_193);  div_47 = primals_193 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        convert_element_type_491: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_195, torch.bfloat16);  primals_195 = None
        convert_element_type_492: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        convert_element_type_493: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None
        view_260: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_493, [2048, 768]);  convert_element_type_493 = None
        permute_130: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_492, [1, 0]);  convert_element_type_492 = None
        addmm_70: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_491, view_260, permute_130);  convert_element_type_491 = None
        view_261: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [16, 128, 3072])
        convert_element_type_497: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_173: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.5)
        mul_174: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.7071067811865476);  convert_element_type_497 = None
        erf_11: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_84: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_175: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, add_84);  mul_173 = add_84 = None
        convert_element_type_498: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None
        inductor_lookup_seed_default_58: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58)
        inductor_random_default_2: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 3072], inductor_lookup_seed_default_58, 'rand');  inductor_lookup_seed_default_58 = None
        convert_element_type_default_1: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.bfloat16);  inductor_random_default_2 = None
        gt_59: "b8[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_176: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_59, convert_element_type_498);  convert_element_type_498 = None
        mul_177: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, 1.1111111111111112);  mul_176 = None
        convert_element_type_499: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16);  primals_197 = None
        convert_element_type_500: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_196, torch.bfloat16);  primals_196 = None
        view_262: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_177, [2048, 3072]);  mul_177 = None
        permute_131: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_500, [1, 0]);  convert_element_type_500 = None
        addmm_71: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_499, view_262, permute_131);  convert_element_type_499 = None
        view_263: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [16, 128, 768]);  addmm_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default_59: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 59)
        inductor_random_default_1: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_59, 'rand');  inductor_lookup_seed_default_59 = None
        convert_element_type_default: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_60: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_178: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_60, view_263);  view_263 = None
        mul_179: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, 1.1111111111111112);  mul_178 = None
        add_85: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_81, mul_179);  add_81 = mul_179 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default_60: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60);  inductor_seeds_default = None
        inductor_random_default: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 128, 768], inductor_lookup_seed_default_60, 'rand');  inductor_lookup_seed_default_60 = None
        gt_61: "b8[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_180: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_61, add_85);  add_85 = None
        mul_181: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, 1.1111111111111112);  mul_180 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:42 in forward, code: return self.softmax(self.linear(x[:, 0]))
        select: "f32[16, 768][98304, 1]cuda:0" = torch.ops.aten.select.int(mul_181, 1, 0)
        convert_element_type_504: "bf16[2][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16);  primals_199 = None
        convert_element_type_505: "bf16[2, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        convert_element_type_506: "bf16[16, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(select, torch.bfloat16);  select = None
        permute_132: "bf16[768, 2][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_505, [1, 0]);  convert_element_type_505 = None
        addmm_72: "bf16[16, 2][2, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_504, convert_element_type_506, permute_132);  convert_element_type_504 = None
        convert_element_type_510: "f32[16, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_72, torch.float32);  addmm_72 = None
        amax_12: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_510, [-1], True)
        sub_36: "f32[16, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_510, amax_12);  convert_element_type_510 = amax_12 = None
        exp_12: "f32[16, 2][2, 1]cuda:0" = torch.ops.aten.exp.default(sub_36)
        sum_13: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True);  exp_12 = None
        log: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_37: "f32[16, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_36, log);  sub_36 = log = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:61 in forward, code: return self.softmax(self.linear(x))
        convert_element_type_511: "bf16[20005][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16);  primals_201 = None
        convert_element_type_512: "bf16[20005, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        convert_element_type_513: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        view_264: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_513, [2048, 768]);  convert_element_type_513 = None
        permute_133: "bf16[768, 20005][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_512, [1, 0]);  convert_element_type_512 = None
        constant_pad_nd_default_3: "bf16[768, 20008][20008, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_133, [0, 3, 0, 0])
        full_default_51: "bf16[3][1]cuda:0" = torch.ops.aten.full.default([3], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "bf16[20008][1]cuda:0" = torch.ops.aten.cat.default([convert_element_type_511, full_default_51]);  convert_element_type_511 = full_default_51 = None
        addmm_default: "bf16[2048, 20008][20008, 1]cuda:0" = torch.ops.aten.addmm.default(cat_default, view_264, constant_pad_nd_default_3);  cat_default = constant_pad_nd_default_3 = None
        slice_tensor_1: "bf16[2048, 20005][20008, 1]cuda:0" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -3);  addmm_default = None
        view_265: "bf16[16, 128, 20005][2561024, 20008, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_1, [16, 128, 20005]);  slice_tensor_1 = None
        convert_element_type_517: "f32[16, 128, 20005][2560640, 20005, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        amax_13: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_517, [-1], True)
        sub_38: "f32[16, 128, 20005][2560640, 20005, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_517, amax_13);  convert_element_type_517 = amax_13 = None
        exp_13: "f32[16, 128, 20005][2560640, 20005, 1]cuda:0" = torch.ops.aten.exp.default(sub_38)
        sum_14: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True);  exp_13 = None
        log_1: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.log.default(sum_14);  sum_14 = None
        sub_39: "f32[16, 128, 20005][2560640, 20005, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, log_1);  sub_38 = log_1 = None
        permute_134: "bf16[20005, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:42 in forward, code: return self.softmax(self.linear(x[:, 0]))
        permute_138: "bf16[2, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_142: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        permute_146: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_150: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_155: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_156: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_157: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_158: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_161: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        permute_166: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        permute_171: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_175: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        permute_179: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_183: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_188: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        permute_189: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_190: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_191: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_194: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        permute_199: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        permute_204: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_208: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        permute_212: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_216: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_221: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_222: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_223: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_224: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_227: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        permute_232: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_237: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_241: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        permute_245: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_249: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_254: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_255: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_256: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_257: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_260: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        permute_265: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        permute_270: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_274: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        permute_278: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_282: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_287: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_288: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_289: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_290: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_293: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_298: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        permute_303: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_307: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_311: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_315: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_320: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_321: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_322: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_323: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_326: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        permute_331: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        permute_336: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_340: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_344: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_348: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_353: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_354: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_355: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_356: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_359: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        permute_364: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        permute_369: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_373: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_377: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_381: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_386: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_387: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_388: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_389: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_392: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        permute_397: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_402: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_406: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_410: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_414: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_419: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_420: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_421: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_422: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_425: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_430: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_435: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_439: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_443: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_447: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_452: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_453: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_454: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_455: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_458: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        permute_463: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_468: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_472: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_476: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_480: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_485: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_486: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_487: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_488: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_491: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        permute_496: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        permute_501: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_505: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_509: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_513: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        permute_518: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_519: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_520: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_521: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_524: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_529: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_534: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (sub_37, sub_39, unsqueeze_1, primals_1, primals_5, primals_6, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, primals_112, primals_118, primals_128, primals_134, primals_144, primals_150, primals_160, primals_166, primals_176, primals_182, primals_192, unsqueeze_1, gt_1, sqrt, sub, view, bmm, amax, sum_1, gt_2, view_16, gt_3, sqrt_1, sub_2, view_18, addmm_4, gt_4, view_20, gt_5, gt_6, sqrt_2, sub_3, view_22, where_1, amax_1, sum_2, gt_7, view_38, gt_8, sqrt_3, sub_5, view_40, addmm_10, gt_9, view_42, gt_10, gt_11, sqrt_4, sub_6, view_44, where_2, amax_2, sum_3, gt_12, view_60, gt_13, sqrt_5, sub_8, view_62, addmm_16, gt_14, view_64, gt_15, gt_16, sqrt_6, sub_9, view_66, where_3, amax_3, sum_4, gt_17, view_82, gt_18, sqrt_7, sub_11, view_84, addmm_22, gt_19, view_86, gt_20, gt_21, sqrt_8, sub_12, view_88, where_4, amax_4, sum_5, gt_22, view_104, gt_23, sqrt_9, sub_14, view_106, addmm_28, gt_24, view_108, gt_25, gt_26, sqrt_10, sub_15, view_110, where_5, amax_5, sum_6, gt_27, view_126, gt_28, sqrt_11, sub_17, view_128, addmm_34, gt_29, view_130, gt_30, gt_31, sqrt_12, sub_18, view_132, where_6, amax_6, sum_7, gt_32, view_148, gt_33, sqrt_13, sub_20, view_150, addmm_40, gt_34, view_152, gt_35, gt_36, sqrt_14, sub_21, view_154, where_7, amax_7, sum_8, gt_37, view_170, gt_38, sqrt_15, sub_23, view_172, addmm_46, gt_39, view_174, gt_40, gt_41, sqrt_16, sub_24, view_176, where_8, amax_8, sum_9, gt_42, view_192, gt_43, sqrt_17, sub_26, view_194, addmm_52, gt_44, view_196, gt_45, gt_46, sqrt_18, sub_27, view_198, where_9, amax_9, sum_10, gt_47, view_214, gt_48, sqrt_19, sub_29, view_216, addmm_58, gt_49, view_218, gt_50, gt_51, sqrt_20, sub_30, view_220, where_10, amax_10, sum_11, gt_52, view_236, gt_53, sqrt_21, sub_32, view_238, addmm_64, gt_54, view_240, gt_55, gt_56, sqrt_22, sub_33, view_242, where_11, amax_11, sum_12, gt_57, view_258, gt_58, sqrt_23, sub_35, view_260, addmm_70, gt_59, view_262, gt_60, gt_61, convert_element_type_506, sub_37, view_264, sub_39, permute_134, permute_138, permute_142, permute_146, permute_150, permute_155, permute_156, permute_157, permute_158, permute_161, permute_166, permute_171, permute_175, permute_179, permute_183, permute_188, permute_189, permute_190, permute_191, permute_194, permute_199, permute_204, permute_208, permute_212, permute_216, permute_221, permute_222, permute_223, permute_224, permute_227, permute_232, permute_237, permute_241, permute_245, permute_249, permute_254, permute_255, permute_256, permute_257, permute_260, permute_265, permute_270, permute_274, permute_278, permute_282, permute_287, permute_288, permute_289, permute_290, permute_293, permute_298, permute_303, permute_307, permute_311, permute_315, permute_320, permute_321, permute_322, permute_323, permute_326, permute_331, permute_336, permute_340, permute_344, permute_348, permute_353, permute_354, permute_355, permute_356, permute_359, permute_364, permute_369, permute_373, permute_377, permute_381, permute_386, permute_387, permute_388, permute_389, permute_392, permute_397, permute_402, permute_406, permute_410, permute_414, permute_419, permute_420, permute_421, permute_422, permute_425, permute_430, permute_435, permute_439, permute_443, permute_447, permute_452, permute_453, permute_454, permute_455, permute_458, permute_463, permute_468, permute_472, permute_476, permute_480, permute_485, permute_486, permute_487, permute_488, permute_491, permute_496, permute_501, permute_505, permute_509, permute_513, permute_518, permute_519, permute_520, permute_521, permute_524, permute_529, permute_534)

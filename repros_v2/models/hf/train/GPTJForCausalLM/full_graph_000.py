class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1, 128][128, 1]cuda:0", primals_2: "f32[50400, 4096][4096, 1]cuda:0", primals_3: "f32[4096][1]cuda:0", primals_4: "f32[4096][1]cuda:0", primals_5: "f32[4096, 4096][4096, 1]cuda:0", primals_6: "f32[4096, 4096][4096, 1]cuda:0", primals_7: "f32[4096, 4096][4096, 1]cuda:0", primals_8: "f32[2048, 64][64, 1]cuda:0", primals_9: "f32[4096, 4096][4096, 1]cuda:0", primals_10: "f32[16384, 4096][4096, 1]cuda:0", primals_11: "f32[16384][1]cuda:0", primals_12: "f32[4096, 16384][16384, 1]cuda:0", primals_13: "f32[4096][1]cuda:0", primals_14: "f32[4096][1]cuda:0", primals_15: "f32[4096][1]cuda:0", primals_16: "f32[4096, 4096][4096, 1]cuda:0", primals_17: "f32[4096, 4096][4096, 1]cuda:0", primals_18: "f32[4096, 4096][4096, 1]cuda:0", primals_19: "f32[2048, 64][64, 1]cuda:0", primals_20: "f32[4096, 4096][4096, 1]cuda:0", primals_21: "f32[16384, 4096][4096, 1]cuda:0", primals_22: "f32[16384][1]cuda:0", primals_23: "f32[4096, 16384][16384, 1]cuda:0", primals_24: "f32[4096][1]cuda:0", primals_25: "f32[4096][1]cuda:0", primals_26: "f32[4096][1]cuda:0", primals_27: "f32[4096, 4096][4096, 1]cuda:0", primals_28: "f32[4096, 4096][4096, 1]cuda:0", primals_29: "f32[4096, 4096][4096, 1]cuda:0", primals_30: "f32[2048, 64][64, 1]cuda:0", primals_31: "f32[4096, 4096][4096, 1]cuda:0", primals_32: "f32[16384, 4096][4096, 1]cuda:0", primals_33: "f32[16384][1]cuda:0", primals_34: "f32[4096, 16384][16384, 1]cuda:0", primals_35: "f32[4096][1]cuda:0", primals_36: "f32[4096][1]cuda:0", primals_37: "f32[4096][1]cuda:0", primals_38: "f32[4096, 4096][4096, 1]cuda:0", primals_39: "f32[4096, 4096][4096, 1]cuda:0", primals_40: "f32[4096, 4096][4096, 1]cuda:0", primals_41: "f32[2048, 64][64, 1]cuda:0", primals_42: "f32[4096, 4096][4096, 1]cuda:0", primals_43: "f32[16384, 4096][4096, 1]cuda:0", primals_44: "f32[16384][1]cuda:0", primals_45: "f32[4096, 16384][16384, 1]cuda:0", primals_46: "f32[4096][1]cuda:0", primals_47: "f32[4096][1]cuda:0", primals_48: "f32[4096][1]cuda:0", primals_49: "f32[4096, 4096][4096, 1]cuda:0", primals_50: "f32[4096, 4096][4096, 1]cuda:0", primals_51: "f32[4096, 4096][4096, 1]cuda:0", primals_52: "f32[2048, 64][64, 1]cuda:0", primals_53: "f32[4096, 4096][4096, 1]cuda:0", primals_54: "f32[16384, 4096][4096, 1]cuda:0", primals_55: "f32[16384][1]cuda:0", primals_56: "f32[4096, 16384][16384, 1]cuda:0", primals_57: "f32[4096][1]cuda:0", primals_58: "f32[4096][1]cuda:0", primals_59: "f32[4096][1]cuda:0", primals_60: "f32[4096, 4096][4096, 1]cuda:0", primals_61: "f32[4096, 4096][4096, 1]cuda:0", primals_62: "f32[4096, 4096][4096, 1]cuda:0", primals_63: "f32[2048, 64][64, 1]cuda:0", primals_64: "f32[4096, 4096][4096, 1]cuda:0", primals_65: "f32[16384, 4096][4096, 1]cuda:0", primals_66: "f32[16384][1]cuda:0", primals_67: "f32[4096, 16384][16384, 1]cuda:0", primals_68: "f32[4096][1]cuda:0", primals_69: "f32[4096][1]cuda:0", primals_70: "f32[4096][1]cuda:0", primals_71: "f32[4096, 4096][4096, 1]cuda:0", primals_72: "f32[4096, 4096][4096, 1]cuda:0", primals_73: "f32[4096, 4096][4096, 1]cuda:0", primals_74: "f32[2048, 64][64, 1]cuda:0", primals_75: "f32[4096, 4096][4096, 1]cuda:0", primals_76: "f32[16384, 4096][4096, 1]cuda:0", primals_77: "f32[16384][1]cuda:0", primals_78: "f32[4096, 16384][16384, 1]cuda:0", primals_79: "f32[4096][1]cuda:0", primals_80: "f32[4096][1]cuda:0", primals_81: "f32[4096][1]cuda:0", primals_82: "f32[4096, 4096][4096, 1]cuda:0", primals_83: "f32[4096, 4096][4096, 1]cuda:0", primals_84: "f32[4096, 4096][4096, 1]cuda:0", primals_85: "f32[2048, 64][64, 1]cuda:0", primals_86: "f32[4096, 4096][4096, 1]cuda:0", primals_87: "f32[16384, 4096][4096, 1]cuda:0", primals_88: "f32[16384][1]cuda:0", primals_89: "f32[4096, 16384][16384, 1]cuda:0", primals_90: "f32[4096][1]cuda:0", primals_91: "f32[4096][1]cuda:0", primals_92: "f32[4096][1]cuda:0", primals_93: "f32[4096, 4096][4096, 1]cuda:0", primals_94: "f32[4096, 4096][4096, 1]cuda:0", primals_95: "f32[4096, 4096][4096, 1]cuda:0", primals_96: "f32[2048, 64][64, 1]cuda:0", primals_97: "f32[4096, 4096][4096, 1]cuda:0", primals_98: "f32[16384, 4096][4096, 1]cuda:0", primals_99: "f32[16384][1]cuda:0", primals_100: "f32[4096, 16384][16384, 1]cuda:0", primals_101: "f32[4096][1]cuda:0", primals_102: "f32[4096][1]cuda:0", primals_103: "f32[4096][1]cuda:0", primals_104: "f32[4096, 4096][4096, 1]cuda:0", primals_105: "f32[4096, 4096][4096, 1]cuda:0", primals_106: "f32[4096, 4096][4096, 1]cuda:0", primals_107: "f32[2048, 64][64, 1]cuda:0", primals_108: "f32[4096, 4096][4096, 1]cuda:0", primals_109: "f32[16384, 4096][4096, 1]cuda:0", primals_110: "f32[16384][1]cuda:0", primals_111: "f32[4096, 16384][16384, 1]cuda:0", primals_112: "f32[4096][1]cuda:0", primals_113: "f32[4096][1]cuda:0", primals_114: "f32[4096][1]cuda:0", primals_115: "f32[4096, 4096][4096, 1]cuda:0", primals_116: "f32[4096, 4096][4096, 1]cuda:0", primals_117: "f32[4096, 4096][4096, 1]cuda:0", primals_118: "f32[2048, 64][64, 1]cuda:0", primals_119: "f32[4096, 4096][4096, 1]cuda:0", primals_120: "f32[16384, 4096][4096, 1]cuda:0", primals_121: "f32[16384][1]cuda:0", primals_122: "f32[4096, 16384][16384, 1]cuda:0", primals_123: "f32[4096][1]cuda:0", primals_124: "f32[4096][1]cuda:0", primals_125: "f32[4096][1]cuda:0", primals_126: "f32[4096, 4096][4096, 1]cuda:0", primals_127: "f32[4096, 4096][4096, 1]cuda:0", primals_128: "f32[4096, 4096][4096, 1]cuda:0", primals_129: "f32[2048, 64][64, 1]cuda:0", primals_130: "f32[4096, 4096][4096, 1]cuda:0", primals_131: "f32[16384, 4096][4096, 1]cuda:0", primals_132: "f32[16384][1]cuda:0", primals_133: "f32[4096, 16384][16384, 1]cuda:0", primals_134: "f32[4096][1]cuda:0", primals_135: "f32[4096][1]cuda:0", primals_136: "f32[4096][1]cuda:0", primals_137: "f32[4096, 4096][4096, 1]cuda:0", primals_138: "f32[4096, 4096][4096, 1]cuda:0", primals_139: "f32[4096, 4096][4096, 1]cuda:0", primals_140: "f32[2048, 64][64, 1]cuda:0", primals_141: "f32[4096, 4096][4096, 1]cuda:0", primals_142: "f32[16384, 4096][4096, 1]cuda:0", primals_143: "f32[16384][1]cuda:0", primals_144: "f32[4096, 16384][16384, 1]cuda:0", primals_145: "f32[4096][1]cuda:0", primals_146: "f32[4096][1]cuda:0", primals_147: "f32[4096][1]cuda:0", primals_148: "f32[4096, 4096][4096, 1]cuda:0", primals_149: "f32[4096, 4096][4096, 1]cuda:0", primals_150: "f32[4096, 4096][4096, 1]cuda:0", primals_151: "f32[2048, 64][64, 1]cuda:0", primals_152: "f32[4096, 4096][4096, 1]cuda:0", primals_153: "f32[16384, 4096][4096, 1]cuda:0", primals_154: "f32[16384][1]cuda:0", primals_155: "f32[4096, 16384][16384, 1]cuda:0", primals_156: "f32[4096][1]cuda:0", primals_157: "f32[4096][1]cuda:0", primals_158: "f32[4096][1]cuda:0", primals_159: "f32[4096, 4096][4096, 1]cuda:0", primals_160: "f32[4096, 4096][4096, 1]cuda:0", primals_161: "f32[4096, 4096][4096, 1]cuda:0", primals_162: "f32[2048, 64][64, 1]cuda:0", primals_163: "f32[4096, 4096][4096, 1]cuda:0", primals_164: "f32[16384, 4096][4096, 1]cuda:0", primals_165: "f32[16384][1]cuda:0", primals_166: "f32[4096, 16384][16384, 1]cuda:0", primals_167: "f32[4096][1]cuda:0", primals_168: "f32[4096][1]cuda:0", primals_169: "f32[4096][1]cuda:0", primals_170: "f32[4096, 4096][4096, 1]cuda:0", primals_171: "f32[4096, 4096][4096, 1]cuda:0", primals_172: "f32[4096, 4096][4096, 1]cuda:0", primals_173: "f32[2048, 64][64, 1]cuda:0", primals_174: "f32[4096, 4096][4096, 1]cuda:0", primals_175: "f32[16384, 4096][4096, 1]cuda:0", primals_176: "f32[16384][1]cuda:0", primals_177: "f32[4096, 16384][16384, 1]cuda:0", primals_178: "f32[4096][1]cuda:0", primals_179: "f32[4096][1]cuda:0", primals_180: "f32[4096][1]cuda:0", primals_181: "f32[4096, 4096][4096, 1]cuda:0", primals_182: "f32[4096, 4096][4096, 1]cuda:0", primals_183: "f32[4096, 4096][4096, 1]cuda:0", primals_184: "f32[2048, 64][64, 1]cuda:0", primals_185: "f32[4096, 4096][4096, 1]cuda:0", primals_186: "f32[16384, 4096][4096, 1]cuda:0", primals_187: "f32[16384][1]cuda:0", primals_188: "f32[4096, 16384][16384, 1]cuda:0", primals_189: "f32[4096][1]cuda:0", primals_190: "f32[4096][1]cuda:0", primals_191: "f32[4096][1]cuda:0", primals_192: "f32[4096, 4096][4096, 1]cuda:0", primals_193: "f32[4096, 4096][4096, 1]cuda:0", primals_194: "f32[4096, 4096][4096, 1]cuda:0", primals_195: "f32[2048, 64][64, 1]cuda:0", primals_196: "f32[4096, 4096][4096, 1]cuda:0", primals_197: "f32[16384, 4096][4096, 1]cuda:0", primals_198: "f32[16384][1]cuda:0", primals_199: "f32[4096, 16384][16384, 1]cuda:0", primals_200: "f32[4096][1]cuda:0", primals_201: "f32[4096][1]cuda:0", primals_202: "f32[4096][1]cuda:0", primals_203: "f32[4096, 4096][4096, 1]cuda:0", primals_204: "f32[4096, 4096][4096, 1]cuda:0", primals_205: "f32[4096, 4096][4096, 1]cuda:0", primals_206: "f32[2048, 64][64, 1]cuda:0", primals_207: "f32[4096, 4096][4096, 1]cuda:0", primals_208: "f32[16384, 4096][4096, 1]cuda:0", primals_209: "f32[16384][1]cuda:0", primals_210: "f32[4096, 16384][16384, 1]cuda:0", primals_211: "f32[4096][1]cuda:0", primals_212: "f32[4096][1]cuda:0", primals_213: "f32[4096][1]cuda:0", primals_214: "f32[4096, 4096][4096, 1]cuda:0", primals_215: "f32[4096, 4096][4096, 1]cuda:0", primals_216: "f32[4096, 4096][4096, 1]cuda:0", primals_217: "f32[2048, 64][64, 1]cuda:0", primals_218: "f32[4096, 4096][4096, 1]cuda:0", primals_219: "f32[16384, 4096][4096, 1]cuda:0", primals_220: "f32[16384][1]cuda:0", primals_221: "f32[4096, 16384][16384, 1]cuda:0", primals_222: "f32[4096][1]cuda:0", primals_223: "f32[4096][1]cuda:0", primals_224: "f32[4096][1]cuda:0", primals_225: "f32[4096, 4096][4096, 1]cuda:0", primals_226: "f32[4096, 4096][4096, 1]cuda:0", primals_227: "f32[4096, 4096][4096, 1]cuda:0", primals_228: "f32[2048, 64][64, 1]cuda:0", primals_229: "f32[4096, 4096][4096, 1]cuda:0", primals_230: "f32[16384, 4096][4096, 1]cuda:0", primals_231: "f32[16384][1]cuda:0", primals_232: "f32[4096, 16384][16384, 1]cuda:0", primals_233: "f32[4096][1]cuda:0", primals_234: "f32[4096][1]cuda:0", primals_235: "f32[4096][1]cuda:0", primals_236: "f32[4096, 4096][4096, 1]cuda:0", primals_237: "f32[4096, 4096][4096, 1]cuda:0", primals_238: "f32[4096, 4096][4096, 1]cuda:0", primals_239: "f32[2048, 64][64, 1]cuda:0", primals_240: "f32[4096, 4096][4096, 1]cuda:0", primals_241: "f32[16384, 4096][4096, 1]cuda:0", primals_242: "f32[16384][1]cuda:0", primals_243: "f32[4096, 16384][16384, 1]cuda:0", primals_244: "f32[4096][1]cuda:0", primals_245: "f32[4096][1]cuda:0", primals_246: "f32[4096][1]cuda:0", primals_247: "f32[4096, 4096][4096, 1]cuda:0", primals_248: "f32[4096, 4096][4096, 1]cuda:0", primals_249: "f32[4096, 4096][4096, 1]cuda:0", primals_250: "f32[2048, 64][64, 1]cuda:0", primals_251: "f32[4096, 4096][4096, 1]cuda:0", primals_252: "f32[16384, 4096][4096, 1]cuda:0", primals_253: "f32[16384][1]cuda:0", primals_254: "f32[4096, 16384][16384, 1]cuda:0", primals_255: "f32[4096][1]cuda:0", primals_256: "f32[4096][1]cuda:0", primals_257: "f32[4096][1]cuda:0", primals_258: "f32[4096, 4096][4096, 1]cuda:0", primals_259: "f32[4096, 4096][4096, 1]cuda:0", primals_260: "f32[4096, 4096][4096, 1]cuda:0", primals_261: "f32[2048, 64][64, 1]cuda:0", primals_262: "f32[4096, 4096][4096, 1]cuda:0", primals_263: "f32[16384, 4096][4096, 1]cuda:0", primals_264: "f32[16384][1]cuda:0", primals_265: "f32[4096, 16384][16384, 1]cuda:0", primals_266: "f32[4096][1]cuda:0", primals_267: "f32[4096][1]cuda:0", primals_268: "f32[4096][1]cuda:0", primals_269: "f32[4096, 4096][4096, 1]cuda:0", primals_270: "f32[4096, 4096][4096, 1]cuda:0", primals_271: "f32[4096, 4096][4096, 1]cuda:0", primals_272: "f32[2048, 64][64, 1]cuda:0", primals_273: "f32[4096, 4096][4096, 1]cuda:0", primals_274: "f32[16384, 4096][4096, 1]cuda:0", primals_275: "f32[16384][1]cuda:0", primals_276: "f32[4096, 16384][16384, 1]cuda:0", primals_277: "f32[4096][1]cuda:0", primals_278: "f32[4096][1]cuda:0", primals_279: "f32[4096][1]cuda:0", primals_280: "f32[4096, 4096][4096, 1]cuda:0", primals_281: "f32[4096, 4096][4096, 1]cuda:0", primals_282: "f32[4096, 4096][4096, 1]cuda:0", primals_283: "f32[2048, 64][64, 1]cuda:0", primals_284: "f32[4096, 4096][4096, 1]cuda:0", primals_285: "f32[16384, 4096][4096, 1]cuda:0", primals_286: "f32[16384][1]cuda:0", primals_287: "f32[4096, 16384][16384, 1]cuda:0", primals_288: "f32[4096][1]cuda:0", primals_289: "f32[4096][1]cuda:0", primals_290: "f32[4096][1]cuda:0", primals_291: "f32[4096, 4096][4096, 1]cuda:0", primals_292: "f32[4096, 4096][4096, 1]cuda:0", primals_293: "f32[4096, 4096][4096, 1]cuda:0", primals_294: "f32[2048, 64][64, 1]cuda:0", primals_295: "f32[4096, 4096][4096, 1]cuda:0", primals_296: "f32[16384, 4096][4096, 1]cuda:0", primals_297: "f32[16384][1]cuda:0", primals_298: "f32[4096, 16384][16384, 1]cuda:0", primals_299: "f32[4096][1]cuda:0", primals_300: "f32[4096][1]cuda:0", primals_301: "f32[4096][1]cuda:0", primals_302: "f32[4096, 4096][4096, 1]cuda:0", primals_303: "f32[4096, 4096][4096, 1]cuda:0", primals_304: "f32[4096, 4096][4096, 1]cuda:0", primals_305: "f32[2048, 64][64, 1]cuda:0", primals_306: "f32[4096, 4096][4096, 1]cuda:0", primals_307: "f32[16384, 4096][4096, 1]cuda:0", primals_308: "f32[16384][1]cuda:0", primals_309: "f32[4096, 16384][16384, 1]cuda:0", primals_310: "f32[4096][1]cuda:0", primals_311: "f32[4096][1]cuda:0", primals_312: "f32[4096][1]cuda:0", primals_313: "f32[50400, 4096][4096, 1]cuda:0", primals_314: "f32[50400][1]cuda:0", primals_315: "i64[1, 128][128, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:494 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:502 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:503 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[1, 1][128, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[1, 129][129, 1]cuda:0" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = None
        slice_2: "i64[1, 128][129, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat, -1, 0, 128)
        slice_3: "i64[1, 128][129, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat, -1, 1, 129);  cat = None
        sub_1: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_5: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        unsqueeze_6: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(bitwise_and_1, [1, -1, 128, 128]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(embedding, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_3: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_2: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(embedding, getitem_1)
        mul: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_1: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_3);  mul = None
        add_4: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_1: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        permute: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [128, 4096]);  convert_element_type_1 = None
        mm: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [1, 128, 4096]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_4: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_1: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        mm_1: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view, permute_1)
        view_3: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [1, 128, 4096]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_8: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        permute_2: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        mm_2: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view, permute_2)
        view_5: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [1, 128, 4096]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_6: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [1, 128, 16, 256]);  view_1 = None
        view_7: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [1, 128, 16, 256]);  view_3 = None
        view_8: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [1, 128, 16, 256]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_3: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_8, [1, 1, 1]);  primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_10: "i64[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        repeat_1: "i64[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.repeat.default(unsqueeze_10, [1, 1, 64]);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat, 1, repeat_1);  repeat = None
        convert_element_type_12: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather, torch.bfloat16);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split = torch.ops.aten.split.Tensor(convert_element_type_12, 32, -1);  convert_element_type_12 = None
        getitem_2: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split[0]
        getitem_3: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split[1];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_4: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_7, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_5: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_7, 3, 64, 9223372036854775807);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_6: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_6, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_7: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_6, 3, 64, 9223372036854775807);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_11: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_2, 2);  getitem_2 = None
        unsqueeze_12: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 4);  unsqueeze_11 = None
        expand_1: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_12, [1, 128, 1, 32, 2])
        clone_1: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [1, 128, 1, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_13: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_3, 2);  getitem_3 = None
        unsqueeze_14: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 4);  unsqueeze_13 = None
        expand_2: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_14, [1, 128, 1, 32, 2])
        clone_2: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1, 128, 1, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_2: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_4, view_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_8: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_9: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_4, 3, 1, 9223372036854775807, 2);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_9);  slice_9 = None
        unsqueeze_15: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
        unsqueeze_16: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_8, 4);  slice_8 = None
        cat_1: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_15, unsqueeze_16], -1);  unsqueeze_15 = unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_11: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [1, 128, 16, 64]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_3: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_11, view_9);  view_11 = None
        add_5: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_4: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_6, view_10);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_10: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_11: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_6, 3, 1, 9223372036854775807, 2);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_1: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_11);  slice_11 = None
        unsqueeze_21: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_1, 4);  neg_1 = None
        unsqueeze_22: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_10, 4);  slice_10 = None
        cat_2: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_21, unsqueeze_22], -1);  unsqueeze_21 = unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_14: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [1, 128, 16, 64]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_5: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_14, view_9);  view_14 = view_9 = None
        add_6: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_3: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_5, slice_5], -1);  add_5 = slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_4: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_6, slice_7], -1);  add_6 = slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_4: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_5: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_4, [0, 2, 1, 3]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_14: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_6: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_14, [0, 1, 3, 2]);  convert_element_type_14 = None
        convert_element_type_15: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_6, torch.bfloat16);  permute_6 = None
        convert_element_type_default_55: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_5, torch.bfloat16);  permute_5 = None
        expand_5: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_55, [1, 16, 128, 256]);  convert_element_type_default_55 = None
        view_15: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_5, [16, 128, 256]);  expand_5 = None
        expand_6: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_15, [1, 16, 256, 128]);  convert_element_type_15 = None
        view_16: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_6, [16, 256, 128]);  expand_6 = None
        bmm: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_15, view_16)
        view_17: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [1, 16, 128, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_17, 16.0);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_7: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div, where);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_7, [-1], True)
        sub_3: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_7, amax);  add_7 = amax = None
        exp: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_1: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_19: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_7: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_19, [1, 16, 128, 128]);  convert_element_type_19 = None
        view_18: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_7, [16, 128, 128]);  expand_7 = None
        expand_8: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_3, [1, 16, 128, 256]);  permute_3 = None
        view_19: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_8, [16, 128, 256]);  expand_8 = None
        bmm_1: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_18, view_19)
        view_20: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [1, 16, 128, 256]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_7: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        clone_6: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_21: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [1, 128, 4096]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_22: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        permute_8: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_22, [1, 0]);  convert_element_type_22 = None
        view_22: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_21, [128, 4096]);  view_21 = None
        mm_3: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_8)
        view_23: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [1, 128, 4096]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_25: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_26: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        permute_9: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_26, [1, 0]);  convert_element_type_26 = None
        addmm: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_25, view, permute_9);  convert_element_type_25 = None
        view_25: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_25, 0.5)
        convert_element_type_31: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_25, torch.float32)
        pow_1: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_31, 3.0);  convert_element_type_31 = None
        mul_7: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_25, mul_7);  view_25 = mul_7 = None
        mul_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
        tanh: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_9: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_9: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, add_9);  mul_6 = add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_32: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_33: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_34: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_26: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_34, [128, 16384]);  convert_element_type_34 = None
        permute_10: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_33, [1, 0]);  convert_element_type_33 = None
        addmm_1: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_32, view_26, permute_10);  convert_element_type_32 = None
        view_27: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 4096]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_10: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_23, view_27);  view_23 = view_27 = None
        add_11: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_10, embedding)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_1[0]
        getitem_5: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_12: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_1: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_4: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_11, getitem_5)
        mul_10: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = None
        mul_11: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, primals_14);  mul_10 = None
        add_13: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, primals_15);  mul_11 = primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_38: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_39: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        permute_11: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_38, [1, 0]);  convert_element_type_38 = None
        view_28: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_39, [128, 4096]);  convert_element_type_39 = None
        mm_4: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_11)
        view_29: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [1, 128, 4096]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_42: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        permute_12: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_42, [1, 0]);  convert_element_type_42 = None
        mm_5: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_12)
        view_31: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [1, 128, 4096]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_46: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        permute_13: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_46, [1, 0]);  convert_element_type_46 = None
        mm_6: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_13)
        view_33: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [1, 128, 4096]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_34: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [1, 128, 16, 256]);  view_29 = None
        view_35: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_31, [1, 128, 16, 256]);  view_31 = None
        view_36: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_33, [1, 128, 16, 256]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_14: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_2: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_19, [1, 1, 1]);  primals_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_1: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_2, 1, repeat_1);  repeat_2 = None
        convert_element_type_50: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_1, torch.bfloat16);  gather_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_1 = torch.ops.aten.split.Tensor(convert_element_type_50, 32, -1);  convert_element_type_50 = None
        getitem_6: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_1[0]
        getitem_7: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_1[1];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_12: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_35, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_13: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_35, 3, 64, 9223372036854775807);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_14: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_34, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_15: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_34, 3, 64, 9223372036854775807);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_24: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_6, 2);  getitem_6 = None
        unsqueeze_25: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 4);  unsqueeze_24 = None
        expand_9: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_25, [1, 128, 1, 32, 2])
        clone_9: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_37: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [1, 128, 1, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_26: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_7, 2);  getitem_7 = None
        unsqueeze_27: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 4);  unsqueeze_26 = None
        expand_10: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_27, [1, 128, 1, 32, 2])
        clone_10: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_38: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1, 128, 1, 64]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_12: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_12, view_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_16: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_17: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_12, 3, 1, 9223372036854775807, 2);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_2: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_17);  slice_17 = None
        unsqueeze_28: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_2, 4);  neg_2 = None
        unsqueeze_29: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_16, 4);  slice_16 = None
        cat_5: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_28, unsqueeze_29], -1);  unsqueeze_28 = unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_39: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [1, 128, 16, 64]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_13: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_39, view_37);  view_39 = None
        add_14: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        mul_14: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_14, view_38);  view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_18: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_19: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_14, 3, 1, 9223372036854775807, 2);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_3: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_19);  slice_19 = None
        unsqueeze_34: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_3, 4);  neg_3 = None
        unsqueeze_35: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_18, 4);  slice_18 = None
        cat_6: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_34, unsqueeze_35], -1);  unsqueeze_34 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_42: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [1, 128, 16, 64]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_15: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_42, view_37);  view_42 = view_37 = None
        add_15: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_7: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_14, slice_13], -1);  add_14 = slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_8: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_15, slice_15], -1);  add_15 = slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_15: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_7, [0, 2, 1, 3]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_16: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_8, [0, 2, 1, 3]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_52: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_15, torch.float32);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_17: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_52, [0, 1, 3, 2]);  convert_element_type_52 = None
        convert_element_type_53: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_17, torch.bfloat16);  permute_17 = None
        convert_element_type_default_54: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_16, torch.bfloat16);  permute_16 = None
        expand_13: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_54, [1, 16, 128, 256]);  convert_element_type_default_54 = None
        view_43: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_13, [16, 128, 256]);  expand_13 = None
        expand_14: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_53, [1, 16, 256, 128]);  convert_element_type_53 = None
        view_44: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_14, [16, 256, 128]);  expand_14 = None
        bmm_2: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_43, view_44)
        view_45: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [1, 16, 128, 128]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_2: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_45, 16.0);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_16: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_2, where);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_1: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_16, [-1], True)
        sub_5: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_16, amax_1);  add_16 = amax_1 = None
        exp_1: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_57: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_15: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_57, [1, 16, 128, 128]);  convert_element_type_57 = None
        view_46: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_15, [16, 128, 128]);  expand_15 = None
        expand_16: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_14, [1, 16, 128, 256]);  permute_14 = None
        view_47: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_16, [16, 128, 256]);  expand_16 = None
        bmm_3: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_46, view_47)
        view_48: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [1, 16, 128, 256]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_18: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        clone_14: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_49: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [1, 128, 4096]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_60: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        permute_19: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_60, [1, 0]);  convert_element_type_60 = None
        view_50: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_49, [128, 4096]);  view_49 = None
        mm_7: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_50, permute_19)
        view_51: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [1, 128, 4096]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_63: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convert_element_type_64: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        permute_20: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        addmm_2: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_63, view_28, permute_20);  convert_element_type_63 = None
        view_53: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_16: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_53, 0.5)
        convert_element_type_69: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_53, torch.float32)
        pow_2: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 3.0);  convert_element_type_69 = None
        mul_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_53, mul_17);  view_53 = mul_17 = None
        mul_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_19: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, add_18);  mul_16 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_70: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convert_element_type_71: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convert_element_type_72: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        view_54: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_72, [128, 16384]);  convert_element_type_72 = None
        permute_21: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        addmm_3: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_70, view_54, permute_21);  convert_element_type_70 = None
        view_55: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 4096]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_19: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_51, view_55);  view_51 = view_55 = None
        add_20: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_19, add_11);  add_19 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_2[0]
        getitem_9: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_21: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_6: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_20, getitem_9);  getitem_9 = None
        mul_20: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = None
        mul_21: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, primals_25)
        add_22: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, primals_26);  mul_21 = primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_76: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        convert_element_type_77: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None
        permute_22: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_76, [1, 0]);  convert_element_type_76 = None
        view_56: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_77, [128, 4096]);  convert_element_type_77 = None
        mm_8: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_56, permute_22)
        view_57: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [1, 128, 4096]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_80: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        permute_23: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        mm_9: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_56, permute_23)
        view_59: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [1, 128, 4096]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_84: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        permute_24: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        mm_10: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_56, permute_24)
        view_61: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [1, 128, 4096]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_62: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_57, [1, 128, 16, 256]);  view_57 = None
        view_63: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [1, 128, 16, 256]);  view_59 = None
        view_64: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_61, [1, 128, 16, 256]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_25: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_4: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_30, [1, 1, 1]);  primals_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_2: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_4, 1, repeat_1);  repeat_4 = None
        convert_element_type_88: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_2, torch.bfloat16);  gather_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_2 = torch.ops.aten.split.Tensor(convert_element_type_88, 32, -1);  convert_element_type_88 = None
        getitem_10: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_2[0]
        getitem_11: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_2[1];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_20: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_63, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_21: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_63, 3, 64, 9223372036854775807);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_22: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_62, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_23: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_62, 3, 64, 9223372036854775807);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_37: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_10, 2);  getitem_10 = None
        unsqueeze_38: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 4);  unsqueeze_37 = None
        expand_17: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_38, [1, 128, 1, 32, 2])
        clone_17: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_65: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [1, 128, 1, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_39: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_11, 2);  getitem_11 = None
        unsqueeze_40: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 4);  unsqueeze_39 = None
        expand_18: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_40, [1, 128, 1, 32, 2])
        clone_18: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_66: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [1, 128, 1, 64]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_22: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_20, view_66)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_24: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_25: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_20, 3, 1, 9223372036854775807, 2);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_4: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_25);  slice_25 = None
        unsqueeze_41: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_4, 4);  neg_4 = None
        unsqueeze_42: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_24, 4);  slice_24 = None
        cat_9: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_41, unsqueeze_42], -1);  unsqueeze_41 = unsqueeze_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_67: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [1, 128, 16, 64]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_23: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_67, view_65);  view_67 = None
        add_23: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        mul_24: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_22, view_66);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_26: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_27: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_22, 3, 1, 9223372036854775807, 2);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_5: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_27);  slice_27 = None
        unsqueeze_47: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_5, 4);  neg_5 = None
        unsqueeze_48: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_26, 4);  slice_26 = None
        cat_10: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_47, unsqueeze_48], -1);  unsqueeze_47 = unsqueeze_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_70: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [1, 128, 16, 64]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_25: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_70, view_65);  view_70 = view_65 = None
        add_24: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_11: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_23, slice_21], -1);  add_23 = slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_12: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_24, slice_23], -1);  add_24 = slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_26: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_11, [0, 2, 1, 3]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_27: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_12, [0, 2, 1, 3]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_90: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_26, torch.float32);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_28: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_90, [0, 1, 3, 2]);  convert_element_type_90 = None
        convert_element_type_91: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_28, torch.bfloat16);  permute_28 = None
        convert_element_type_default_53: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_27, torch.bfloat16);  permute_27 = None
        expand_21: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_53, [1, 16, 128, 256]);  convert_element_type_default_53 = None
        view_71: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_21, [16, 128, 256]);  expand_21 = None
        expand_22: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_91, [1, 16, 256, 128]);  convert_element_type_91 = None
        view_72: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_22, [16, 256, 128]);  expand_22 = None
        bmm_4: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_71, view_72)
        view_73: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [1, 16, 128, 128]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_4: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_73, 16.0);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_25: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_4, where);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_2: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_25, [-1], True)
        sub_7: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_25, amax_2);  add_25 = amax_2 = None
        exp_2: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_95: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_23: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_95, [1, 16, 128, 128]);  convert_element_type_95 = None
        view_74: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_23, [16, 128, 128]);  expand_23 = None
        expand_24: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_25, [1, 16, 128, 256]);  permute_25 = None
        view_75: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_24, [16, 128, 256]);  expand_24 = None
        bmm_5: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_74, view_75)
        view_76: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [1, 16, 128, 256]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_29: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        clone_22: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_77: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [1, 128, 4096]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_98: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        permute_30: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_98, [1, 0]);  convert_element_type_98 = None
        view_78: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_77, [128, 4096]);  view_77 = None
        mm_11: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_78, permute_30)
        view_79: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [1, 128, 4096]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_101: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        convert_element_type_102: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        permute_31: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_102, [1, 0]);  convert_element_type_102 = None
        addmm_4: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_101, view_56, permute_31);  convert_element_type_101 = None
        view_81: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_26: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_81, 0.5)
        convert_element_type_107: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_81, torch.float32)
        pow_3: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_107, 3.0);  convert_element_type_107 = None
        mul_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_26: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_81, mul_27);  view_81 = mul_27 = None
        mul_28: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_26, 0.7978845608028654);  add_26 = None
        tanh_2: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_29: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, add_27);  mul_26 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_108: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_109: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        convert_element_type_110: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        view_82: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_110, [128, 16384]);  convert_element_type_110 = None
        permute_32: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        addmm_5: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_108, view_82, permute_32);  convert_element_type_108 = None
        view_83: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 4096]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_28: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_79, view_83);  view_79 = view_83 = None
        add_29: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_28, add_20);  add_28 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_3[0]
        getitem_13: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_30: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_3: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_8: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_29, getitem_13);  getitem_13 = None
        mul_30: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = None
        mul_31: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, primals_36)
        add_31: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, primals_37);  mul_31 = primals_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_114: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convert_element_type_115: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        permute_33: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        view_84: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_115, [128, 4096]);  convert_element_type_115 = None
        mm_12: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_84, permute_33)
        view_85: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [1, 128, 4096]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_118: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        permute_34: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_118, [1, 0]);  convert_element_type_118 = None
        mm_13: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_84, permute_34)
        view_87: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [1, 128, 4096]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_122: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        permute_35: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_122, [1, 0]);  convert_element_type_122 = None
        mm_14: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_84, permute_35)
        view_89: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [1, 128, 4096]);  mm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_90: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_85, [1, 128, 16, 256]);  view_85 = None
        view_91: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_87, [1, 128, 16, 256]);  view_87 = None
        view_92: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [1, 128, 16, 256]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_36: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_6: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_41, [1, 1, 1]);  primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_3: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_6, 1, repeat_1);  repeat_6 = None
        convert_element_type_126: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_3, torch.bfloat16);  gather_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_3 = torch.ops.aten.split.Tensor(convert_element_type_126, 32, -1);  convert_element_type_126 = None
        getitem_14: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_3[0]
        getitem_15: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_3[1];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_28: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_91, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_29: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_91, 3, 64, 9223372036854775807);  view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_30: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_90, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_31: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_90, 3, 64, 9223372036854775807);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_50: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_14, 2);  getitem_14 = None
        unsqueeze_51: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, 4);  unsqueeze_50 = None
        expand_25: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_51, [1, 128, 1, 32, 2])
        clone_25: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_93: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [1, 128, 1, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_52: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_15, 2);  getitem_15 = None
        unsqueeze_53: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 4);  unsqueeze_52 = None
        expand_26: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_53, [1, 128, 1, 32, 2])
        clone_26: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_94: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [1, 128, 1, 64]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_32: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_28, view_94)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_32: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_28, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_33: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_28, 3, 1, 9223372036854775807, 2);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_6: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_33);  slice_33 = None
        unsqueeze_54: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_6, 4);  neg_6 = None
        unsqueeze_55: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_32, 4);  slice_32 = None
        cat_13: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_54, unsqueeze_55], -1);  unsqueeze_54 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_95: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_13, [1, 128, 16, 64]);  cat_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_33: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_95, view_93);  view_95 = None
        add_32: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        mul_34: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_30, view_94);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_34: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_30, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_35: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_30, 3, 1, 9223372036854775807, 2);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_7: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_35);  slice_35 = None
        unsqueeze_60: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_7, 4);  neg_7 = None
        unsqueeze_61: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_34, 4);  slice_34 = None
        cat_14: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_60, unsqueeze_61], -1);  unsqueeze_60 = unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_98: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_14, [1, 128, 16, 64]);  cat_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_35: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_98, view_93);  view_98 = view_93 = None
        add_33: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_15: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_32, slice_29], -1);  add_32 = slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_16: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_33, slice_31], -1);  add_33 = slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_37: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_15, [0, 2, 1, 3]);  cat_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_38: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_16, [0, 2, 1, 3]);  cat_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_128: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_37, torch.float32);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_39: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_128, [0, 1, 3, 2]);  convert_element_type_128 = None
        convert_element_type_129: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_39, torch.bfloat16);  permute_39 = None
        convert_element_type_default_52: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_38, torch.bfloat16);  permute_38 = None
        expand_29: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_52, [1, 16, 128, 256]);  convert_element_type_default_52 = None
        view_99: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_29, [16, 128, 256]);  expand_29 = None
        expand_30: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_129, [1, 16, 256, 128]);  convert_element_type_129 = None
        view_100: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_30, [16, 256, 128]);  expand_30 = None
        bmm_6: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_99, view_100)
        view_101: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [1, 16, 128, 128]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_6: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_101, 16.0);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_34: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_6, where);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_3: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_34, [-1], True)
        sub_9: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_34, amax_3);  add_34 = amax_3 = None
        exp_3: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_4: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_133: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_31: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_133, [1, 16, 128, 128]);  convert_element_type_133 = None
        view_102: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_31, [16, 128, 128]);  expand_31 = None
        expand_32: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_36, [1, 16, 128, 256]);  permute_36 = None
        view_103: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_32, [16, 128, 256]);  expand_32 = None
        bmm_7: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_102, view_103)
        view_104: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [1, 16, 128, 256]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_40: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_30: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_105: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [1, 128, 4096]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_136: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        permute_41: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_136, [1, 0]);  convert_element_type_136 = None
        view_106: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_105, [128, 4096]);  view_105 = None
        mm_15: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_106, permute_41)
        view_107: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [1, 128, 4096]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_139: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_140: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        permute_42: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_140, [1, 0]);  convert_element_type_140 = None
        addmm_6: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_139, view_84, permute_42);  convert_element_type_139 = None
        view_109: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        convert_element_type_145: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_109, torch.float32)
        pow_4: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_145, 3.0);  convert_element_type_145 = None
        mul_37: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_35: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_109, mul_37);  view_109 = mul_37 = None
        mul_38: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_35, 0.7978845608028654);  add_35 = None
        tanh_3: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_36: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_39: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, add_36);  mul_36 = add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_146: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convert_element_type_147: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_148: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        view_110: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_148, [128, 16384]);  convert_element_type_148 = None
        permute_43: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_147, [1, 0]);  convert_element_type_147 = None
        addmm_7: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_146, view_110, permute_43);  convert_element_type_146 = None
        view_111: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 4096]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_37: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_107, view_111);  view_107 = view_111 = None
        add_38: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_37, add_29);  add_37 = add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_4[0]
        getitem_17: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_39: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_10: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_38, getitem_17);  getitem_17 = None
        mul_40: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_4);  sub_10 = None
        mul_41: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, primals_47)
        add_40: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, primals_48);  mul_41 = primals_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_152: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        convert_element_type_153: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None
        permute_44: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_152, [1, 0]);  convert_element_type_152 = None
        view_112: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_153, [128, 4096]);  convert_element_type_153 = None
        mm_16: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_112, permute_44)
        view_113: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [1, 128, 4096]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_156: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        permute_45: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_156, [1, 0]);  convert_element_type_156 = None
        mm_17: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_112, permute_45)
        view_115: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [1, 128, 4096]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_160: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        permute_46: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        mm_18: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_112, permute_46)
        view_117: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [1, 128, 4096]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_118: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [1, 128, 16, 256]);  view_113 = None
        view_119: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_115, [1, 128, 16, 256]);  view_115 = None
        view_120: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_117, [1, 128, 16, 256]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_47: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_8: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_52, [1, 1, 1]);  primals_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_4: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_8, 1, repeat_1);  repeat_8 = None
        convert_element_type_164: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_4, torch.bfloat16);  gather_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_4 = torch.ops.aten.split.Tensor(convert_element_type_164, 32, -1);  convert_element_type_164 = None
        getitem_18: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_4[0]
        getitem_19: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_4[1];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_36: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_119, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_37: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_119, 3, 64, 9223372036854775807);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_38: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_118, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_39: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_118, 3, 64, 9223372036854775807);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_63: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_18, 2);  getitem_18 = None
        unsqueeze_64: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 4);  unsqueeze_63 = None
        expand_33: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_64, [1, 128, 1, 32, 2])
        clone_33: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_121: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [1, 128, 1, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_65: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_19, 2);  getitem_19 = None
        unsqueeze_66: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 4);  unsqueeze_65 = None
        expand_34: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_66, [1, 128, 1, 32, 2])
        clone_34: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_122: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [1, 128, 1, 64]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_42: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_36, view_122)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_40: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_36, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_41: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_36, 3, 1, 9223372036854775807, 2);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_8: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_41);  slice_41 = None
        unsqueeze_67: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_8, 4);  neg_8 = None
        unsqueeze_68: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_40, 4);  slice_40 = None
        cat_17: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_67, unsqueeze_68], -1);  unsqueeze_67 = unsqueeze_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_123: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_17, [1, 128, 16, 64]);  cat_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_43: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_123, view_121);  view_123 = None
        add_41: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
        mul_44: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_38, view_122);  view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_42: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_43: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_38, 3, 1, 9223372036854775807, 2);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_9: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_43);  slice_43 = None
        unsqueeze_73: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_9, 4);  neg_9 = None
        unsqueeze_74: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_42, 4);  slice_42 = None
        cat_18: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_73, unsqueeze_74], -1);  unsqueeze_73 = unsqueeze_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_126: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_18, [1, 128, 16, 64]);  cat_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_45: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_126, view_121);  view_126 = view_121 = None
        add_42: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_19: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_41, slice_37], -1);  add_41 = slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_20: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_42, slice_39], -1);  add_42 = slice_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_48: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_19, [0, 2, 1, 3]);  cat_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_49: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_20, [0, 2, 1, 3]);  cat_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_166: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_48, torch.float32);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_50: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_166, [0, 1, 3, 2]);  convert_element_type_166 = None
        convert_element_type_167: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_50, torch.bfloat16);  permute_50 = None
        convert_element_type_default_51: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_49, torch.bfloat16);  permute_49 = None
        expand_37: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_51, [1, 16, 128, 256]);  convert_element_type_default_51 = None
        view_127: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_37, [16, 128, 256]);  expand_37 = None
        expand_38: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_167, [1, 16, 256, 128]);  convert_element_type_167 = None
        view_128: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_38, [16, 256, 128]);  expand_38 = None
        bmm_8: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_127, view_128)
        view_129: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [1, 16, 128, 128]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_8: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_129, 16.0);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_43: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_8, where);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_4: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_43, [-1], True)
        sub_11: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_43, amax_4);  add_43 = amax_4 = None
        exp_4: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_5: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_171: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_39: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_171, [1, 16, 128, 128]);  convert_element_type_171 = None
        view_130: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_39, [16, 128, 128]);  expand_39 = None
        expand_40: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_47, [1, 16, 128, 256]);  permute_47 = None
        view_131: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_40, [16, 128, 256]);  expand_40 = None
        bmm_9: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_130, view_131)
        view_132: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [1, 16, 128, 256]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_51: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        clone_38: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_133: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [1, 128, 4096]);  clone_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_174: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        permute_52: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        view_134: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [128, 4096]);  view_133 = None
        mm_19: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_134, permute_52)
        view_135: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [1, 128, 4096]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_177: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convert_element_type_178: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        permute_53: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_178, [1, 0]);  convert_element_type_178 = None
        addmm_8: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_177, view_112, permute_53);  convert_element_type_177 = None
        view_137: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_137, 0.5)
        convert_element_type_183: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_137, torch.float32)
        pow_5: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 3.0);  convert_element_type_183 = None
        mul_47: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_44: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_137, mul_47);  view_137 = mul_47 = None
        mul_48: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
        tanh_4: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_45: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_49: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, add_45);  mul_46 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_184: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_185: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convert_element_type_186: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        view_138: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_186, [128, 16384]);  convert_element_type_186 = None
        permute_54: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        addmm_9: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_184, view_138, permute_54);  convert_element_type_184 = None
        view_139: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 4096]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_46: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_135, view_139);  view_135 = view_139 = None
        add_47: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_46, add_38);  add_46 = add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_5[0]
        getitem_21: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_48: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_5: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_12: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_47, getitem_21);  getitem_21 = None
        mul_50: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_5);  sub_12 = None
        mul_51: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, primals_58)
        add_49: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, primals_59);  mul_51 = primals_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_190: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convert_element_type_191: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None
        permute_55: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        view_140: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_191, [128, 4096]);  convert_element_type_191 = None
        mm_20: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_140, permute_55)
        view_141: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [1, 128, 4096]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_194: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        permute_56: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_194, [1, 0]);  convert_element_type_194 = None
        mm_21: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_140, permute_56)
        view_143: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [1, 128, 4096]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_198: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        permute_57: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_198, [1, 0]);  convert_element_type_198 = None
        mm_22: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_140, permute_57)
        view_145: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [1, 128, 4096]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_146: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_141, [1, 128, 16, 256]);  view_141 = None
        view_147: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_143, [1, 128, 16, 256]);  view_143 = None
        view_148: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_145, [1, 128, 16, 256]);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_58: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_10: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_63, [1, 1, 1]);  primals_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_5: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_10, 1, repeat_1);  repeat_10 = None
        convert_element_type_202: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_5, torch.bfloat16);  gather_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_5 = torch.ops.aten.split.Tensor(convert_element_type_202, 32, -1);  convert_element_type_202 = None
        getitem_22: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_5[0]
        getitem_23: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_5[1];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_44: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_147, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_45: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_147, 3, 64, 9223372036854775807);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_46: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_146, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_47: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_146, 3, 64, 9223372036854775807);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_76: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_22, 2);  getitem_22 = None
        unsqueeze_77: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 4);  unsqueeze_76 = None
        expand_41: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_77, [1, 128, 1, 32, 2])
        clone_41: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_149: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [1, 128, 1, 64]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_78: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_23, 2);  getitem_23 = None
        unsqueeze_79: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 4);  unsqueeze_78 = None
        expand_42: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_79, [1, 128, 1, 32, 2])
        clone_42: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_150: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [1, 128, 1, 64]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_52: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_44, view_150)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_48: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_44, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_49: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_44, 3, 1, 9223372036854775807, 2);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_10: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_49);  slice_49 = None
        unsqueeze_80: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_10, 4);  neg_10 = None
        unsqueeze_81: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_48, 4);  slice_48 = None
        cat_21: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_80, unsqueeze_81], -1);  unsqueeze_80 = unsqueeze_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_151: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_21, [1, 128, 16, 64]);  cat_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_53: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_151, view_149);  view_151 = None
        add_50: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
        mul_54: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_46, view_150);  view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_50: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_51: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_46, 3, 1, 9223372036854775807, 2);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_11: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_51);  slice_51 = None
        unsqueeze_86: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_11, 4);  neg_11 = None
        unsqueeze_87: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_50, 4);  slice_50 = None
        cat_22: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_86, unsqueeze_87], -1);  unsqueeze_86 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_154: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_22, [1, 128, 16, 64]);  cat_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_55: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_154, view_149);  view_154 = view_149 = None
        add_51: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_23: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_50, slice_45], -1);  add_50 = slice_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_24: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_51, slice_47], -1);  add_51 = slice_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_59: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_23, [0, 2, 1, 3]);  cat_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_60: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_24, [0, 2, 1, 3]);  cat_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_204: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_59, torch.float32);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_61: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_204, [0, 1, 3, 2]);  convert_element_type_204 = None
        convert_element_type_205: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_61, torch.bfloat16);  permute_61 = None
        convert_element_type_default_50: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_60, torch.bfloat16);  permute_60 = None
        expand_45: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_50, [1, 16, 128, 256]);  convert_element_type_default_50 = None
        view_155: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_45, [16, 128, 256]);  expand_45 = None
        expand_46: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_205, [1, 16, 256, 128]);  convert_element_type_205 = None
        view_156: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_46, [16, 256, 128]);  expand_46 = None
        bmm_10: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_155, view_156)
        view_157: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [1, 16, 128, 128]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_10: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_157, 16.0);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_52: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_10, where);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_5: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_52, [-1], True)
        sub_13: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_52, amax_5);  add_52 = amax_5 = None
        exp_5: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_6: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_209: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_47: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_209, [1, 16, 128, 128]);  convert_element_type_209 = None
        view_158: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_47, [16, 128, 128]);  expand_47 = None
        expand_48: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_58, [1, 16, 128, 256]);  permute_58 = None
        view_159: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_48, [16, 128, 256]);  expand_48 = None
        bmm_11: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_158, view_159)
        view_160: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [1, 16, 128, 256]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_62: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        clone_46: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_161: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [1, 128, 4096]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_212: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        permute_63: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_212, [1, 0]);  convert_element_type_212 = None
        view_162: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [128, 4096]);  view_161 = None
        mm_23: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_162, permute_63)
        view_163: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [1, 128, 4096]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_215: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        convert_element_type_216: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        permute_64: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_216, [1, 0]);  convert_element_type_216 = None
        addmm_10: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_215, view_140, permute_64);  convert_element_type_215 = None
        view_165: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_165, 0.5)
        convert_element_type_221: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_165, torch.float32)
        pow_6: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_221, 3.0);  convert_element_type_221 = None
        mul_57: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_53: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_165, mul_57);  view_165 = mul_57 = None
        mul_58: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_5: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_54: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_59: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, add_54);  mul_56 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_222: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convert_element_type_223: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_224: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        view_166: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_224, [128, 16384]);  convert_element_type_224 = None
        permute_65: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        addmm_11: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_222, view_166, permute_65);  convert_element_type_222 = None
        view_167: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 4096]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_55: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_163, view_167);  view_163 = view_167 = None
        add_56: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_55, add_47);  add_55 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_6[0]
        getitem_25: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_57: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_14: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_56, getitem_25);  getitem_25 = None
        mul_60: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_6);  sub_14 = None
        mul_61: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, primals_69)
        add_58: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, primals_70);  mul_61 = primals_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_228: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        convert_element_type_229: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.bfloat16);  add_58 = None
        permute_66: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_228, [1, 0]);  convert_element_type_228 = None
        view_168: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_229, [128, 4096]);  convert_element_type_229 = None
        mm_24: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_168, permute_66)
        view_169: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [1, 128, 4096]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_232: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        permute_67: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_232, [1, 0]);  convert_element_type_232 = None
        mm_25: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_168, permute_67)
        view_171: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [1, 128, 4096]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_236: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        permute_68: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        mm_26: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_168, permute_68)
        view_173: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [1, 128, 4096]);  mm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_174: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [1, 128, 16, 256]);  view_169 = None
        view_175: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_171, [1, 128, 16, 256]);  view_171 = None
        view_176: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_173, [1, 128, 16, 256]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_69: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_12: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_74, [1, 1, 1]);  primals_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_6: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_12, 1, repeat_1);  repeat_12 = None
        convert_element_type_240: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_6, torch.bfloat16);  gather_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_6 = torch.ops.aten.split.Tensor(convert_element_type_240, 32, -1);  convert_element_type_240 = None
        getitem_26: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_6[0]
        getitem_27: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_6[1];  split_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_52: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_175, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_53: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_175, 3, 64, 9223372036854775807);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_54: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_174, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_55: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_174, 3, 64, 9223372036854775807);  view_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_89: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_26, 2);  getitem_26 = None
        unsqueeze_90: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 4);  unsqueeze_89 = None
        expand_49: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_90, [1, 128, 1, 32, 2])
        clone_49: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_177: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [1, 128, 1, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_91: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_27, 2);  getitem_27 = None
        unsqueeze_92: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 4);  unsqueeze_91 = None
        expand_50: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_92, [1, 128, 1, 32, 2])
        clone_50: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_178: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [1, 128, 1, 64]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_62: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_52, view_178)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_56: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_52, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_57: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_52, 3, 1, 9223372036854775807, 2);  slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_12: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_57);  slice_57 = None
        unsqueeze_93: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_12, 4);  neg_12 = None
        unsqueeze_94: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_56, 4);  slice_56 = None
        cat_25: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_93, unsqueeze_94], -1);  unsqueeze_93 = unsqueeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_179: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_25, [1, 128, 16, 64]);  cat_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_63: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_179, view_177);  view_179 = None
        add_59: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
        mul_64: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_54, view_178);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_58: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_54, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_59: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_54, 3, 1, 9223372036854775807, 2);  slice_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_13: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_59);  slice_59 = None
        unsqueeze_99: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_13, 4);  neg_13 = None
        unsqueeze_100: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_58, 4);  slice_58 = None
        cat_26: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_99, unsqueeze_100], -1);  unsqueeze_99 = unsqueeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_182: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_26, [1, 128, 16, 64]);  cat_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_65: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_182, view_177);  view_182 = view_177 = None
        add_60: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_27: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_59, slice_53], -1);  add_59 = slice_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_28: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_60, slice_55], -1);  add_60 = slice_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_70: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_27, [0, 2, 1, 3]);  cat_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_71: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_28, [0, 2, 1, 3]);  cat_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_242: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_70, torch.float32);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_72: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_242, [0, 1, 3, 2]);  convert_element_type_242 = None
        convert_element_type_243: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_72, torch.bfloat16);  permute_72 = None
        convert_element_type_default_49: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_71, torch.bfloat16);  permute_71 = None
        expand_53: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_49, [1, 16, 128, 256]);  convert_element_type_default_49 = None
        view_183: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_53, [16, 128, 256]);  expand_53 = None
        expand_54: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_243, [1, 16, 256, 128]);  convert_element_type_243 = None
        view_184: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_54, [16, 256, 128]);  expand_54 = None
        bmm_12: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_183, view_184)
        view_185: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [1, 16, 128, 128]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_12: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_185, 16.0);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_61: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_12, where);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_6: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_61, [-1], True)
        sub_15: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_61, amax_6);  add_61 = amax_6 = None
        exp_6: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_7: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_247: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_55: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_247, [1, 16, 128, 128]);  convert_element_type_247 = None
        view_186: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_55, [16, 128, 128]);  expand_55 = None
        expand_56: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_69, [1, 16, 128, 256]);  permute_69 = None
        view_187: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_56, [16, 128, 256]);  expand_56 = None
        bmm_13: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_186, view_187)
        view_188: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [1, 16, 128, 256]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_73: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        clone_54: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_189: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [1, 128, 4096]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_250: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        permute_74: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_250, [1, 0]);  convert_element_type_250 = None
        view_190: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_189, [128, 4096]);  view_189 = None
        mm_27: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_190, permute_74)
        view_191: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [1, 128, 4096]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_253: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        convert_element_type_254: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        permute_75: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_254, [1, 0]);  convert_element_type_254 = None
        addmm_12: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_253, view_168, permute_75);  convert_element_type_253 = None
        view_193: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_66: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_193, 0.5)
        convert_element_type_259: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_193, torch.float32)
        pow_7: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_259, 3.0);  convert_element_type_259 = None
        mul_67: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_62: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, mul_67);  view_193 = mul_67 = None
        mul_68: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_6: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_63: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_69: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, add_63);  mul_66 = add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_260: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_261: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        convert_element_type_262: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        view_194: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_262, [128, 16384]);  convert_element_type_262 = None
        permute_76: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_261, [1, 0]);  convert_element_type_261 = None
        addmm_13: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_260, view_194, permute_76);  convert_element_type_260 = None
        view_195: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 4096]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_64: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_191, view_195);  view_191 = view_195 = None
        add_65: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_64, add_56);  add_64 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_7[0]
        getitem_29: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_66: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_7: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_16: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_65, getitem_29);  getitem_29 = None
        mul_70: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_7);  sub_16 = None
        mul_71: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, primals_80)
        add_67: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, primals_81);  mul_71 = primals_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_266: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        convert_element_type_267: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None
        permute_77: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_266, [1, 0]);  convert_element_type_266 = None
        view_196: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_267, [128, 4096]);  convert_element_type_267 = None
        mm_28: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_196, permute_77)
        view_197: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [1, 128, 4096]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_270: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        permute_78: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        mm_29: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_196, permute_78)
        view_199: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [1, 128, 4096]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_274: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        permute_79: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_274, [1, 0]);  convert_element_type_274 = None
        mm_30: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_196, permute_79)
        view_201: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [1, 128, 4096]);  mm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_202: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_197, [1, 128, 16, 256]);  view_197 = None
        view_203: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [1, 128, 16, 256]);  view_199 = None
        view_204: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_201, [1, 128, 16, 256]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_80: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_14: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_85, [1, 1, 1]);  primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_7: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_14, 1, repeat_1);  repeat_14 = None
        convert_element_type_278: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_7, torch.bfloat16);  gather_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_7 = torch.ops.aten.split.Tensor(convert_element_type_278, 32, -1);  convert_element_type_278 = None
        getitem_30: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_7[0]
        getitem_31: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_7[1];  split_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_60: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_203, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_61: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_203, 3, 64, 9223372036854775807);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_62: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_202, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_63: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_202, 3, 64, 9223372036854775807);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_102: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_30, 2);  getitem_30 = None
        unsqueeze_103: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 4);  unsqueeze_102 = None
        expand_57: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_103, [1, 128, 1, 32, 2])
        clone_57: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_205: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [1, 128, 1, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_104: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_31, 2);  getitem_31 = None
        unsqueeze_105: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, 4);  unsqueeze_104 = None
        expand_58: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_105, [1, 128, 1, 32, 2])
        clone_58: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_206: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [1, 128, 1, 64]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_72: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_60, view_206)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_64: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_60, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_65: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_60, 3, 1, 9223372036854775807, 2);  slice_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_14: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_65);  slice_65 = None
        unsqueeze_106: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_14, 4);  neg_14 = None
        unsqueeze_107: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_64, 4);  slice_64 = None
        cat_29: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_106, unsqueeze_107], -1);  unsqueeze_106 = unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_207: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_29, [1, 128, 16, 64]);  cat_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_73: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_207, view_205);  view_207 = None
        add_68: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
        mul_74: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_62, view_206);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_66: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_62, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_67: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_62, 3, 1, 9223372036854775807, 2);  slice_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_15: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_67);  slice_67 = None
        unsqueeze_112: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_15, 4);  neg_15 = None
        unsqueeze_113: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_66, 4);  slice_66 = None
        cat_30: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_112, unsqueeze_113], -1);  unsqueeze_112 = unsqueeze_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_210: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_30, [1, 128, 16, 64]);  cat_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_75: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_210, view_205);  view_210 = view_205 = None
        add_69: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_31: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_68, slice_61], -1);  add_68 = slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_32: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_69, slice_63], -1);  add_69 = slice_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_81: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_31, [0, 2, 1, 3]);  cat_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_82: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_32, [0, 2, 1, 3]);  cat_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_280: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_81, torch.float32);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_83: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_280, [0, 1, 3, 2]);  convert_element_type_280 = None
        convert_element_type_281: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_83, torch.bfloat16);  permute_83 = None
        convert_element_type_default_48: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_82, torch.bfloat16);  permute_82 = None
        expand_61: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_48, [1, 16, 128, 256]);  convert_element_type_default_48 = None
        view_211: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_61, [16, 128, 256]);  expand_61 = None
        expand_62: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_281, [1, 16, 256, 128]);  convert_element_type_281 = None
        view_212: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_62, [16, 256, 128]);  expand_62 = None
        bmm_14: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_211, view_212)
        view_213: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [1, 16, 128, 128]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_14: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_213, 16.0);  view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_70: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_14, where);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_7: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_70, [-1], True)
        sub_17: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_70, amax_7);  add_70 = amax_7 = None
        exp_7: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_8: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_285: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_63: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_285, [1, 16, 128, 128]);  convert_element_type_285 = None
        view_214: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_63, [16, 128, 128]);  expand_63 = None
        expand_64: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_80, [1, 16, 128, 256]);  permute_80 = None
        view_215: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_64, [16, 128, 256]);  expand_64 = None
        bmm_15: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_214, view_215)
        view_216: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [1, 16, 128, 256]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_84: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
        clone_62: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_217: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_62, [1, 128, 4096]);  clone_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_288: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        permute_85: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_288, [1, 0]);  convert_element_type_288 = None
        view_218: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_217, [128, 4096]);  view_217 = None
        mm_31: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_218, permute_85)
        view_219: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [1, 128, 4096]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_291: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        convert_element_type_292: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        permute_86: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_292, [1, 0]);  convert_element_type_292 = None
        addmm_14: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_291, view_196, permute_86);  convert_element_type_291 = None
        view_221: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_221, 0.5)
        convert_element_type_297: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_221, torch.float32)
        pow_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_297, 3.0);  convert_element_type_297 = None
        mul_77: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_71: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_221, mul_77);  view_221 = mul_77 = None
        mul_78: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_7: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_72: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_79: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, add_72);  mul_76 = add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_298: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        convert_element_type_299: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        convert_element_type_300: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        view_222: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_300, [128, 16384]);  convert_element_type_300 = None
        permute_87: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_299, [1, 0]);  convert_element_type_299 = None
        addmm_15: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_298, view_222, permute_87);  convert_element_type_298 = None
        view_223: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [1, 128, 4096]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_73: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_219, view_223);  view_219 = view_223 = None
        add_74: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_73, add_65);  add_73 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_8[0]
        getitem_33: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_75: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_8: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_18: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_74, getitem_33);  getitem_33 = None
        mul_80: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_8);  sub_18 = None
        mul_81: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, primals_91)
        add_76: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, primals_92);  mul_81 = primals_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_304: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convert_element_type_305: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None
        permute_88: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_304, [1, 0]);  convert_element_type_304 = None
        view_224: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_305, [128, 4096]);  convert_element_type_305 = None
        mm_32: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_224, permute_88)
        view_225: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [1, 128, 4096]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_308: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        permute_89: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_308, [1, 0]);  convert_element_type_308 = None
        mm_33: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_224, permute_89)
        view_227: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [1, 128, 4096]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_312: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        permute_90: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_312, [1, 0]);  convert_element_type_312 = None
        mm_34: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_224, permute_90)
        view_229: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [1, 128, 4096]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_230: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_225, [1, 128, 16, 256]);  view_225 = None
        view_231: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_227, [1, 128, 16, 256]);  view_227 = None
        view_232: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_229, [1, 128, 16, 256]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_91: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_16: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_96, [1, 1, 1]);  primals_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_8: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_16, 1, repeat_1);  repeat_16 = None
        convert_element_type_316: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_8, torch.bfloat16);  gather_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_8 = torch.ops.aten.split.Tensor(convert_element_type_316, 32, -1);  convert_element_type_316 = None
        getitem_34: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_8[0]
        getitem_35: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_8[1];  split_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_68: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_231, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_69: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_231, 3, 64, 9223372036854775807);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_70: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_230, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_71: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_230, 3, 64, 9223372036854775807);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_115: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_34, 2);  getitem_34 = None
        unsqueeze_116: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 4);  unsqueeze_115 = None
        expand_65: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_116, [1, 128, 1, 32, 2])
        clone_65: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_233: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [1, 128, 1, 64]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_117: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_35, 2);  getitem_35 = None
        unsqueeze_118: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 4);  unsqueeze_117 = None
        expand_66: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_118, [1, 128, 1, 32, 2])
        clone_66: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_234: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [1, 128, 1, 64]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_82: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_68, view_234)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_72: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_68, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_73: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_68, 3, 1, 9223372036854775807, 2);  slice_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_16: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_73);  slice_73 = None
        unsqueeze_119: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_16, 4);  neg_16 = None
        unsqueeze_120: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_72, 4);  slice_72 = None
        cat_33: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_119, unsqueeze_120], -1);  unsqueeze_119 = unsqueeze_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_235: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_33, [1, 128, 16, 64]);  cat_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_83: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_235, view_233);  view_235 = None
        add_77: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
        mul_84: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_70, view_234);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_74: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_70, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_75: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_70, 3, 1, 9223372036854775807, 2);  slice_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_17: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_75);  slice_75 = None
        unsqueeze_125: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_17, 4);  neg_17 = None
        unsqueeze_126: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_74, 4);  slice_74 = None
        cat_34: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_125, unsqueeze_126], -1);  unsqueeze_125 = unsqueeze_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_238: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_34, [1, 128, 16, 64]);  cat_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_85: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_238, view_233);  view_238 = view_233 = None
        add_78: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_35: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_77, slice_69], -1);  add_77 = slice_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_36: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_78, slice_71], -1);  add_78 = slice_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_92: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_35, [0, 2, 1, 3]);  cat_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_93: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_36, [0, 2, 1, 3]);  cat_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_318: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_92, torch.float32);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_94: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_318, [0, 1, 3, 2]);  convert_element_type_318 = None
        convert_element_type_319: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_94, torch.bfloat16);  permute_94 = None
        convert_element_type_default_47: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_93, torch.bfloat16);  permute_93 = None
        expand_69: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_47, [1, 16, 128, 256]);  convert_element_type_default_47 = None
        view_239: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_69, [16, 128, 256]);  expand_69 = None
        expand_70: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_319, [1, 16, 256, 128]);  convert_element_type_319 = None
        view_240: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_70, [16, 256, 128]);  expand_70 = None
        bmm_16: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_239, view_240)
        view_241: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [1, 16, 128, 128]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_16: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_241, 16.0);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_79: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_16, where);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_8: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_79, [-1], True)
        sub_19: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_79, amax_8);  add_79 = amax_8 = None
        exp_8: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_9: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_323: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_71: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_323, [1, 16, 128, 128]);  convert_element_type_323 = None
        view_242: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_71, [16, 128, 128]);  expand_71 = None
        expand_72: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_91, [1, 16, 128, 256]);  permute_91 = None
        view_243: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_72, [16, 128, 256]);  expand_72 = None
        bmm_17: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_242, view_243)
        view_244: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [1, 16, 128, 256]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_95: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        clone_70: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_245: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [1, 128, 4096]);  clone_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_326: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        permute_96: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_326, [1, 0]);  convert_element_type_326 = None
        view_246: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [128, 4096]);  view_245 = None
        mm_35: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_246, permute_96)
        view_247: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [1, 128, 4096]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_329: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convert_element_type_330: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        permute_97: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_330, [1, 0]);  convert_element_type_330 = None
        addmm_16: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_329, view_224, permute_97);  convert_element_type_329 = None
        view_249: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_86: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_249, 0.5)
        convert_element_type_335: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_249, torch.float32)
        pow_9: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_335, 3.0);  convert_element_type_335 = None
        mul_87: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_80: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_249, mul_87);  view_249 = mul_87 = None
        mul_88: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        tanh_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_81: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_89: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, add_81);  mul_86 = add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_336: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        convert_element_type_337: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        convert_element_type_338: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        view_250: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_338, [128, 16384]);  convert_element_type_338 = None
        permute_98: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_337, [1, 0]);  convert_element_type_337 = None
        addmm_17: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_336, view_250, permute_98);  convert_element_type_336 = None
        view_251: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 4096]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_82: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_247, view_251);  view_247 = view_251 = None
        add_83: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_82, add_74);  add_82 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_9[0]
        getitem_37: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_84: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_9: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_20: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_83, getitem_37);  getitem_37 = None
        mul_90: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_9);  sub_20 = None
        mul_91: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, primals_102)
        add_85: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, primals_103);  mul_91 = primals_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_342: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convert_element_type_343: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None
        permute_99: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_342, [1, 0]);  convert_element_type_342 = None
        view_252: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_343, [128, 4096]);  convert_element_type_343 = None
        mm_36: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_252, permute_99)
        view_253: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [1, 128, 4096]);  mm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_346: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        permute_100: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_346, [1, 0]);  convert_element_type_346 = None
        mm_37: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_252, permute_100)
        view_255: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [1, 128, 4096]);  mm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_350: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        permute_101: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_350, [1, 0]);  convert_element_type_350 = None
        mm_38: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_252, permute_101)
        view_257: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [1, 128, 4096]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_258: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_253, [1, 128, 16, 256]);  view_253 = None
        view_259: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_255, [1, 128, 16, 256]);  view_255 = None
        view_260: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [1, 128, 16, 256]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_102: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_18: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_107, [1, 1, 1]);  primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_9: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_18, 1, repeat_1);  repeat_18 = None
        convert_element_type_354: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_9, torch.bfloat16);  gather_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_9 = torch.ops.aten.split.Tensor(convert_element_type_354, 32, -1);  convert_element_type_354 = None
        getitem_38: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_9[0]
        getitem_39: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_9[1];  split_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_76: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_259, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_77: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_259, 3, 64, 9223372036854775807);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_78: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_258, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_79: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_258, 3, 64, 9223372036854775807);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_128: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_38, 2);  getitem_38 = None
        unsqueeze_129: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 4);  unsqueeze_128 = None
        expand_73: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_129, [1, 128, 1, 32, 2])
        clone_73: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_261: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [1, 128, 1, 64]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_130: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_39, 2);  getitem_39 = None
        unsqueeze_131: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 4);  unsqueeze_130 = None
        expand_74: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_131, [1, 128, 1, 32, 2])
        clone_74: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_262: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [1, 128, 1, 64]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_92: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_76, view_262)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_80: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_76, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_81: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_76, 3, 1, 9223372036854775807, 2);  slice_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_18: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_81);  slice_81 = None
        unsqueeze_132: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_18, 4);  neg_18 = None
        unsqueeze_133: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_80, 4);  slice_80 = None
        cat_37: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_132, unsqueeze_133], -1);  unsqueeze_132 = unsqueeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_263: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_37, [1, 128, 16, 64]);  cat_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_93: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_263, view_261);  view_263 = None
        add_86: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        mul_94: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_78, view_262);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_82: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_78, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_83: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_78, 3, 1, 9223372036854775807, 2);  slice_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_19: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_83);  slice_83 = None
        unsqueeze_138: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_19, 4);  neg_19 = None
        unsqueeze_139: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_82, 4);  slice_82 = None
        cat_38: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_138, unsqueeze_139], -1);  unsqueeze_138 = unsqueeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_266: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_38, [1, 128, 16, 64]);  cat_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_95: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_266, view_261);  view_266 = view_261 = None
        add_87: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_39: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_86, slice_77], -1);  add_86 = slice_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_40: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_87, slice_79], -1);  add_87 = slice_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_103: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_39, [0, 2, 1, 3]);  cat_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_104: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_40, [0, 2, 1, 3]);  cat_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_356: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_103, torch.float32);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_105: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_356, [0, 1, 3, 2]);  convert_element_type_356 = None
        convert_element_type_357: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_105, torch.bfloat16);  permute_105 = None
        convert_element_type_default_46: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_104, torch.bfloat16);  permute_104 = None
        expand_77: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_46, [1, 16, 128, 256]);  convert_element_type_default_46 = None
        view_267: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_77, [16, 128, 256]);  expand_77 = None
        expand_78: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_357, [1, 16, 256, 128]);  convert_element_type_357 = None
        view_268: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_78, [16, 256, 128]);  expand_78 = None
        bmm_18: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_267, view_268)
        view_269: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [1, 16, 128, 128]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_18: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_269, 16.0);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_88: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_18, where);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_9: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_88, [-1], True)
        sub_21: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_88, amax_9);  add_88 = amax_9 = None
        exp_9: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_10: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_361: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_79: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_361, [1, 16, 128, 128]);  convert_element_type_361 = None
        view_270: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_79, [16, 128, 128]);  expand_79 = None
        expand_80: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_102, [1, 16, 128, 256]);  permute_102 = None
        view_271: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_80, [16, 128, 256]);  expand_80 = None
        bmm_19: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_270, view_271)
        view_272: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [1, 16, 128, 256]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_106: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        clone_78: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_273: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [1, 128, 4096]);  clone_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_364: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        permute_107: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_364, [1, 0]);  convert_element_type_364 = None
        view_274: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_273, [128, 4096]);  view_273 = None
        mm_39: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_274, permute_107)
        view_275: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [1, 128, 4096]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_367: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convert_element_type_368: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        permute_108: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_368, [1, 0]);  convert_element_type_368 = None
        addmm_18: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_367, view_252, permute_108);  convert_element_type_367 = None
        view_277: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_96: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_277, 0.5)
        convert_element_type_373: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_277, torch.float32)
        pow_10: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_373, 3.0);  convert_element_type_373 = None
        mul_97: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_89: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_277, mul_97);  view_277 = mul_97 = None
        mul_98: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_9: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_90: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_99: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, add_90);  mul_96 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_374: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        convert_element_type_375: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        convert_element_type_376: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None
        view_278: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_376, [128, 16384]);  convert_element_type_376 = None
        permute_109: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_375, [1, 0]);  convert_element_type_375 = None
        addmm_19: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_374, view_278, permute_109);  convert_element_type_374 = None
        view_279: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 4096]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_91: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_275, view_279);  view_275 = view_279 = None
        add_92: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_91, add_83);  add_91 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_10[0]
        getitem_41: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_93: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_10: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_22: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_92, getitem_41);  getitem_41 = None
        mul_100: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_10);  sub_22 = None
        mul_101: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, primals_113)
        add_94: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, primals_114);  mul_101 = primals_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_380: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convert_element_type_381: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None
        permute_110: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_380, [1, 0]);  convert_element_type_380 = None
        view_280: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_381, [128, 4096]);  convert_element_type_381 = None
        mm_40: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_280, permute_110)
        view_281: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [1, 128, 4096]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_384: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        permute_111: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_384, [1, 0]);  convert_element_type_384 = None
        mm_41: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_280, permute_111)
        view_283: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [1, 128, 4096]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_388: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        permute_112: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_388, [1, 0]);  convert_element_type_388 = None
        mm_42: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_280, permute_112)
        view_285: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [1, 128, 4096]);  mm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_286: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_281, [1, 128, 16, 256]);  view_281 = None
        view_287: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_283, [1, 128, 16, 256]);  view_283 = None
        view_288: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_285, [1, 128, 16, 256]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_113: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_20: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_118, [1, 1, 1]);  primals_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_10: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_20, 1, repeat_1);  repeat_20 = None
        convert_element_type_392: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_10, torch.bfloat16);  gather_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_10 = torch.ops.aten.split.Tensor(convert_element_type_392, 32, -1);  convert_element_type_392 = None
        getitem_42: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_10[0]
        getitem_43: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_10[1];  split_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_84: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_287, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_85: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_287, 3, 64, 9223372036854775807);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_86: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_286, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_87: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_286, 3, 64, 9223372036854775807);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_141: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_42, 2);  getitem_42 = None
        unsqueeze_142: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 4);  unsqueeze_141 = None
        expand_81: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_142, [1, 128, 1, 32, 2])
        clone_81: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_289: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [1, 128, 1, 64]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_143: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_43, 2);  getitem_43 = None
        unsqueeze_144: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 4);  unsqueeze_143 = None
        expand_82: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_144, [1, 128, 1, 32, 2])
        clone_82: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_290: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [1, 128, 1, 64]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_102: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_84, view_290)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_88: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_84, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_89: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_84, 3, 1, 9223372036854775807, 2);  slice_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_20: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_89);  slice_89 = None
        unsqueeze_145: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_20, 4);  neg_20 = None
        unsqueeze_146: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_88, 4);  slice_88 = None
        cat_41: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_145, unsqueeze_146], -1);  unsqueeze_145 = unsqueeze_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_291: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_41, [1, 128, 16, 64]);  cat_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_103: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_291, view_289);  view_291 = None
        add_95: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        mul_104: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_86, view_290);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_90: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_86, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_91: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_86, 3, 1, 9223372036854775807, 2);  slice_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_21: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_91);  slice_91 = None
        unsqueeze_151: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_21, 4);  neg_21 = None
        unsqueeze_152: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_90, 4);  slice_90 = None
        cat_42: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_151, unsqueeze_152], -1);  unsqueeze_151 = unsqueeze_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_294: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_42, [1, 128, 16, 64]);  cat_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_105: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_294, view_289);  view_294 = view_289 = None
        add_96: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_43: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_95, slice_85], -1);  add_95 = slice_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_44: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_96, slice_87], -1);  add_96 = slice_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_114: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_43, [0, 2, 1, 3]);  cat_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_115: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_44, [0, 2, 1, 3]);  cat_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_394: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_114, torch.float32);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_116: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_394, [0, 1, 3, 2]);  convert_element_type_394 = None
        convert_element_type_395: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_116, torch.bfloat16);  permute_116 = None
        convert_element_type_default_45: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_115, torch.bfloat16);  permute_115 = None
        expand_85: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_45, [1, 16, 128, 256]);  convert_element_type_default_45 = None
        view_295: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_85, [16, 128, 256]);  expand_85 = None
        expand_86: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_395, [1, 16, 256, 128]);  convert_element_type_395 = None
        view_296: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_86, [16, 256, 128]);  expand_86 = None
        bmm_20: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_295, view_296)
        view_297: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [1, 16, 128, 128]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_20: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_297, 16.0);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_97: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_20, where);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_10: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_97, [-1], True)
        sub_23: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_97, amax_10);  add_97 = amax_10 = None
        exp_10: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_11: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_399: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_87: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_399, [1, 16, 128, 128]);  convert_element_type_399 = None
        view_298: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_87, [16, 128, 128]);  expand_87 = None
        expand_88: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_113, [1, 16, 128, 256]);  permute_113 = None
        view_299: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_88, [16, 128, 256]);  expand_88 = None
        bmm_21: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_298, view_299)
        view_300: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [1, 16, 128, 256]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_117: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_86: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_301: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [1, 128, 4096]);  clone_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_402: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        permute_118: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_402, [1, 0]);  convert_element_type_402 = None
        view_302: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_301, [128, 4096]);  view_301 = None
        mm_43: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_302, permute_118)
        view_303: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [1, 128, 4096]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_405: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_406: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        permute_119: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_406, [1, 0]);  convert_element_type_406 = None
        addmm_20: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_405, view_280, permute_119);  convert_element_type_405 = None
        view_305: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_106: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        convert_element_type_411: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32)
        pow_11: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_411, 3.0);  convert_element_type_411 = None
        mul_107: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_98: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_305, mul_107);  view_305 = mul_107 = None
        mul_108: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, 0.7978845608028654);  add_98 = None
        tanh_10: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_99: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_109: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, add_99);  mul_106 = add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_412: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        convert_element_type_413: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convert_element_type_414: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        view_306: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_414, [128, 16384]);  convert_element_type_414 = None
        permute_120: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_413, [1, 0]);  convert_element_type_413 = None
        addmm_21: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_412, view_306, permute_120);  convert_element_type_412 = None
        view_307: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [1, 128, 4096]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_100: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_303, view_307);  view_303 = view_307 = None
        add_101: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_100, add_92);  add_100 = add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_11[0]
        getitem_45: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_102: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_11: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_24: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_101, getitem_45);  getitem_45 = None
        mul_110: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_11);  sub_24 = None
        mul_111: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, primals_124)
        add_103: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, primals_125);  mul_111 = primals_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_418: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        convert_element_type_419: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None
        permute_121: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_418, [1, 0]);  convert_element_type_418 = None
        view_308: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_419, [128, 4096]);  convert_element_type_419 = None
        mm_44: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_308, permute_121)
        view_309: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [1, 128, 4096]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_422: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        permute_122: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_422, [1, 0]);  convert_element_type_422 = None
        mm_45: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_308, permute_122)
        view_311: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [1, 128, 4096]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_426: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        permute_123: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_426, [1, 0]);  convert_element_type_426 = None
        mm_46: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_308, permute_123)
        view_313: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [1, 128, 4096]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_314: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_309, [1, 128, 16, 256]);  view_309 = None
        view_315: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_311, [1, 128, 16, 256]);  view_311 = None
        view_316: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_313, [1, 128, 16, 256]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_124: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_22: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_129, [1, 1, 1]);  primals_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_11: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_22, 1, repeat_1);  repeat_22 = None
        convert_element_type_430: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_11, torch.bfloat16);  gather_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_11 = torch.ops.aten.split.Tensor(convert_element_type_430, 32, -1);  convert_element_type_430 = None
        getitem_46: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_11[0]
        getitem_47: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_11[1];  split_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_92: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_315, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_93: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_315, 3, 64, 9223372036854775807);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_94: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_314, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_95: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_314, 3, 64, 9223372036854775807);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_154: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_46, 2);  getitem_46 = None
        unsqueeze_155: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 4);  unsqueeze_154 = None
        expand_89: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_155, [1, 128, 1, 32, 2])
        clone_89: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_317: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [1, 128, 1, 64]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_156: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_47, 2);  getitem_47 = None
        unsqueeze_157: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 4);  unsqueeze_156 = None
        expand_90: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_157, [1, 128, 1, 32, 2])
        clone_90: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_318: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [1, 128, 1, 64]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_112: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_92, view_318)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_96: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_92, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_97: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_92, 3, 1, 9223372036854775807, 2);  slice_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_22: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_97);  slice_97 = None
        unsqueeze_158: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_22, 4);  neg_22 = None
        unsqueeze_159: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_96, 4);  slice_96 = None
        cat_45: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_158, unsqueeze_159], -1);  unsqueeze_158 = unsqueeze_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_319: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_45, [1, 128, 16, 64]);  cat_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_113: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_319, view_317);  view_319 = None
        add_104: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
        mul_114: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_94, view_318);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_98: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_94, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_99: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_94, 3, 1, 9223372036854775807, 2);  slice_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_23: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_99);  slice_99 = None
        unsqueeze_164: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_23, 4);  neg_23 = None
        unsqueeze_165: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_98, 4);  slice_98 = None
        cat_46: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_164, unsqueeze_165], -1);  unsqueeze_164 = unsqueeze_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_322: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_46, [1, 128, 16, 64]);  cat_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_115: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_322, view_317);  view_322 = view_317 = None
        add_105: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_47: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_104, slice_93], -1);  add_104 = slice_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_48: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_105, slice_95], -1);  add_105 = slice_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_125: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_47, [0, 2, 1, 3]);  cat_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_126: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_48, [0, 2, 1, 3]);  cat_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_432: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_125, torch.float32);  permute_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_127: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_432, [0, 1, 3, 2]);  convert_element_type_432 = None
        convert_element_type_433: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_127, torch.bfloat16);  permute_127 = None
        convert_element_type_default_44: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_126, torch.bfloat16);  permute_126 = None
        expand_93: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_44, [1, 16, 128, 256]);  convert_element_type_default_44 = None
        view_323: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_93, [16, 128, 256]);  expand_93 = None
        expand_94: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_433, [1, 16, 256, 128]);  convert_element_type_433 = None
        view_324: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_94, [16, 256, 128]);  expand_94 = None
        bmm_22: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_323, view_324)
        view_325: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [1, 16, 128, 128]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_22: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_325, 16.0);  view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_106: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_22, where);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_11: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_106, [-1], True)
        sub_25: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_106, amax_11);  add_106 = amax_11 = None
        exp_11: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_12: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_437: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_95: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_437, [1, 16, 128, 128]);  convert_element_type_437 = None
        view_326: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_95, [16, 128, 128]);  expand_95 = None
        expand_96: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_124, [1, 16, 128, 256]);  permute_124 = None
        view_327: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_96, [16, 128, 256]);  expand_96 = None
        bmm_23: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_326, view_327)
        view_328: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [1, 16, 128, 256]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_128: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        clone_94: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_329: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_94, [1, 128, 4096]);  clone_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_440: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        permute_129: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_440, [1, 0]);  convert_element_type_440 = None
        view_330: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [128, 4096]);  view_329 = None
        mm_47: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_330, permute_129)
        view_331: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [1, 128, 4096]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_443: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        convert_element_type_444: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        permute_130: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_444, [1, 0]);  convert_element_type_444 = None
        addmm_22: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_443, view_308, permute_130);  convert_element_type_443 = None
        view_333: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_333, 0.5)
        convert_element_type_449: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_333, torch.float32)
        pow_12: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_449, 3.0);  convert_element_type_449 = None
        mul_117: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_107: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_333, mul_117);  view_333 = mul_117 = None
        mul_118: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_107, 0.7978845608028654);  add_107 = None
        tanh_11: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_108: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, add_108);  mul_116 = add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_450: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convert_element_type_451: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        convert_element_type_452: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None
        view_334: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_452, [128, 16384]);  convert_element_type_452 = None
        permute_131: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_451, [1, 0]);  convert_element_type_451 = None
        addmm_23: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_450, view_334, permute_131);  convert_element_type_450 = None
        view_335: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 4096]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_109: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_331, view_335);  view_331 = view_335 = None
        add_110: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, add_101);  add_109 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_12[0]
        getitem_49: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_111: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_12: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_26: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_110, getitem_49);  getitem_49 = None
        mul_120: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_12);  sub_26 = None
        mul_121: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, primals_135)
        add_112: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, primals_136);  mul_121 = primals_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_456: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_457: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None
        permute_132: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_456, [1, 0]);  convert_element_type_456 = None
        view_336: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [128, 4096]);  convert_element_type_457 = None
        mm_48: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_132)
        view_337: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [1, 128, 4096]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_460: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        permute_133: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_460, [1, 0]);  convert_element_type_460 = None
        mm_49: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_133)
        view_339: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_49, [1, 128, 4096]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_464: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        permute_134: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_464, [1, 0]);  convert_element_type_464 = None
        mm_50: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_134)
        view_341: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [1, 128, 4096]);  mm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_342: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_337, [1, 128, 16, 256]);  view_337 = None
        view_343: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_339, [1, 128, 16, 256]);  view_339 = None
        view_344: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_341, [1, 128, 16, 256]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_135: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_24: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_140, [1, 1, 1]);  primals_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_12: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_24, 1, repeat_1);  repeat_24 = None
        convert_element_type_468: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_12, torch.bfloat16);  gather_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_12 = torch.ops.aten.split.Tensor(convert_element_type_468, 32, -1);  convert_element_type_468 = None
        getitem_50: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_12[0]
        getitem_51: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_12[1];  split_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_100: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_343, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_101: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_343, 3, 64, 9223372036854775807);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_102: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_342, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_103: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_342, 3, 64, 9223372036854775807);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_167: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_50, 2);  getitem_50 = None
        unsqueeze_168: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 4);  unsqueeze_167 = None
        expand_97: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_168, [1, 128, 1, 32, 2])
        clone_97: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_345: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [1, 128, 1, 64]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_169: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_51, 2);  getitem_51 = None
        unsqueeze_170: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 4);  unsqueeze_169 = None
        expand_98: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_170, [1, 128, 1, 32, 2])
        clone_98: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_346: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [1, 128, 1, 64]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_122: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_100, view_346)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_104: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_100, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_105: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_100, 3, 1, 9223372036854775807, 2);  slice_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_24: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_105);  slice_105 = None
        unsqueeze_171: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_24, 4);  neg_24 = None
        unsqueeze_172: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_104, 4);  slice_104 = None
        cat_49: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_171, unsqueeze_172], -1);  unsqueeze_171 = unsqueeze_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_347: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_49, [1, 128, 16, 64]);  cat_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_123: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_347, view_345);  view_347 = None
        add_113: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
        mul_124: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_102, view_346);  view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_106: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_102, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_107: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_102, 3, 1, 9223372036854775807, 2);  slice_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_25: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_107);  slice_107 = None
        unsqueeze_177: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_25, 4);  neg_25 = None
        unsqueeze_178: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_106, 4);  slice_106 = None
        cat_50: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_177, unsqueeze_178], -1);  unsqueeze_177 = unsqueeze_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_350: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_50, [1, 128, 16, 64]);  cat_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_125: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_350, view_345);  view_350 = view_345 = None
        add_114: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_51: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_113, slice_101], -1);  add_113 = slice_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_52: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_114, slice_103], -1);  add_114 = slice_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_136: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_51, [0, 2, 1, 3]);  cat_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_137: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_52, [0, 2, 1, 3]);  cat_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_470: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_136, torch.float32);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_138: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_470, [0, 1, 3, 2]);  convert_element_type_470 = None
        convert_element_type_471: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_138, torch.bfloat16);  permute_138 = None
        convert_element_type_default_43: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_137, torch.bfloat16);  permute_137 = None
        expand_101: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_43, [1, 16, 128, 256]);  convert_element_type_default_43 = None
        view_351: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_101, [16, 128, 256]);  expand_101 = None
        expand_102: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_471, [1, 16, 256, 128]);  convert_element_type_471 = None
        view_352: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_102, [16, 256, 128]);  expand_102 = None
        bmm_24: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_351, view_352)
        view_353: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_24: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_353, 16.0);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_115: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_24, where);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_12: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_115, [-1], True)
        sub_27: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_115, amax_12);  add_115 = amax_12 = None
        exp_12: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_13: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_25: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_475: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_103: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_475, [1, 16, 128, 128]);  convert_element_type_475 = None
        view_354: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_103, [16, 128, 128]);  expand_103 = None
        expand_104: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_135, [1, 16, 128, 256]);  permute_135 = None
        view_355: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_104, [16, 128, 256]);  expand_104 = None
        bmm_25: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_354, view_355)
        view_356: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [1, 16, 128, 256]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_139: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        clone_102: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_357: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [1, 128, 4096]);  clone_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_478: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        permute_140: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_478, [1, 0]);  convert_element_type_478 = None
        view_358: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_357, [128, 4096]);  view_357 = None
        mm_51: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_358, permute_140)
        view_359: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_51, [1, 128, 4096]);  mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_481: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        convert_element_type_482: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        permute_141: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_482, [1, 0]);  convert_element_type_482 = None
        addmm_24: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_481, view_336, permute_141);  convert_element_type_481 = None
        view_361: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_126: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_361, 0.5)
        convert_element_type_487: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_361, torch.float32)
        pow_13: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_487, 3.0);  convert_element_type_487 = None
        mul_127: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_116: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_361, mul_127);  view_361 = mul_127 = None
        mul_128: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, 0.7978845608028654);  add_116 = None
        tanh_12: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_128);  mul_128 = None
        add_117: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_129: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, add_117);  mul_126 = add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_488: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16);  primals_145 = None
        convert_element_type_489: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        convert_element_type_490: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        view_362: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_490, [128, 16384]);  convert_element_type_490 = None
        permute_142: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_489, [1, 0]);  convert_element_type_489 = None
        addmm_25: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_488, view_362, permute_142);  convert_element_type_488 = None
        view_363: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 4096]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_118: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_359, view_363);  view_359 = view_363 = None
        add_119: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, add_110);  add_118 = add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_13[0]
        getitem_53: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_120: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_13: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        sub_28: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_119, getitem_53);  getitem_53 = None
        mul_130: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_13);  sub_28 = None
        mul_131: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, primals_146)
        add_121: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, primals_147);  mul_131 = primals_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_494: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convert_element_type_495: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None
        permute_143: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_494, [1, 0]);  convert_element_type_494 = None
        view_364: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_495, [128, 4096]);  convert_element_type_495 = None
        mm_52: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_364, permute_143)
        view_365: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [1, 128, 4096]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_498: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        permute_144: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_498, [1, 0]);  convert_element_type_498 = None
        mm_53: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_364, permute_144)
        view_367: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [1, 128, 4096]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_502: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        permute_145: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_502, [1, 0]);  convert_element_type_502 = None
        mm_54: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_364, permute_145)
        view_369: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [1, 128, 4096]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_370: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [1, 128, 16, 256]);  view_365 = None
        view_371: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_367, [1, 128, 16, 256]);  view_367 = None
        view_372: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_369, [1, 128, 16, 256]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_146: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_26: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_151, [1, 1, 1]);  primals_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_13: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_26, 1, repeat_1);  repeat_26 = None
        convert_element_type_506: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_13, torch.bfloat16);  gather_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_13 = torch.ops.aten.split.Tensor(convert_element_type_506, 32, -1);  convert_element_type_506 = None
        getitem_54: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_13[0]
        getitem_55: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_13[1];  split_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_108: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_371, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_109: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_371, 3, 64, 9223372036854775807);  view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_110: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_370, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_111: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_370, 3, 64, 9223372036854775807);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_180: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_54, 2);  getitem_54 = None
        unsqueeze_181: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 4);  unsqueeze_180 = None
        expand_105: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_181, [1, 128, 1, 32, 2])
        clone_105: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_373: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [1, 128, 1, 64]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_182: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_55, 2);  getitem_55 = None
        unsqueeze_183: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 4);  unsqueeze_182 = None
        expand_106: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_183, [1, 128, 1, 32, 2])
        clone_106: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_374: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [1, 128, 1, 64]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_132: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_108, view_374)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_112: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_108, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_113: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_108, 3, 1, 9223372036854775807, 2);  slice_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_26: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_113);  slice_113 = None
        unsqueeze_184: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_26, 4);  neg_26 = None
        unsqueeze_185: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_112, 4);  slice_112 = None
        cat_53: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_184, unsqueeze_185], -1);  unsqueeze_184 = unsqueeze_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_375: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_53, [1, 128, 16, 64]);  cat_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_133: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_375, view_373);  view_375 = None
        add_122: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
        mul_134: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_110, view_374);  view_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_114: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_110, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_115: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_110, 3, 1, 9223372036854775807, 2);  slice_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_27: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_115);  slice_115 = None
        unsqueeze_190: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_27, 4);  neg_27 = None
        unsqueeze_191: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_114, 4);  slice_114 = None
        cat_54: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_190, unsqueeze_191], -1);  unsqueeze_190 = unsqueeze_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_378: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_54, [1, 128, 16, 64]);  cat_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_135: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_378, view_373);  view_378 = view_373 = None
        add_123: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_55: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_122, slice_109], -1);  add_122 = slice_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_56: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_123, slice_111], -1);  add_123 = slice_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_147: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_55, [0, 2, 1, 3]);  cat_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_148: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_56, [0, 2, 1, 3]);  cat_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_508: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_147, torch.float32);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_149: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_508, [0, 1, 3, 2]);  convert_element_type_508 = None
        convert_element_type_509: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_149, torch.bfloat16);  permute_149 = None
        convert_element_type_default_42: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_148, torch.bfloat16);  permute_148 = None
        expand_109: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_42, [1, 16, 128, 256]);  convert_element_type_default_42 = None
        view_379: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_109, [16, 128, 256]);  expand_109 = None
        expand_110: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_509, [1, 16, 256, 128]);  convert_element_type_509 = None
        view_380: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_110, [16, 256, 128]);  expand_110 = None
        bmm_26: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_379, view_380)
        view_381: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [1, 16, 128, 128]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_26: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_381, 16.0);  view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_124: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_26, where);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_13: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_124, [-1], True)
        sub_29: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_124, amax_13);  add_124 = amax_13 = None
        exp_13: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
        sum_14: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_27: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_513: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_111: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_513, [1, 16, 128, 128]);  convert_element_type_513 = None
        view_382: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_111, [16, 128, 128]);  expand_111 = None
        expand_112: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_146, [1, 16, 128, 256]);  permute_146 = None
        view_383: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_112, [16, 128, 256]);  expand_112 = None
        bmm_27: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_382, view_383)
        view_384: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [1, 16, 128, 256]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_150: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        clone_110: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_385: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [1, 128, 4096]);  clone_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_516: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        permute_151: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_516, [1, 0]);  convert_element_type_516 = None
        view_386: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_385, [128, 4096]);  view_385 = None
        mm_55: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_386, permute_151)
        view_387: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [1, 128, 4096]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_519: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16);  primals_154 = None
        convert_element_type_520: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        permute_152: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_520, [1, 0]);  convert_element_type_520 = None
        addmm_26: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_519, view_364, permute_152);  convert_element_type_519 = None
        view_389: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_136: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_389, 0.5)
        convert_element_type_525: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_389, torch.float32)
        pow_14: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_525, 3.0);  convert_element_type_525 = None
        mul_137: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_125: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_389, mul_137);  view_389 = mul_137 = None
        mul_138: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, 0.7978845608028654);  add_125 = None
        tanh_13: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_138);  mul_138 = None
        add_126: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_139: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, add_126);  mul_136 = add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_526: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_527: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        convert_element_type_528: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_139, torch.bfloat16);  mul_139 = None
        view_390: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_528, [128, 16384]);  convert_element_type_528 = None
        permute_153: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_527, [1, 0]);  convert_element_type_527 = None
        addmm_27: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_526, view_390, permute_153);  convert_element_type_526 = None
        view_391: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [1, 128, 4096]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_127: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_387, view_391);  view_387 = view_391 = None
        add_128: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_127, add_119);  add_127 = add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_56: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_14[0]
        getitem_57: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_129: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_14: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_30: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_128, getitem_57);  getitem_57 = None
        mul_140: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_14);  sub_30 = None
        mul_141: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, primals_157)
        add_130: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_141, primals_158);  mul_141 = primals_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_532: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        convert_element_type_533: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_130, torch.bfloat16);  add_130 = None
        permute_154: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_532, [1, 0]);  convert_element_type_532 = None
        view_392: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_533, [128, 4096]);  convert_element_type_533 = None
        mm_56: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_392, permute_154)
        view_393: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [1, 128, 4096]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_536: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        permute_155: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_536, [1, 0]);  convert_element_type_536 = None
        mm_57: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_392, permute_155)
        view_395: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [1, 128, 4096]);  mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_540: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16);  primals_161 = None
        permute_156: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_540, [1, 0]);  convert_element_type_540 = None
        mm_58: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_392, permute_156)
        view_397: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [1, 128, 4096]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_398: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_393, [1, 128, 16, 256]);  view_393 = None
        view_399: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_395, [1, 128, 16, 256]);  view_395 = None
        view_400: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [1, 128, 16, 256]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_157: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_28: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_162, [1, 1, 1]);  primals_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_14: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_28, 1, repeat_1);  repeat_28 = None
        convert_element_type_544: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_14, torch.bfloat16);  gather_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_14 = torch.ops.aten.split.Tensor(convert_element_type_544, 32, -1);  convert_element_type_544 = None
        getitem_58: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_14[0]
        getitem_59: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_14[1];  split_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_116: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_399, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_117: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_399, 3, 64, 9223372036854775807);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_118: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_398, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_119: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_398, 3, 64, 9223372036854775807);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_193: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_58, 2);  getitem_58 = None
        unsqueeze_194: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 4);  unsqueeze_193 = None
        expand_113: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_194, [1, 128, 1, 32, 2])
        clone_113: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_401: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [1, 128, 1, 64]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_195: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_59, 2);  getitem_59 = None
        unsqueeze_196: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 4);  unsqueeze_195 = None
        expand_114: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_196, [1, 128, 1, 32, 2])
        clone_114: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
        view_402: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [1, 128, 1, 64]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_142: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_116, view_402)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_120: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_116, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_121: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_116, 3, 1, 9223372036854775807, 2);  slice_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_28: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_121);  slice_121 = None
        unsqueeze_197: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_28, 4);  neg_28 = None
        unsqueeze_198: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_120, 4);  slice_120 = None
        cat_57: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_197, unsqueeze_198], -1);  unsqueeze_197 = unsqueeze_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_403: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_57, [1, 128, 16, 64]);  cat_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_143: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_403, view_401);  view_403 = None
        add_131: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        mul_144: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_118, view_402);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_122: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_118, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_123: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_118, 3, 1, 9223372036854775807, 2);  slice_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_29: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_123);  slice_123 = None
        unsqueeze_203: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_29, 4);  neg_29 = None
        unsqueeze_204: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_122, 4);  slice_122 = None
        cat_58: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_203, unsqueeze_204], -1);  unsqueeze_203 = unsqueeze_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_406: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_58, [1, 128, 16, 64]);  cat_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_145: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_406, view_401);  view_406 = view_401 = None
        add_132: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_59: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_131, slice_117], -1);  add_131 = slice_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_60: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_132, slice_119], -1);  add_132 = slice_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_158: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_59, [0, 2, 1, 3]);  cat_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_159: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_60, [0, 2, 1, 3]);  cat_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_546: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_158, torch.float32);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_160: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_546, [0, 1, 3, 2]);  convert_element_type_546 = None
        convert_element_type_547: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_160, torch.bfloat16);  permute_160 = None
        convert_element_type_default_41: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_159, torch.bfloat16);  permute_159 = None
        expand_117: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_41, [1, 16, 128, 256]);  convert_element_type_default_41 = None
        view_407: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_117, [16, 128, 256]);  expand_117 = None
        expand_118: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_547, [1, 16, 256, 128]);  convert_element_type_547 = None
        view_408: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_118, [16, 256, 128]);  expand_118 = None
        bmm_28: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_407, view_408)
        view_409: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_28: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_409, 16.0);  view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_133: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_28, where);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_14: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_133, [-1], True)
        sub_31: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_133, amax_14);  add_133 = amax_14 = None
        exp_14: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_15: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_29: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_551: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_29, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_119: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_551, [1, 16, 128, 128]);  convert_element_type_551 = None
        view_410: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_119, [16, 128, 128]);  expand_119 = None
        expand_120: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_157, [1, 16, 128, 256]);  permute_157 = None
        view_411: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_120, [16, 128, 256]);  expand_120 = None
        bmm_29: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_410, view_411)
        view_412: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [1, 16, 128, 256]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_161: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        clone_118: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_413: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_118, [1, 128, 4096]);  clone_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_554: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        permute_162: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_554, [1, 0]);  convert_element_type_554 = None
        view_414: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_413, [128, 4096]);  view_413 = None
        mm_59: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_414, permute_162)
        view_415: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [1, 128, 4096]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_557: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16);  primals_165 = None
        convert_element_type_558: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        permute_163: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_558, [1, 0]);  convert_element_type_558 = None
        addmm_28: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_557, view_392, permute_163);  convert_element_type_557 = None
        view_417: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_146: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_417, 0.5)
        convert_element_type_563: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_417, torch.float32)
        pow_15: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_563, 3.0);  convert_element_type_563 = None
        mul_147: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_134: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_417, mul_147);  view_417 = mul_147 = None
        mul_148: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_134, 0.7978845608028654);  add_134 = None
        tanh_14: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_148);  mul_148 = None
        add_135: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_149: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, add_135);  mul_146 = add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_564: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_167, torch.bfloat16);  primals_167 = None
        convert_element_type_565: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_166, torch.bfloat16);  primals_166 = None
        convert_element_type_566: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None
        view_418: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_566, [128, 16384]);  convert_element_type_566 = None
        permute_164: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_565, [1, 0]);  convert_element_type_565 = None
        addmm_29: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_564, view_418, permute_164);  convert_element_type_564 = None
        view_419: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 4096]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_136: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_415, view_419);  view_415 = view_419 = None
        add_137: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_136, add_128);  add_136 = add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_60: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_15[0]
        getitem_61: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_138: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_15: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_32: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_137, getitem_61);  getitem_61 = None
        mul_150: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_15);  sub_32 = None
        mul_151: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, primals_168)
        add_139: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, primals_169);  mul_151 = primals_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_570: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convert_element_type_571: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None
        permute_165: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_570, [1, 0]);  convert_element_type_570 = None
        view_420: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_571, [128, 4096]);  convert_element_type_571 = None
        mm_60: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_420, permute_165)
        view_421: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [1, 128, 4096]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_574: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        permute_166: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_574, [1, 0]);  convert_element_type_574 = None
        mm_61: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_420, permute_166)
        view_423: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_61, [1, 128, 4096]);  mm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_578: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        permute_167: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_578, [1, 0]);  convert_element_type_578 = None
        mm_62: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_420, permute_167)
        view_425: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [1, 128, 4096]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_426: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_421, [1, 128, 16, 256]);  view_421 = None
        view_427: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_423, [1, 128, 16, 256]);  view_423 = None
        view_428: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_425, [1, 128, 16, 256]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_168: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_30: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_173, [1, 1, 1]);  primals_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_15: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_30, 1, repeat_1);  repeat_30 = None
        convert_element_type_582: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_15, torch.bfloat16);  gather_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_15 = torch.ops.aten.split.Tensor(convert_element_type_582, 32, -1);  convert_element_type_582 = None
        getitem_62: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_15[0]
        getitem_63: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_15[1];  split_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_124: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_427, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_125: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_427, 3, 64, 9223372036854775807);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_126: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_426, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_127: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_426, 3, 64, 9223372036854775807);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_206: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_62, 2);  getitem_62 = None
        unsqueeze_207: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 4);  unsqueeze_206 = None
        expand_121: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_207, [1, 128, 1, 32, 2])
        clone_121: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_429: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [1, 128, 1, 64]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_208: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_63, 2);  getitem_63 = None
        unsqueeze_209: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 4);  unsqueeze_208 = None
        expand_122: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_209, [1, 128, 1, 32, 2])
        clone_122: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_430: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [1, 128, 1, 64]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_152: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_124, view_430)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_128: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_124, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_129: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_124, 3, 1, 9223372036854775807, 2);  slice_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_30: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_129);  slice_129 = None
        unsqueeze_210: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_30, 4);  neg_30 = None
        unsqueeze_211: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_128, 4);  slice_128 = None
        cat_61: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_210, unsqueeze_211], -1);  unsqueeze_210 = unsqueeze_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_431: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_61, [1, 128, 16, 64]);  cat_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_153: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_431, view_429);  view_431 = None
        add_140: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
        mul_154: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_126, view_430);  view_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_130: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_126, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_131: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_126, 3, 1, 9223372036854775807, 2);  slice_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_31: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_131);  slice_131 = None
        unsqueeze_216: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_31, 4);  neg_31 = None
        unsqueeze_217: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_130, 4);  slice_130 = None
        cat_62: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_216, unsqueeze_217], -1);  unsqueeze_216 = unsqueeze_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_434: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_62, [1, 128, 16, 64]);  cat_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_155: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_434, view_429);  view_434 = view_429 = None
        add_141: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_63: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_140, slice_125], -1);  add_140 = slice_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_64: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_141, slice_127], -1);  add_141 = slice_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_169: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_63, [0, 2, 1, 3]);  cat_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_170: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_64, [0, 2, 1, 3]);  cat_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_584: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_169, torch.float32);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_171: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_584, [0, 1, 3, 2]);  convert_element_type_584 = None
        convert_element_type_585: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_171, torch.bfloat16);  permute_171 = None
        convert_element_type_default_40: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_170, torch.bfloat16);  permute_170 = None
        expand_125: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_40, [1, 16, 128, 256]);  convert_element_type_default_40 = None
        view_435: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_125, [16, 128, 256]);  expand_125 = None
        expand_126: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_585, [1, 16, 256, 128]);  convert_element_type_585 = None
        view_436: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_126, [16, 256, 128]);  expand_126 = None
        bmm_30: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_435, view_436)
        view_437: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [1, 16, 128, 128]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_30: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_437, 16.0);  view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_142: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_30, where);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_15: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_142, [-1], True)
        sub_33: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_142, amax_15);  add_142 = amax_15 = None
        exp_15: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_16: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_31: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_589: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_127: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_589, [1, 16, 128, 128]);  convert_element_type_589 = None
        view_438: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_127, [16, 128, 128]);  expand_127 = None
        expand_128: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_168, [1, 16, 128, 256]);  permute_168 = None
        view_439: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_128, [16, 128, 256]);  expand_128 = None
        bmm_31: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_438, view_439)
        view_440: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [1, 16, 128, 256]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_172: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None
        clone_126: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_441: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_126, [1, 128, 4096]);  clone_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_592: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16);  primals_174 = None
        permute_173: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_592, [1, 0]);  convert_element_type_592 = None
        view_442: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [128, 4096]);  view_441 = None
        mm_63: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_442, permute_173)
        view_443: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [1, 128, 4096]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_595: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        convert_element_type_596: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        permute_174: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_596, [1, 0]);  convert_element_type_596 = None
        addmm_30: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_595, view_420, permute_174);  convert_element_type_595 = None
        view_445: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_445, 0.5)
        convert_element_type_601: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_445, torch.float32)
        pow_16: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_601, 3.0);  convert_element_type_601 = None
        mul_157: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_143: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_445, mul_157);  view_445 = mul_157 = None
        mul_158: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_143, 0.7978845608028654);  add_143 = None
        tanh_15: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_144: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_159: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, add_144);  mul_156 = add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_602: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        convert_element_type_603: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_177, torch.bfloat16);  primals_177 = None
        convert_element_type_604: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_159, torch.bfloat16);  mul_159 = None
        view_446: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_604, [128, 16384]);  convert_element_type_604 = None
        permute_175: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_603, [1, 0]);  convert_element_type_603 = None
        addmm_31: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_602, view_446, permute_175);  convert_element_type_602 = None
        view_447: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 4096]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_145: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_443, view_447);  view_443 = view_447 = None
        add_146: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, add_137);  add_145 = add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_16[0]
        getitem_65: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_147: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_16: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        sub_34: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_146, getitem_65);  getitem_65 = None
        mul_160: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_16);  sub_34 = None
        mul_161: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, primals_179)
        add_148: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_161, primals_180);  mul_161 = primals_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_608: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        convert_element_type_609: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_148, torch.bfloat16);  add_148 = None
        permute_176: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_608, [1, 0]);  convert_element_type_608 = None
        view_448: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_609, [128, 4096]);  convert_element_type_609 = None
        mm_64: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_448, permute_176)
        view_449: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [1, 128, 4096]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_612: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        permute_177: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_612, [1, 0]);  convert_element_type_612 = None
        mm_65: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_448, permute_177)
        view_451: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [1, 128, 4096]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_616: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16);  primals_183 = None
        permute_178: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_616, [1, 0]);  convert_element_type_616 = None
        mm_66: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_448, permute_178)
        view_453: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [1, 128, 4096]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_454: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_449, [1, 128, 16, 256]);  view_449 = None
        view_455: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_451, [1, 128, 16, 256]);  view_451 = None
        view_456: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_453, [1, 128, 16, 256]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_179: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_32: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_184, [1, 1, 1]);  primals_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_16: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_32, 1, repeat_1);  repeat_32 = None
        convert_element_type_620: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_16, torch.bfloat16);  gather_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_16 = torch.ops.aten.split.Tensor(convert_element_type_620, 32, -1);  convert_element_type_620 = None
        getitem_66: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_16[0]
        getitem_67: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_16[1];  split_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_132: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_455, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_133: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_455, 3, 64, 9223372036854775807);  view_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_134: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_454, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_135: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_454, 3, 64, 9223372036854775807);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_219: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_66, 2);  getitem_66 = None
        unsqueeze_220: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 4);  unsqueeze_219 = None
        expand_129: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_220, [1, 128, 1, 32, 2])
        clone_129: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_457: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [1, 128, 1, 64]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_221: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_67, 2);  getitem_67 = None
        unsqueeze_222: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 4);  unsqueeze_221 = None
        expand_130: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_222, [1, 128, 1, 32, 2])
        clone_130: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_458: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [1, 128, 1, 64]);  clone_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_162: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_132, view_458)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_136: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_132, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_137: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_132, 3, 1, 9223372036854775807, 2);  slice_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_32: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_137);  slice_137 = None
        unsqueeze_223: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_32, 4);  neg_32 = None
        unsqueeze_224: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_136, 4);  slice_136 = None
        cat_65: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_223, unsqueeze_224], -1);  unsqueeze_223 = unsqueeze_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_459: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_65, [1, 128, 16, 64]);  cat_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_163: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_459, view_457);  view_459 = None
        add_149: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        mul_164: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_134, view_458);  view_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_138: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_134, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_139: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_134, 3, 1, 9223372036854775807, 2);  slice_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_33: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_139);  slice_139 = None
        unsqueeze_229: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_33, 4);  neg_33 = None
        unsqueeze_230: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_138, 4);  slice_138 = None
        cat_66: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_229, unsqueeze_230], -1);  unsqueeze_229 = unsqueeze_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_462: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_66, [1, 128, 16, 64]);  cat_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_165: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_462, view_457);  view_462 = view_457 = None
        add_150: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_67: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_149, slice_133], -1);  add_149 = slice_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_68: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_150, slice_135], -1);  add_150 = slice_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_180: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_67, [0, 2, 1, 3]);  cat_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_181: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_68, [0, 2, 1, 3]);  cat_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_622: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_180, torch.float32);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_182: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_622, [0, 1, 3, 2]);  convert_element_type_622 = None
        convert_element_type_623: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_182, torch.bfloat16);  permute_182 = None
        convert_element_type_default_39: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_181, torch.bfloat16);  permute_181 = None
        expand_133: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_39, [1, 16, 128, 256]);  convert_element_type_default_39 = None
        view_463: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_133, [16, 128, 256]);  expand_133 = None
        expand_134: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_623, [1, 16, 256, 128]);  convert_element_type_623 = None
        view_464: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_134, [16, 256, 128]);  expand_134 = None
        bmm_32: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_463, view_464)
        view_465: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_32: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_465, 16.0);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_151: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_32, where);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_16: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_151, [-1], True)
        sub_35: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_151, amax_16);  add_151 = amax_16 = None
        exp_16: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
        sum_17: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_33: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_627: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_135: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_627, [1, 16, 128, 128]);  convert_element_type_627 = None
        view_466: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_135, [16, 128, 128]);  expand_135 = None
        expand_136: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_179, [1, 16, 128, 256]);  permute_179 = None
        view_467: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_136, [16, 128, 256]);  expand_136 = None
        bmm_33: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_466, view_467)
        view_468: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [1, 16, 128, 256]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_183: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
        clone_134: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_469: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_134, [1, 128, 4096]);  clone_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_630: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        permute_184: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_630, [1, 0]);  convert_element_type_630 = None
        view_470: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_469, [128, 4096]);  view_469 = None
        mm_67: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_470, permute_184)
        view_471: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [1, 128, 4096]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_633: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_187, torch.bfloat16);  primals_187 = None
        convert_element_type_634: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_186, torch.bfloat16);  primals_186 = None
        permute_185: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_634, [1, 0]);  convert_element_type_634 = None
        addmm_32: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_633, view_448, permute_185);  convert_element_type_633 = None
        view_473: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_166: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_473, 0.5)
        convert_element_type_639: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_473, torch.float32)
        pow_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_639, 3.0);  convert_element_type_639 = None
        mul_167: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_152: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_473, mul_167);  view_473 = mul_167 = None
        mul_168: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_152, 0.7978845608028654);  add_152 = None
        tanh_16: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_168);  mul_168 = None
        add_153: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_169: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, add_153);  mul_166 = add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_640: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        convert_element_type_641: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convert_element_type_642: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_169, torch.bfloat16);  mul_169 = None
        view_474: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_642, [128, 16384]);  convert_element_type_642 = None
        permute_186: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_641, [1, 0]);  convert_element_type_641 = None
        addmm_33: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_640, view_474, permute_186);  convert_element_type_640 = None
        view_475: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [1, 128, 4096]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_154: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_471, view_475);  view_471 = view_475 = None
        add_155: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_154, add_146);  add_154 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_155, [2], correction = 0, keepdim = True)
        getitem_68: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_17[0]
        getitem_69: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_156: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_17: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_36: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_155, getitem_69);  getitem_69 = None
        mul_170: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_17);  sub_36 = None
        mul_171: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, primals_190)
        add_157: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_171, primals_191);  mul_171 = primals_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_646: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16);  primals_192 = None
        convert_element_type_647: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16);  add_157 = None
        permute_187: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_646, [1, 0]);  convert_element_type_646 = None
        view_476: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_647, [128, 4096]);  convert_element_type_647 = None
        mm_68: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_476, permute_187)
        view_477: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [1, 128, 4096]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_650: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_193, torch.bfloat16);  primals_193 = None
        permute_188: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_650, [1, 0]);  convert_element_type_650 = None
        mm_69: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_476, permute_188)
        view_479: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [1, 128, 4096]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_654: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        permute_189: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_654, [1, 0]);  convert_element_type_654 = None
        mm_70: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_476, permute_189)
        view_481: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [1, 128, 4096]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_482: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_477, [1, 128, 16, 256]);  view_477 = None
        view_483: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_479, [1, 128, 16, 256]);  view_479 = None
        view_484: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_481, [1, 128, 16, 256]);  view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_190: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_34: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_195, [1, 1, 1]);  primals_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_17: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_34, 1, repeat_1);  repeat_34 = None
        convert_element_type_658: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_17, torch.bfloat16);  gather_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_17 = torch.ops.aten.split.Tensor(convert_element_type_658, 32, -1);  convert_element_type_658 = None
        getitem_70: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_17[0]
        getitem_71: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_17[1];  split_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_140: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_483, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_141: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_483, 3, 64, 9223372036854775807);  view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_142: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_482, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_143: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_482, 3, 64, 9223372036854775807);  view_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_232: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_70, 2);  getitem_70 = None
        unsqueeze_233: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 4);  unsqueeze_232 = None
        expand_137: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_233, [1, 128, 1, 32, 2])
        clone_137: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_485: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [1, 128, 1, 64]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_234: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_71, 2);  getitem_71 = None
        unsqueeze_235: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 4);  unsqueeze_234 = None
        expand_138: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_235, [1, 128, 1, 32, 2])
        clone_138: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
        view_486: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [1, 128, 1, 64]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_172: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_140, view_486)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_144: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_140, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_145: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_140, 3, 1, 9223372036854775807, 2);  slice_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_34: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_145);  slice_145 = None
        unsqueeze_236: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_34, 4);  neg_34 = None
        unsqueeze_237: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_144, 4);  slice_144 = None
        cat_69: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_236, unsqueeze_237], -1);  unsqueeze_236 = unsqueeze_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_487: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_69, [1, 128, 16, 64]);  cat_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_173: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_487, view_485);  view_487 = None
        add_158: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        mul_174: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_142, view_486);  view_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_146: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_142, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_147: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_142, 3, 1, 9223372036854775807, 2);  slice_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_35: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_147);  slice_147 = None
        unsqueeze_242: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_35, 4);  neg_35 = None
        unsqueeze_243: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_146, 4);  slice_146 = None
        cat_70: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_242, unsqueeze_243], -1);  unsqueeze_242 = unsqueeze_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_490: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_70, [1, 128, 16, 64]);  cat_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_175: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_490, view_485);  view_490 = view_485 = None
        add_159: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_71: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_158, slice_141], -1);  add_158 = slice_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_72: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_159, slice_143], -1);  add_159 = slice_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_191: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_71, [0, 2, 1, 3]);  cat_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_192: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_72, [0, 2, 1, 3]);  cat_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_660: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_191, torch.float32);  permute_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_193: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_660, [0, 1, 3, 2]);  convert_element_type_660 = None
        convert_element_type_661: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_193, torch.bfloat16);  permute_193 = None
        convert_element_type_default_38: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_192, torch.bfloat16);  permute_192 = None
        expand_141: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_38, [1, 16, 128, 256]);  convert_element_type_default_38 = None
        view_491: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_141, [16, 128, 256]);  expand_141 = None
        expand_142: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_661, [1, 16, 256, 128]);  convert_element_type_661 = None
        view_492: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_142, [16, 256, 128]);  expand_142 = None
        bmm_34: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_491, view_492)
        view_493: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [1, 16, 128, 128]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_34: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_493, 16.0);  view_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_160: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_34, where);  div_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_17: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_160, [-1], True)
        sub_37: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_160, amax_17);  add_160 = amax_17 = None
        exp_17: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_18: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_35: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_665: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_35, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_143: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_665, [1, 16, 128, 128]);  convert_element_type_665 = None
        view_494: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_143, [16, 128, 128]);  expand_143 = None
        expand_144: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_190, [1, 16, 128, 256]);  permute_190 = None
        view_495: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_144, [16, 128, 256]);  expand_144 = None
        bmm_35: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_494, view_495)
        view_496: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [1, 16, 128, 256]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_194: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        clone_142: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_497: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_142, [1, 128, 4096]);  clone_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_668: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_196, torch.bfloat16);  primals_196 = None
        permute_195: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_668, [1, 0]);  convert_element_type_668 = None
        view_498: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_497, [128, 4096]);  view_497 = None
        mm_71: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_498, permute_195)
        view_499: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [1, 128, 4096]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_671: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        convert_element_type_672: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16);  primals_197 = None
        permute_196: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_672, [1, 0]);  convert_element_type_672 = None
        addmm_34: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_671, view_476, permute_196);  convert_element_type_671 = None
        view_501: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_176: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        convert_element_type_677: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_501, torch.float32)
        pow_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_677, 3.0);  convert_element_type_677 = None
        mul_177: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_161: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_501, mul_177);  view_501 = mul_177 = None
        mul_178: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
        add_162: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_179: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, add_162);  mul_176 = add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_678: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        convert_element_type_679: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16);  primals_199 = None
        convert_element_type_680: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_179, torch.bfloat16);  mul_179 = None
        view_502: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_680, [128, 16384]);  convert_element_type_680 = None
        permute_197: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_679, [1, 0]);  convert_element_type_679 = None
        addmm_35: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_678, view_502, permute_197);  convert_element_type_678 = None
        view_503: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 4096]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_163: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_499, view_503);  view_499 = view_503 = None
        add_164: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_163, add_155);  add_163 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_72: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_18[0]
        getitem_73: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_165: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_18: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_38: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_164, getitem_73);  getitem_73 = None
        mul_180: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_18);  sub_38 = None
        mul_181: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, primals_201)
        add_166: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_181, primals_202);  mul_181 = primals_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_684: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_203, torch.bfloat16);  primals_203 = None
        convert_element_type_685: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_166, torch.bfloat16);  add_166 = None
        permute_198: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_684, [1, 0]);  convert_element_type_684 = None
        view_504: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_685, [128, 4096]);  convert_element_type_685 = None
        mm_72: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_504, permute_198)
        view_505: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [1, 128, 4096]);  mm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_688: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_204, torch.bfloat16);  primals_204 = None
        permute_199: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_688, [1, 0]);  convert_element_type_688 = None
        mm_73: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_504, permute_199)
        view_507: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_73, [1, 128, 4096]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_692: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_205, torch.bfloat16);  primals_205 = None
        permute_200: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_692, [1, 0]);  convert_element_type_692 = None
        mm_74: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_504, permute_200)
        view_509: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [1, 128, 4096]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_510: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_505, [1, 128, 16, 256]);  view_505 = None
        view_511: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_507, [1, 128, 16, 256]);  view_507 = None
        view_512: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_509, [1, 128, 16, 256]);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_201: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_36: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_206, [1, 1, 1]);  primals_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_18: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_36, 1, repeat_1);  repeat_36 = None
        convert_element_type_696: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_18, torch.bfloat16);  gather_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_18 = torch.ops.aten.split.Tensor(convert_element_type_696, 32, -1);  convert_element_type_696 = None
        getitem_74: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_18[0]
        getitem_75: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_18[1];  split_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_148: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_511, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_149: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_511, 3, 64, 9223372036854775807);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_150: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_510, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_151: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_510, 3, 64, 9223372036854775807);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_245: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_74, 2);  getitem_74 = None
        unsqueeze_246: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 4);  unsqueeze_245 = None
        expand_145: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_246, [1, 128, 1, 32, 2])
        clone_145: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_513: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [1, 128, 1, 64]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_247: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_75, 2);  getitem_75 = None
        unsqueeze_248: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 4);  unsqueeze_247 = None
        expand_146: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_248, [1, 128, 1, 32, 2])
        clone_146: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_514: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [1, 128, 1, 64]);  clone_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_182: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_148, view_514)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_152: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_148, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_153: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_148, 3, 1, 9223372036854775807, 2);  slice_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_36: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_153);  slice_153 = None
        unsqueeze_249: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_36, 4);  neg_36 = None
        unsqueeze_250: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_152, 4);  slice_152 = None
        cat_73: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_249, unsqueeze_250], -1);  unsqueeze_249 = unsqueeze_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_515: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_73, [1, 128, 16, 64]);  cat_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_183: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_515, view_513);  view_515 = None
        add_167: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
        mul_184: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_150, view_514);  view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_154: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_150, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_155: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_150, 3, 1, 9223372036854775807, 2);  slice_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_37: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_155);  slice_155 = None
        unsqueeze_255: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_37, 4);  neg_37 = None
        unsqueeze_256: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_154, 4);  slice_154 = None
        cat_74: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_255, unsqueeze_256], -1);  unsqueeze_255 = unsqueeze_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_518: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_74, [1, 128, 16, 64]);  cat_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_185: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_518, view_513);  view_518 = view_513 = None
        add_168: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_75: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_167, slice_149], -1);  add_167 = slice_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_76: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_168, slice_151], -1);  add_168 = slice_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_202: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_75, [0, 2, 1, 3]);  cat_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_203: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_76, [0, 2, 1, 3]);  cat_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_698: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_202, torch.float32);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_204: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_698, [0, 1, 3, 2]);  convert_element_type_698 = None
        convert_element_type_699: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_204, torch.bfloat16);  permute_204 = None
        convert_element_type_default_37: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_203, torch.bfloat16);  permute_203 = None
        expand_149: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_37, [1, 16, 128, 256]);  convert_element_type_default_37 = None
        view_519: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_149, [16, 128, 256]);  expand_149 = None
        expand_150: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_699, [1, 16, 256, 128]);  convert_element_type_699 = None
        view_520: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_150, [16, 256, 128]);  expand_150 = None
        bmm_36: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_519, view_520)
        view_521: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_36: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_521, 16.0);  view_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_169: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_36, where);  div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_18: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_169, [-1], True)
        sub_39: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_169, amax_18);  add_169 = amax_18 = None
        exp_18: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_19: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_37: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_703: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_37, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_151: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_703, [1, 16, 128, 128]);  convert_element_type_703 = None
        view_522: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_151, [16, 128, 128]);  expand_151 = None
        expand_152: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_201, [1, 16, 128, 256]);  permute_201 = None
        view_523: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_152, [16, 128, 256]);  expand_152 = None
        bmm_37: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_522, view_523)
        view_524: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [1, 16, 128, 256]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_205: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_524, [0, 2, 1, 3]);  view_524 = None
        clone_150: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_525: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_150, [1, 128, 4096]);  clone_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_706: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16);  primals_207 = None
        permute_206: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_706, [1, 0]);  convert_element_type_706 = None
        view_526: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_525, [128, 4096]);  view_525 = None
        mm_75: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_526, permute_206)
        view_527: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [1, 128, 4096]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_709: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_209, torch.bfloat16);  primals_209 = None
        convert_element_type_710: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_208, torch.bfloat16);  primals_208 = None
        permute_207: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_710, [1, 0]);  convert_element_type_710 = None
        addmm_36: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_709, view_504, permute_207);  convert_element_type_709 = None
        view_529: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_186: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        convert_element_type_715: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32)
        pow_19: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_715, 3.0);  convert_element_type_715 = None
        mul_187: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_170: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_529, mul_187);  view_529 = mul_187 = None
        mul_188: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_170, 0.7978845608028654);  add_170 = None
        tanh_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_188);  mul_188 = None
        add_171: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_189: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, add_171);  mul_186 = add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_716: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_211, torch.bfloat16);  primals_211 = None
        convert_element_type_717: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_210, torch.bfloat16);  primals_210 = None
        convert_element_type_718: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        view_530: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_718, [128, 16384]);  convert_element_type_718 = None
        permute_208: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_717, [1, 0]);  convert_element_type_717 = None
        addmm_37: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_716, view_530, permute_208);  convert_element_type_716 = None
        view_531: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 4096]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_172: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_527, view_531);  view_527 = view_531 = None
        add_173: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_172, add_164);  add_172 = add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_76: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_19[0]
        getitem_77: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_174: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_19: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_40: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_173, getitem_77);  getitem_77 = None
        mul_190: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_19);  sub_40 = None
        mul_191: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, primals_212)
        add_175: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_191, primals_213);  mul_191 = primals_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_722: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_214, torch.bfloat16);  primals_214 = None
        convert_element_type_723: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None
        permute_209: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_722, [1, 0]);  convert_element_type_722 = None
        view_532: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_723, [128, 4096]);  convert_element_type_723 = None
        mm_76: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_532, permute_209)
        view_533: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [1, 128, 4096]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_726: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_215, torch.bfloat16);  primals_215 = None
        permute_210: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_726, [1, 0]);  convert_element_type_726 = None
        mm_77: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_532, permute_210)
        view_535: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [1, 128, 4096]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_730: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_216, torch.bfloat16);  primals_216 = None
        permute_211: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_730, [1, 0]);  convert_element_type_730 = None
        mm_78: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_532, permute_211)
        view_537: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [1, 128, 4096]);  mm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_538: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_533, [1, 128, 16, 256]);  view_533 = None
        view_539: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_535, [1, 128, 16, 256]);  view_535 = None
        view_540: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_537, [1, 128, 16, 256]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_212: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_38: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_217, [1, 1, 1]);  primals_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_19: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_38, 1, repeat_1);  repeat_38 = None
        convert_element_type_734: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_19, torch.bfloat16);  gather_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_19 = torch.ops.aten.split.Tensor(convert_element_type_734, 32, -1);  convert_element_type_734 = None
        getitem_78: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_19[0]
        getitem_79: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_19[1];  split_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_156: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_539, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_157: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_539, 3, 64, 9223372036854775807);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_158: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_538, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_159: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_538, 3, 64, 9223372036854775807);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_258: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_78, 2);  getitem_78 = None
        unsqueeze_259: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 4);  unsqueeze_258 = None
        expand_153: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_259, [1, 128, 1, 32, 2])
        clone_153: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_541: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [1, 128, 1, 64]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_260: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_79, 2);  getitem_79 = None
        unsqueeze_261: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 4);  unsqueeze_260 = None
        expand_154: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_261, [1, 128, 1, 32, 2])
        clone_154: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_542: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [1, 128, 1, 64]);  clone_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_192: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_156, view_542)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_160: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_156, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_161: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_156, 3, 1, 9223372036854775807, 2);  slice_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_38: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_161);  slice_161 = None
        unsqueeze_262: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_38, 4);  neg_38 = None
        unsqueeze_263: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_160, 4);  slice_160 = None
        cat_77: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_262, unsqueeze_263], -1);  unsqueeze_262 = unsqueeze_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_543: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_77, [1, 128, 16, 64]);  cat_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_193: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_543, view_541);  view_543 = None
        add_176: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
        mul_194: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_158, view_542);  view_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_162: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_158, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_163: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_158, 3, 1, 9223372036854775807, 2);  slice_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_39: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_163);  slice_163 = None
        unsqueeze_268: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_39, 4);  neg_39 = None
        unsqueeze_269: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_162, 4);  slice_162 = None
        cat_78: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_268, unsqueeze_269], -1);  unsqueeze_268 = unsqueeze_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_546: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_78, [1, 128, 16, 64]);  cat_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_195: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_546, view_541);  view_546 = view_541 = None
        add_177: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_79: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_176, slice_157], -1);  add_176 = slice_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_80: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_177, slice_159], -1);  add_177 = slice_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_213: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_79, [0, 2, 1, 3]);  cat_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_214: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_80, [0, 2, 1, 3]);  cat_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_736: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_213, torch.float32);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_215: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_736, [0, 1, 3, 2]);  convert_element_type_736 = None
        convert_element_type_737: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_215, torch.bfloat16);  permute_215 = None
        convert_element_type_default_36: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_214, torch.bfloat16);  permute_214 = None
        expand_157: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_36, [1, 16, 128, 256]);  convert_element_type_default_36 = None
        view_547: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_157, [16, 128, 256]);  expand_157 = None
        expand_158: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_737, [1, 16, 256, 128]);  convert_element_type_737 = None
        view_548: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_158, [16, 256, 128]);  expand_158 = None
        bmm_38: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_547, view_548)
        view_549: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [1, 16, 128, 128]);  bmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_38: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_549, 16.0);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_178: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_38, where);  div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_19: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_178, [-1], True)
        sub_41: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_178, amax_19);  add_178 = amax_19 = None
        exp_19: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        sum_20: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_39: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_741: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_39, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_159: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_741, [1, 16, 128, 128]);  convert_element_type_741 = None
        view_550: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_159, [16, 128, 128]);  expand_159 = None
        expand_160: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_212, [1, 16, 128, 256]);  permute_212 = None
        view_551: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_160, [16, 128, 256]);  expand_160 = None
        bmm_39: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_550, view_551)
        view_552: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [1, 16, 128, 256]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_216: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None
        clone_158: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_553: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_158, [1, 128, 4096]);  clone_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_744: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        permute_217: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_744, [1, 0]);  convert_element_type_744 = None
        view_554: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_553, [128, 4096]);  view_553 = None
        mm_79: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_554, permute_217)
        view_555: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [1, 128, 4096]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_747: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_220, torch.bfloat16);  primals_220 = None
        convert_element_type_748: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_219, torch.bfloat16);  primals_219 = None
        permute_218: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_748, [1, 0]);  convert_element_type_748 = None
        addmm_38: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_747, view_532, permute_218);  convert_element_type_747 = None
        view_557: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_196: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_557, 0.5)
        convert_element_type_753: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_557, torch.float32)
        pow_20: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_753, 3.0);  convert_element_type_753 = None
        mul_197: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_179: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_557, mul_197);  view_557 = mul_197 = None
        mul_198: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_179, 0.7978845608028654);  add_179 = None
        tanh_19: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_198);  mul_198 = None
        add_180: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_199: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, add_180);  mul_196 = add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_754: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_222, torch.bfloat16);  primals_222 = None
        convert_element_type_755: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_221, torch.bfloat16);  primals_221 = None
        convert_element_type_756: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_199, torch.bfloat16);  mul_199 = None
        view_558: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_756, [128, 16384]);  convert_element_type_756 = None
        permute_219: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_755, [1, 0]);  convert_element_type_755 = None
        addmm_39: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_754, view_558, permute_219);  convert_element_type_754 = None
        view_559: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [1, 128, 4096]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_181: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_555, view_559);  view_555 = view_559 = None
        add_182: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_181, add_173);  add_181 = add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_182, [2], correction = 0, keepdim = True)
        getitem_80: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_20[0]
        getitem_81: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_183: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_20: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        sub_42: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_182, getitem_81);  getitem_81 = None
        mul_200: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_20);  sub_42 = None
        mul_201: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, primals_223)
        add_184: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_201, primals_224);  mul_201 = primals_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_760: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_225, torch.bfloat16);  primals_225 = None
        convert_element_type_761: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None
        permute_220: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_760, [1, 0]);  convert_element_type_760 = None
        view_560: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_761, [128, 4096]);  convert_element_type_761 = None
        mm_80: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_560, permute_220)
        view_561: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [1, 128, 4096]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_764: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_226, torch.bfloat16);  primals_226 = None
        permute_221: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_764, [1, 0]);  convert_element_type_764 = None
        mm_81: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_560, permute_221)
        view_563: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_81, [1, 128, 4096]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_768: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_227, torch.bfloat16);  primals_227 = None
        permute_222: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_768, [1, 0]);  convert_element_type_768 = None
        mm_82: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_560, permute_222)
        view_565: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [1, 128, 4096]);  mm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_566: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_561, [1, 128, 16, 256]);  view_561 = None
        view_567: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_563, [1, 128, 16, 256]);  view_563 = None
        view_568: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_565, [1, 128, 16, 256]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_223: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_40: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_228, [1, 1, 1]);  primals_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_20: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_40, 1, repeat_1);  repeat_40 = None
        convert_element_type_772: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_20, torch.bfloat16);  gather_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_20 = torch.ops.aten.split.Tensor(convert_element_type_772, 32, -1);  convert_element_type_772 = None
        getitem_82: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_20[0]
        getitem_83: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_20[1];  split_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_164: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_567, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_165: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_567, 3, 64, 9223372036854775807);  view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_166: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_566, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_167: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_566, 3, 64, 9223372036854775807);  view_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_271: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_82, 2);  getitem_82 = None
        unsqueeze_272: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 4);  unsqueeze_271 = None
        expand_161: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_272, [1, 128, 1, 32, 2])
        clone_161: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_569: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [1, 128, 1, 64]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_273: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_83, 2);  getitem_83 = None
        unsqueeze_274: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 4);  unsqueeze_273 = None
        expand_162: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_274, [1, 128, 1, 32, 2])
        clone_162: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_570: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [1, 128, 1, 64]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_202: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_164, view_570)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_168: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_164, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_169: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_164, 3, 1, 9223372036854775807, 2);  slice_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_40: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_169);  slice_169 = None
        unsqueeze_275: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_40, 4);  neg_40 = None
        unsqueeze_276: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_168, 4);  slice_168 = None
        cat_81: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_275, unsqueeze_276], -1);  unsqueeze_275 = unsqueeze_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_571: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_81, [1, 128, 16, 64]);  cat_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_203: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_571, view_569);  view_571 = None
        add_185: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
        mul_204: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_166, view_570);  view_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_170: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_166, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_171: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_166, 3, 1, 9223372036854775807, 2);  slice_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_41: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_171);  slice_171 = None
        unsqueeze_281: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_41, 4);  neg_41 = None
        unsqueeze_282: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_170, 4);  slice_170 = None
        cat_82: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_281, unsqueeze_282], -1);  unsqueeze_281 = unsqueeze_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_574: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_82, [1, 128, 16, 64]);  cat_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_205: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_574, view_569);  view_574 = view_569 = None
        add_186: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_83: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_185, slice_165], -1);  add_185 = slice_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_84: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_186, slice_167], -1);  add_186 = slice_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_224: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_83, [0, 2, 1, 3]);  cat_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_225: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_84, [0, 2, 1, 3]);  cat_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_774: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_224, torch.float32);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_226: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_774, [0, 1, 3, 2]);  convert_element_type_774 = None
        convert_element_type_775: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_226, torch.bfloat16);  permute_226 = None
        convert_element_type_default_35: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_225, torch.bfloat16);  permute_225 = None
        expand_165: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_35, [1, 16, 128, 256]);  convert_element_type_default_35 = None
        view_575: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_165, [16, 128, 256]);  expand_165 = None
        expand_166: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_775, [1, 16, 256, 128]);  convert_element_type_775 = None
        view_576: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_166, [16, 256, 128]);  expand_166 = None
        bmm_40: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_575, view_576)
        view_577: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_40: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_577, 16.0);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_187: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_40, where);  div_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_20: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_187, [-1], True)
        sub_43: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_187, amax_20);  add_187 = amax_20 = None
        exp_20: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_21: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_41: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_779: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_41, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_167: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_779, [1, 16, 128, 128]);  convert_element_type_779 = None
        view_578: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_167, [16, 128, 128]);  expand_167 = None
        expand_168: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_223, [1, 16, 128, 256]);  permute_223 = None
        view_579: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_168, [16, 128, 256]);  expand_168 = None
        bmm_41: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_578, view_579)
        view_580: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [1, 16, 128, 256]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_227: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None
        clone_166: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_581: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_166, [1, 128, 4096]);  clone_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_782: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_229, torch.bfloat16);  primals_229 = None
        permute_228: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_782, [1, 0]);  convert_element_type_782 = None
        view_582: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_581, [128, 4096]);  view_581 = None
        mm_83: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_582, permute_228)
        view_583: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_83, [1, 128, 4096]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_785: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_231, torch.bfloat16);  primals_231 = None
        convert_element_type_786: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        permute_229: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_786, [1, 0]);  convert_element_type_786 = None
        addmm_40: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_785, view_560, permute_229);  convert_element_type_785 = None
        view_585: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_206: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_585, 0.5)
        convert_element_type_791: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_585, torch.float32)
        pow_21: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_791, 3.0);  convert_element_type_791 = None
        mul_207: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_188: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_585, mul_207);  view_585 = mul_207 = None
        mul_208: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_188, 0.7978845608028654);  add_188 = None
        tanh_20: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_208);  mul_208 = None
        add_189: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_209: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, add_189);  mul_206 = add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_792: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_233, torch.bfloat16);  primals_233 = None
        convert_element_type_793: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_232, torch.bfloat16);  primals_232 = None
        convert_element_type_794: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_209, torch.bfloat16);  mul_209 = None
        view_586: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_794, [128, 16384]);  convert_element_type_794 = None
        permute_230: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_793, [1, 0]);  convert_element_type_793 = None
        addmm_41: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_792, view_586, permute_230);  convert_element_type_792 = None
        view_587: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [1, 128, 4096]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_190: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_583, view_587);  view_583 = view_587 = None
        add_191: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_190, add_182);  add_190 = add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_191, [2], correction = 0, keepdim = True)
        getitem_84: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_21[0]
        getitem_85: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_192: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_21: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_44: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_191, getitem_85);  getitem_85 = None
        mul_210: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_21);  sub_44 = None
        mul_211: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, primals_234)
        add_193: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_211, primals_235);  mul_211 = primals_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_798: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        convert_element_type_799: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16);  add_193 = None
        permute_231: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_798, [1, 0]);  convert_element_type_798 = None
        view_588: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_799, [128, 4096]);  convert_element_type_799 = None
        mm_84: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_588, permute_231)
        view_589: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [1, 128, 4096]);  mm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_802: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_237, torch.bfloat16);  primals_237 = None
        permute_232: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_802, [1, 0]);  convert_element_type_802 = None
        mm_85: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_588, permute_232)
        view_591: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_85, [1, 128, 4096]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_806: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_238, torch.bfloat16);  primals_238 = None
        permute_233: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_806, [1, 0]);  convert_element_type_806 = None
        mm_86: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_588, permute_233)
        view_593: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [1, 128, 4096]);  mm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_594: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_589, [1, 128, 16, 256]);  view_589 = None
        view_595: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_591, [1, 128, 16, 256]);  view_591 = None
        view_596: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_593, [1, 128, 16, 256]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_234: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_42: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_239, [1, 1, 1]);  primals_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_21: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_42, 1, repeat_1);  repeat_42 = None
        convert_element_type_810: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_21, torch.bfloat16);  gather_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_21 = torch.ops.aten.split.Tensor(convert_element_type_810, 32, -1);  convert_element_type_810 = None
        getitem_86: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_21[0]
        getitem_87: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_21[1];  split_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_172: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_595, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_173: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_595, 3, 64, 9223372036854775807);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_174: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_594, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_175: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_594, 3, 64, 9223372036854775807);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_284: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_86, 2);  getitem_86 = None
        unsqueeze_285: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 4);  unsqueeze_284 = None
        expand_169: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_285, [1, 128, 1, 32, 2])
        clone_169: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_597: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [1, 128, 1, 64]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_286: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_87, 2);  getitem_87 = None
        unsqueeze_287: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 4);  unsqueeze_286 = None
        expand_170: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_287, [1, 128, 1, 32, 2])
        clone_170: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
        view_598: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [1, 128, 1, 64]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_212: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_172, view_598)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_176: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_172, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_177: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_172, 3, 1, 9223372036854775807, 2);  slice_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_42: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_177);  slice_177 = None
        unsqueeze_288: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_42, 4);  neg_42 = None
        unsqueeze_289: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_176, 4);  slice_176 = None
        cat_85: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_288, unsqueeze_289], -1);  unsqueeze_288 = unsqueeze_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_599: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_85, [1, 128, 16, 64]);  cat_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_213: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_599, view_597);  view_599 = None
        add_194: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
        mul_214: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_174, view_598);  view_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_178: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_174, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_179: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_174, 3, 1, 9223372036854775807, 2);  slice_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_43: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_179);  slice_179 = None
        unsqueeze_294: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_43, 4);  neg_43 = None
        unsqueeze_295: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_178, 4);  slice_178 = None
        cat_86: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_294, unsqueeze_295], -1);  unsqueeze_294 = unsqueeze_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_602: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_86, [1, 128, 16, 64]);  cat_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_215: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_602, view_597);  view_602 = view_597 = None
        add_195: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_87: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_194, slice_173], -1);  add_194 = slice_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_88: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_195, slice_175], -1);  add_195 = slice_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_235: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_87, [0, 2, 1, 3]);  cat_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_236: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_88, [0, 2, 1, 3]);  cat_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_812: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_235, torch.float32);  permute_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_237: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_812, [0, 1, 3, 2]);  convert_element_type_812 = None
        convert_element_type_813: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_237, torch.bfloat16);  permute_237 = None
        convert_element_type_default_34: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_236, torch.bfloat16);  permute_236 = None
        expand_173: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_34, [1, 16, 128, 256]);  convert_element_type_default_34 = None
        view_603: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_173, [16, 128, 256]);  expand_173 = None
        expand_174: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_813, [1, 16, 256, 128]);  convert_element_type_813 = None
        view_604: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_174, [16, 256, 128]);  expand_174 = None
        bmm_42: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_603, view_604)
        view_605: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [1, 16, 128, 128]);  bmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_42: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_605, 16.0);  view_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_196: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_42, where);  div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_21: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_196, [-1], True)
        sub_45: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_196, amax_21);  add_196 = amax_21 = None
        exp_21: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_22: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_43: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_817: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_43, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_175: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_817, [1, 16, 128, 128]);  convert_element_type_817 = None
        view_606: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_175, [16, 128, 128]);  expand_175 = None
        expand_176: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_234, [1, 16, 128, 256]);  permute_234 = None
        view_607: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_176, [16, 128, 256]);  expand_176 = None
        bmm_43: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_606, view_607)
        view_608: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [1, 16, 128, 256]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_238: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None
        clone_174: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_609: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_174, [1, 128, 4096]);  clone_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_820: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_240, torch.bfloat16);  primals_240 = None
        permute_239: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_820, [1, 0]);  convert_element_type_820 = None
        view_610: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_609, [128, 4096]);  view_609 = None
        mm_87: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_610, permute_239)
        view_611: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [1, 128, 4096]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_823: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16);  primals_242 = None
        convert_element_type_824: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_241, torch.bfloat16);  primals_241 = None
        permute_240: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_824, [1, 0]);  convert_element_type_824 = None
        addmm_42: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_823, view_588, permute_240);  convert_element_type_823 = None
        view_613: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_216: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_613, 0.5)
        convert_element_type_829: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_613, torch.float32)
        pow_22: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_829, 3.0);  convert_element_type_829 = None
        mul_217: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_197: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_613, mul_217);  view_613 = mul_217 = None
        mul_218: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_197, 0.7978845608028654);  add_197 = None
        tanh_21: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_218);  mul_218 = None
        add_198: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_219: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_216, add_198);  mul_216 = add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_830: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_244, torch.bfloat16);  primals_244 = None
        convert_element_type_831: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_243, torch.bfloat16);  primals_243 = None
        convert_element_type_832: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_219, torch.bfloat16);  mul_219 = None
        view_614: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_832, [128, 16384]);  convert_element_type_832 = None
        permute_241: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_831, [1, 0]);  convert_element_type_831 = None
        addmm_43: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_830, view_614, permute_241);  convert_element_type_830 = None
        view_615: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [1, 128, 4096]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_199: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_611, view_615);  view_611 = view_615 = None
        add_200: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_199, add_191);  add_199 = add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_200, [2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_22[0]
        getitem_89: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_201: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_22: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        sub_46: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_200, getitem_89);  getitem_89 = None
        mul_220: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_22);  sub_46 = None
        mul_221: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, primals_245)
        add_202: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, primals_246);  mul_221 = primals_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_836: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_247, torch.bfloat16);  primals_247 = None
        convert_element_type_837: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_202, torch.bfloat16);  add_202 = None
        permute_242: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_836, [1, 0]);  convert_element_type_836 = None
        view_616: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_837, [128, 4096]);  convert_element_type_837 = None
        mm_88: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_616, permute_242)
        view_617: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [1, 128, 4096]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_840: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16);  primals_248 = None
        permute_243: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_840, [1, 0]);  convert_element_type_840 = None
        mm_89: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_616, permute_243)
        view_619: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [1, 128, 4096]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_844: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_249, torch.bfloat16);  primals_249 = None
        permute_244: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_844, [1, 0]);  convert_element_type_844 = None
        mm_90: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_616, permute_244)
        view_621: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [1, 128, 4096]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_622: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_617, [1, 128, 16, 256]);  view_617 = None
        view_623: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_619, [1, 128, 16, 256]);  view_619 = None
        view_624: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_621, [1, 128, 16, 256]);  view_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_245: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_624, [0, 2, 1, 3]);  view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_44: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_250, [1, 1, 1]);  primals_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_22: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_44, 1, repeat_1);  repeat_44 = None
        convert_element_type_848: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_22, torch.bfloat16);  gather_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_22 = torch.ops.aten.split.Tensor(convert_element_type_848, 32, -1);  convert_element_type_848 = None
        getitem_90: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_22[0]
        getitem_91: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_22[1];  split_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_180: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_623, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_181: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_623, 3, 64, 9223372036854775807);  view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_182: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_622, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_183: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_622, 3, 64, 9223372036854775807);  view_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_297: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_90, 2);  getitem_90 = None
        unsqueeze_298: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 4);  unsqueeze_297 = None
        expand_177: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_298, [1, 128, 1, 32, 2])
        clone_177: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_625: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [1, 128, 1, 64]);  clone_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_299: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_91, 2);  getitem_91 = None
        unsqueeze_300: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 4);  unsqueeze_299 = None
        expand_178: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_300, [1, 128, 1, 32, 2])
        clone_178: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_626: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [1, 128, 1, 64]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_222: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_180, view_626)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_184: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_180, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_185: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_180, 3, 1, 9223372036854775807, 2);  slice_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_44: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_185);  slice_185 = None
        unsqueeze_301: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_44, 4);  neg_44 = None
        unsqueeze_302: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_184, 4);  slice_184 = None
        cat_89: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_301, unsqueeze_302], -1);  unsqueeze_301 = unsqueeze_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_627: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_89, [1, 128, 16, 64]);  cat_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_223: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_627, view_625);  view_627 = None
        add_203: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
        mul_224: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_182, view_626);  view_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_186: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_182, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_187: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_182, 3, 1, 9223372036854775807, 2);  slice_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_45: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_187);  slice_187 = None
        unsqueeze_307: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_45, 4);  neg_45 = None
        unsqueeze_308: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_186, 4);  slice_186 = None
        cat_90: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_307, unsqueeze_308], -1);  unsqueeze_307 = unsqueeze_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_630: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_90, [1, 128, 16, 64]);  cat_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_225: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_630, view_625);  view_630 = view_625 = None
        add_204: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_91: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_203, slice_181], -1);  add_203 = slice_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_92: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_204, slice_183], -1);  add_204 = slice_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_246: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_91, [0, 2, 1, 3]);  cat_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_247: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_92, [0, 2, 1, 3]);  cat_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_850: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_246, torch.float32);  permute_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_248: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_850, [0, 1, 3, 2]);  convert_element_type_850 = None
        convert_element_type_851: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_248, torch.bfloat16);  permute_248 = None
        convert_element_type_default_33: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_247, torch.bfloat16);  permute_247 = None
        expand_181: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_33, [1, 16, 128, 256]);  convert_element_type_default_33 = None
        view_631: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_181, [16, 128, 256]);  expand_181 = None
        expand_182: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_851, [1, 16, 256, 128]);  convert_element_type_851 = None
        view_632: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_182, [16, 256, 128]);  expand_182 = None
        bmm_44: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_631, view_632)
        view_633: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_44: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_633, 16.0);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_205: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_44, where);  div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_22: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_205, [-1], True)
        sub_47: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_205, amax_22);  add_205 = amax_22 = None
        exp_22: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
        sum_23: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_45: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_855: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_45, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_183: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_855, [1, 16, 128, 128]);  convert_element_type_855 = None
        view_634: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_183, [16, 128, 128]);  expand_183 = None
        expand_184: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_245, [1, 16, 128, 256]);  permute_245 = None
        view_635: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_184, [16, 128, 256]);  expand_184 = None
        bmm_45: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_634, view_635)
        view_636: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [1, 16, 128, 256]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_249: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_636, [0, 2, 1, 3]);  view_636 = None
        clone_182: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_637: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_182, [1, 128, 4096]);  clone_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_858: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_251, torch.bfloat16);  primals_251 = None
        permute_250: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_858, [1, 0]);  convert_element_type_858 = None
        view_638: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_637, [128, 4096]);  view_637 = None
        mm_91: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_638, permute_250)
        view_639: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [1, 128, 4096]);  mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_861: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_253, torch.bfloat16);  primals_253 = None
        convert_element_type_862: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_252, torch.bfloat16);  primals_252 = None
        permute_251: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_862, [1, 0]);  convert_element_type_862 = None
        addmm_44: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_861, view_616, permute_251);  convert_element_type_861 = None
        view_641: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_226: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_641, 0.5)
        convert_element_type_867: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_641, torch.float32)
        pow_23: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_867, 3.0);  convert_element_type_867 = None
        mul_227: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_206: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_641, mul_227);  view_641 = mul_227 = None
        mul_228: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_206, 0.7978845608028654);  add_206 = None
        tanh_22: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_228);  mul_228 = None
        add_207: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_229: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_226, add_207);  mul_226 = add_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_868: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_255, torch.bfloat16);  primals_255 = None
        convert_element_type_869: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        convert_element_type_870: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_229, torch.bfloat16);  mul_229 = None
        view_642: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_870, [128, 16384]);  convert_element_type_870 = None
        permute_252: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_869, [1, 0]);  convert_element_type_869 = None
        addmm_45: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_868, view_642, permute_252);  convert_element_type_868 = None
        view_643: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [1, 128, 4096]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_208: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_639, view_643);  view_639 = view_643 = None
        add_209: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_208, add_200);  add_208 = add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_209, [2], correction = 0, keepdim = True)
        getitem_92: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_23[0]
        getitem_93: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_210: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_23: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        sub_48: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_209, getitem_93);  getitem_93 = None
        mul_230: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_23);  sub_48 = None
        mul_231: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, primals_256)
        add_211: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_231, primals_257);  mul_231 = primals_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_874: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_258, torch.bfloat16);  primals_258 = None
        convert_element_type_875: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_211, torch.bfloat16);  add_211 = None
        permute_253: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_874, [1, 0]);  convert_element_type_874 = None
        view_644: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_875, [128, 4096]);  convert_element_type_875 = None
        mm_92: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_644, permute_253)
        view_645: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [1, 128, 4096]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_878: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_259, torch.bfloat16);  primals_259 = None
        permute_254: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_878, [1, 0]);  convert_element_type_878 = None
        mm_93: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_644, permute_254)
        view_647: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_93, [1, 128, 4096]);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_882: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        permute_255: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_882, [1, 0]);  convert_element_type_882 = None
        mm_94: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_644, permute_255)
        view_649: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [1, 128, 4096]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_650: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_645, [1, 128, 16, 256]);  view_645 = None
        view_651: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_647, [1, 128, 16, 256]);  view_647 = None
        view_652: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_649, [1, 128, 16, 256]);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_256: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_46: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_261, [1, 1, 1]);  primals_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_23: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_46, 1, repeat_1);  repeat_46 = None
        convert_element_type_886: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_23, torch.bfloat16);  gather_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_23 = torch.ops.aten.split.Tensor(convert_element_type_886, 32, -1);  convert_element_type_886 = None
        getitem_94: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_23[0]
        getitem_95: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_23[1];  split_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_188: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_651, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_189: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_651, 3, 64, 9223372036854775807);  view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_190: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_650, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_191: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_650, 3, 64, 9223372036854775807);  view_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_310: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_94, 2);  getitem_94 = None
        unsqueeze_311: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 4);  unsqueeze_310 = None
        expand_185: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_311, [1, 128, 1, 32, 2])
        clone_185: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_653: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [1, 128, 1, 64]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_312: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_95, 2);  getitem_95 = None
        unsqueeze_313: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 4);  unsqueeze_312 = None
        expand_186: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_313, [1, 128, 1, 32, 2])
        clone_186: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_654: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [1, 128, 1, 64]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_232: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_188, view_654)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_192: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_188, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_193: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_188, 3, 1, 9223372036854775807, 2);  slice_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_46: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_193);  slice_193 = None
        unsqueeze_314: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_46, 4);  neg_46 = None
        unsqueeze_315: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_192, 4);  slice_192 = None
        cat_93: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_314, unsqueeze_315], -1);  unsqueeze_314 = unsqueeze_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_655: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_93, [1, 128, 16, 64]);  cat_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_233: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_655, view_653);  view_655 = None
        add_212: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        mul_234: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_190, view_654);  view_654 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_194: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_190, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_195: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_190, 3, 1, 9223372036854775807, 2);  slice_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_47: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_195);  slice_195 = None
        unsqueeze_320: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_47, 4);  neg_47 = None
        unsqueeze_321: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_194, 4);  slice_194 = None
        cat_94: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_320, unsqueeze_321], -1);  unsqueeze_320 = unsqueeze_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_658: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_94, [1, 128, 16, 64]);  cat_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_235: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_658, view_653);  view_658 = view_653 = None
        add_213: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_95: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_212, slice_189], -1);  add_212 = slice_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_96: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_213, slice_191], -1);  add_213 = slice_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_257: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_95, [0, 2, 1, 3]);  cat_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_258: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_96, [0, 2, 1, 3]);  cat_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_888: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_257, torch.float32);  permute_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_259: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_888, [0, 1, 3, 2]);  convert_element_type_888 = None
        convert_element_type_889: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_259, torch.bfloat16);  permute_259 = None
        convert_element_type_default_32: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_258, torch.bfloat16);  permute_258 = None
        expand_189: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_32, [1, 16, 128, 256]);  convert_element_type_default_32 = None
        view_659: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_189, [16, 128, 256]);  expand_189 = None
        expand_190: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_889, [1, 16, 256, 128]);  convert_element_type_889 = None
        view_660: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_190, [16, 256, 128]);  expand_190 = None
        bmm_46: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_659, view_660)
        view_661: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [1, 16, 128, 128]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_46: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_661, 16.0);  view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_214: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_46, where);  div_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_23: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_214, [-1], True)
        sub_49: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_214, amax_23);  add_214 = amax_23 = None
        exp_23: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_24: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_47: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_893: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_47, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_191: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_893, [1, 16, 128, 128]);  convert_element_type_893 = None
        view_662: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_191, [16, 128, 128]);  expand_191 = None
        expand_192: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_256, [1, 16, 128, 256]);  permute_256 = None
        view_663: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_192, [16, 128, 256]);  expand_192 = None
        bmm_47: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_662, view_663)
        view_664: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [1, 16, 128, 256]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_260: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_664, [0, 2, 1, 3]);  view_664 = None
        clone_190: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_665: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_190, [1, 128, 4096]);  clone_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_896: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_262, torch.bfloat16);  primals_262 = None
        permute_261: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_896, [1, 0]);  convert_element_type_896 = None
        view_666: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_665, [128, 4096]);  view_665 = None
        mm_95: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_666, permute_261)
        view_667: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [1, 128, 4096]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_899: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_264, torch.bfloat16);  primals_264 = None
        convert_element_type_900: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_263, torch.bfloat16);  primals_263 = None
        permute_262: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_900, [1, 0]);  convert_element_type_900 = None
        addmm_46: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_899, view_644, permute_262);  convert_element_type_899 = None
        view_669: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_236: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_669, 0.5)
        convert_element_type_905: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_669, torch.float32)
        pow_24: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_905, 3.0);  convert_element_type_905 = None
        mul_237: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_215: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_669, mul_237);  view_669 = mul_237 = None
        mul_238: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_215, 0.7978845608028654);  add_215 = None
        tanh_23: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
        add_216: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_239: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, add_216);  mul_236 = add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_906: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        convert_element_type_907: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_265, torch.bfloat16);  primals_265 = None
        convert_element_type_908: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_239, torch.bfloat16);  mul_239 = None
        view_670: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_908, [128, 16384]);  convert_element_type_908 = None
        permute_263: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_907, [1, 0]);  convert_element_type_907 = None
        addmm_47: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_906, view_670, permute_263);  convert_element_type_906 = None
        view_671: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [1, 128, 4096]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_217: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_667, view_671);  view_667 = view_671 = None
        add_218: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_217, add_209);  add_217 = add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_218, [2], correction = 0, keepdim = True)
        getitem_96: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_24[0]
        getitem_97: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_219: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_24: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        sub_50: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_218, getitem_97);  getitem_97 = None
        mul_240: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_24);  sub_50 = None
        mul_241: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, primals_267)
        add_220: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_241, primals_268);  mul_241 = primals_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_912: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_269, torch.bfloat16);  primals_269 = None
        convert_element_type_913: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_220, torch.bfloat16);  add_220 = None
        permute_264: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_912, [1, 0]);  convert_element_type_912 = None
        view_672: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_913, [128, 4096]);  convert_element_type_913 = None
        mm_96: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_672, permute_264)
        view_673: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [1, 128, 4096]);  mm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_916: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_270, torch.bfloat16);  primals_270 = None
        permute_265: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_916, [1, 0]);  convert_element_type_916 = None
        mm_97: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_672, permute_265)
        view_675: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_97, [1, 128, 4096]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_920: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_271, torch.bfloat16);  primals_271 = None
        permute_266: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_920, [1, 0]);  convert_element_type_920 = None
        mm_98: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_672, permute_266)
        view_677: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [1, 128, 4096]);  mm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_678: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_673, [1, 128, 16, 256]);  view_673 = None
        view_679: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_675, [1, 128, 16, 256]);  view_675 = None
        view_680: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_677, [1, 128, 16, 256]);  view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_267: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_680, [0, 2, 1, 3]);  view_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_48: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_272, [1, 1, 1]);  primals_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_24: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_48, 1, repeat_1);  repeat_48 = None
        convert_element_type_924: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_24, torch.bfloat16);  gather_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_24 = torch.ops.aten.split.Tensor(convert_element_type_924, 32, -1);  convert_element_type_924 = None
        getitem_98: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_24[0]
        getitem_99: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_24[1];  split_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_196: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_679, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_197: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_679, 3, 64, 9223372036854775807);  view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_198: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_678, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_199: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_678, 3, 64, 9223372036854775807);  view_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_323: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_98, 2);  getitem_98 = None
        unsqueeze_324: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 4);  unsqueeze_323 = None
        expand_193: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_324, [1, 128, 1, 32, 2])
        clone_193: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
        view_681: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_193, [1, 128, 1, 64]);  clone_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_325: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_99, 2);  getitem_99 = None
        unsqueeze_326: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 4);  unsqueeze_325 = None
        expand_194: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_326, [1, 128, 1, 32, 2])
        clone_194: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_194, memory_format = torch.contiguous_format);  expand_194 = None
        view_682: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_194, [1, 128, 1, 64]);  clone_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_242: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_196, view_682)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_200: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_196, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_201: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_196, 3, 1, 9223372036854775807, 2);  slice_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_48: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_201);  slice_201 = None
        unsqueeze_327: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_48, 4);  neg_48 = None
        unsqueeze_328: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_200, 4);  slice_200 = None
        cat_97: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_327, unsqueeze_328], -1);  unsqueeze_327 = unsqueeze_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_683: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_97, [1, 128, 16, 64]);  cat_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_243: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_683, view_681);  view_683 = None
        add_221: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        mul_244: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_198, view_682);  view_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_202: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_198, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_203: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_198, 3, 1, 9223372036854775807, 2);  slice_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_49: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_203);  slice_203 = None
        unsqueeze_333: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_49, 4);  neg_49 = None
        unsqueeze_334: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_202, 4);  slice_202 = None
        cat_98: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_333, unsqueeze_334], -1);  unsqueeze_333 = unsqueeze_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_686: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_98, [1, 128, 16, 64]);  cat_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_245: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_686, view_681);  view_686 = view_681 = None
        add_222: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_99: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_221, slice_197], -1);  add_221 = slice_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_100: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_222, slice_199], -1);  add_222 = slice_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_268: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_99, [0, 2, 1, 3]);  cat_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_269: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_100, [0, 2, 1, 3]);  cat_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_926: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_268, torch.float32);  permute_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_270: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_926, [0, 1, 3, 2]);  convert_element_type_926 = None
        convert_element_type_927: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_270, torch.bfloat16);  permute_270 = None
        convert_element_type_default_31: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_269, torch.bfloat16);  permute_269 = None
        expand_197: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_31, [1, 16, 128, 256]);  convert_element_type_default_31 = None
        view_687: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_197, [16, 128, 256]);  expand_197 = None
        expand_198: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_927, [1, 16, 256, 128]);  convert_element_type_927 = None
        view_688: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_198, [16, 256, 128]);  expand_198 = None
        bmm_48: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_687, view_688)
        view_689: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [1, 16, 128, 128]);  bmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_48: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_689, 16.0);  view_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_223: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_48, where);  div_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_24: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_223, [-1], True)
        sub_51: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_223, amax_24);  add_223 = amax_24 = None
        exp_24: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        sum_25: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_49: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_931: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_49, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_199: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_931, [1, 16, 128, 128]);  convert_element_type_931 = None
        view_690: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_199, [16, 128, 128]);  expand_199 = None
        expand_200: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_267, [1, 16, 128, 256]);  permute_267 = None
        view_691: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_200, [16, 128, 256]);  expand_200 = None
        bmm_49: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_690, view_691)
        view_692: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [1, 16, 128, 256]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_271: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None
        clone_198: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_693: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_198, [1, 128, 4096]);  clone_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_934: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_273, torch.bfloat16);  primals_273 = None
        permute_272: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_934, [1, 0]);  convert_element_type_934 = None
        view_694: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_693, [128, 4096]);  view_693 = None
        mm_99: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_694, permute_272)
        view_695: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_99, [1, 128, 4096]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_937: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_275, torch.bfloat16);  primals_275 = None
        convert_element_type_938: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_274, torch.bfloat16);  primals_274 = None
        permute_273: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_938, [1, 0]);  convert_element_type_938 = None
        addmm_48: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_937, view_672, permute_273);  convert_element_type_937 = None
        view_697: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_246: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_697, 0.5)
        convert_element_type_943: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_697, torch.float32)
        pow_25: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_943, 3.0);  convert_element_type_943 = None
        mul_247: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_224: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_697, mul_247);  view_697 = mul_247 = None
        mul_248: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_224, 0.7978845608028654);  add_224 = None
        tanh_24: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_248);  mul_248 = None
        add_225: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_24, 1.0);  tanh_24 = None
        mul_249: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, add_225);  mul_246 = add_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_944: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_277, torch.bfloat16);  primals_277 = None
        convert_element_type_945: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_276, torch.bfloat16);  primals_276 = None
        convert_element_type_946: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_249, torch.bfloat16);  mul_249 = None
        view_698: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_946, [128, 16384]);  convert_element_type_946 = None
        permute_274: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_945, [1, 0]);  convert_element_type_945 = None
        addmm_49: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_944, view_698, permute_274);  convert_element_type_944 = None
        view_699: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [1, 128, 4096]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_226: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_695, view_699);  view_695 = view_699 = None
        add_227: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_226, add_218);  add_226 = add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_227, [2], correction = 0, keepdim = True)
        getitem_100: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_25[0]
        getitem_101: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_228: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_25: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        sub_52: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_227, getitem_101);  getitem_101 = None
        mul_250: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_25);  sub_52 = None
        mul_251: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, primals_278)
        add_229: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_251, primals_279);  mul_251 = primals_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_950: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_280, torch.bfloat16);  primals_280 = None
        convert_element_type_951: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_229, torch.bfloat16);  add_229 = None
        permute_275: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_950, [1, 0]);  convert_element_type_950 = None
        view_700: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_951, [128, 4096]);  convert_element_type_951 = None
        mm_100: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_700, permute_275)
        view_701: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [1, 128, 4096]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_954: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_281, torch.bfloat16);  primals_281 = None
        permute_276: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_954, [1, 0]);  convert_element_type_954 = None
        mm_101: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_700, permute_276)
        view_703: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_101, [1, 128, 4096]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_958: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_282, torch.bfloat16);  primals_282 = None
        permute_277: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_958, [1, 0]);  convert_element_type_958 = None
        mm_102: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_700, permute_277)
        view_705: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [1, 128, 4096]);  mm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_706: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_701, [1, 128, 16, 256]);  view_701 = None
        view_707: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_703, [1, 128, 16, 256]);  view_703 = None
        view_708: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_705, [1, 128, 16, 256]);  view_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_278: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_708, [0, 2, 1, 3]);  view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_50: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_283, [1, 1, 1]);  primals_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_25: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_50, 1, repeat_1);  repeat_50 = None
        convert_element_type_962: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_25, torch.bfloat16);  gather_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_25 = torch.ops.aten.split.Tensor(convert_element_type_962, 32, -1);  convert_element_type_962 = None
        getitem_102: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_25[0]
        getitem_103: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_25[1];  split_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_204: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_707, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_205: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_707, 3, 64, 9223372036854775807);  view_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_206: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_706, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_207: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_706, 3, 64, 9223372036854775807);  view_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_336: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_102, 2);  getitem_102 = None
        unsqueeze_337: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 4);  unsqueeze_336 = None
        expand_201: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_337, [1, 128, 1, 32, 2])
        clone_201: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
        view_709: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [1, 128, 1, 64]);  clone_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_338: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_103, 2);  getitem_103 = None
        unsqueeze_339: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 4);  unsqueeze_338 = None
        expand_202: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_339, [1, 128, 1, 32, 2])
        clone_202: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_202, memory_format = torch.contiguous_format);  expand_202 = None
        view_710: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_202, [1, 128, 1, 64]);  clone_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_252: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_204, view_710)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_208: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_204, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_209: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_204, 3, 1, 9223372036854775807, 2);  slice_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_50: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_209);  slice_209 = None
        unsqueeze_340: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_50, 4);  neg_50 = None
        unsqueeze_341: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_208, 4);  slice_208 = None
        cat_101: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_340, unsqueeze_341], -1);  unsqueeze_340 = unsqueeze_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_711: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_101, [1, 128, 16, 64]);  cat_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_253: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_711, view_709);  view_711 = None
        add_230: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
        mul_254: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_206, view_710);  view_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_210: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_206, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_211: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_206, 3, 1, 9223372036854775807, 2);  slice_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_51: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_211);  slice_211 = None
        unsqueeze_346: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_51, 4);  neg_51 = None
        unsqueeze_347: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_210, 4);  slice_210 = None
        cat_102: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_346, unsqueeze_347], -1);  unsqueeze_346 = unsqueeze_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_714: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_102, [1, 128, 16, 64]);  cat_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_255: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_714, view_709);  view_714 = view_709 = None
        add_231: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_103: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_230, slice_205], -1);  add_230 = slice_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_104: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_231, slice_207], -1);  add_231 = slice_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_279: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_103, [0, 2, 1, 3]);  cat_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_280: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_104, [0, 2, 1, 3]);  cat_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_964: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_279, torch.float32);  permute_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_281: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_964, [0, 1, 3, 2]);  convert_element_type_964 = None
        convert_element_type_965: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_281, torch.bfloat16);  permute_281 = None
        convert_element_type_default_30: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_280, torch.bfloat16);  permute_280 = None
        expand_205: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_30, [1, 16, 128, 256]);  convert_element_type_default_30 = None
        view_715: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_205, [16, 128, 256]);  expand_205 = None
        expand_206: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_965, [1, 16, 256, 128]);  convert_element_type_965 = None
        view_716: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_206, [16, 256, 128]);  expand_206 = None
        bmm_50: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_715, view_716)
        view_717: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [1, 16, 128, 128]);  bmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_50: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_717, 16.0);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_232: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_50, where);  div_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_25: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_232, [-1], True)
        sub_53: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_232, amax_25);  add_232 = amax_25 = None
        exp_25: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
        sum_26: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_51: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_969: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_51, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_207: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_969, [1, 16, 128, 128]);  convert_element_type_969 = None
        view_718: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_207, [16, 128, 128]);  expand_207 = None
        expand_208: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_278, [1, 16, 128, 256]);  permute_278 = None
        view_719: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_208, [16, 128, 256]);  expand_208 = None
        bmm_51: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_718, view_719)
        view_720: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [1, 16, 128, 256]);  bmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_282: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
        clone_206: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_721: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_206, [1, 128, 4096]);  clone_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_972: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        permute_283: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_972, [1, 0]);  convert_element_type_972 = None
        view_722: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_721, [128, 4096]);  view_721 = None
        mm_103: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_722, permute_283)
        view_723: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_103, [1, 128, 4096]);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_975: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_286, torch.bfloat16);  primals_286 = None
        convert_element_type_976: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_285, torch.bfloat16);  primals_285 = None
        permute_284: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_976, [1, 0]);  convert_element_type_976 = None
        addmm_50: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_975, view_700, permute_284);  convert_element_type_975 = None
        view_725: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_256: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_725, 0.5)
        convert_element_type_981: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_725, torch.float32)
        pow_26: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_981, 3.0);  convert_element_type_981 = None
        mul_257: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_233: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_725, mul_257);  view_725 = mul_257 = None
        mul_258: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, 0.7978845608028654);  add_233 = None
        tanh_25: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_258);  mul_258 = None
        add_234: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_25, 1.0);  tanh_25 = None
        mul_259: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, add_234);  mul_256 = add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_982: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_288, torch.bfloat16);  primals_288 = None
        convert_element_type_983: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_287, torch.bfloat16);  primals_287 = None
        convert_element_type_984: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_259, torch.bfloat16);  mul_259 = None
        view_726: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_984, [128, 16384]);  convert_element_type_984 = None
        permute_285: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_983, [1, 0]);  convert_element_type_983 = None
        addmm_51: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_982, view_726, permute_285);  convert_element_type_982 = None
        view_727: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [1, 128, 4096]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_235: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_723, view_727);  view_723 = view_727 = None
        add_236: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_235, add_227);  add_235 = add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_236, [2], correction = 0, keepdim = True)
        getitem_104: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_26[0]
        getitem_105: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_237: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_26: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        sub_54: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_236, getitem_105);  getitem_105 = None
        mul_260: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_26);  sub_54 = None
        mul_261: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_260, primals_289)
        add_238: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_261, primals_290);  mul_261 = primals_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_988: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_291, torch.bfloat16);  primals_291 = None
        convert_element_type_989: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_238, torch.bfloat16);  add_238 = None
        permute_286: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_988, [1, 0]);  convert_element_type_988 = None
        view_728: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_989, [128, 4096]);  convert_element_type_989 = None
        mm_104: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_728, permute_286)
        view_729: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [1, 128, 4096]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_992: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_292, torch.bfloat16);  primals_292 = None
        permute_287: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_992, [1, 0]);  convert_element_type_992 = None
        mm_105: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_728, permute_287)
        view_731: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_105, [1, 128, 4096]);  mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_996: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_293, torch.bfloat16);  primals_293 = None
        permute_288: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_996, [1, 0]);  convert_element_type_996 = None
        mm_106: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_728, permute_288)
        view_733: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [1, 128, 4096]);  mm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_734: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_729, [1, 128, 16, 256]);  view_729 = None
        view_735: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_731, [1, 128, 16, 256]);  view_731 = None
        view_736: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_733, [1, 128, 16, 256]);  view_733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_289: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_736, [0, 2, 1, 3]);  view_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_52: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_294, [1, 1, 1]);  primals_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_26: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_52, 1, repeat_1);  repeat_52 = None
        convert_element_type_1000: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_26, torch.bfloat16);  gather_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_26 = torch.ops.aten.split.Tensor(convert_element_type_1000, 32, -1);  convert_element_type_1000 = None
        getitem_106: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_26[0]
        getitem_107: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_26[1];  split_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_212: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_735, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_213: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_735, 3, 64, 9223372036854775807);  view_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_214: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_734, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_215: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_734, 3, 64, 9223372036854775807);  view_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_349: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_106, 2);  getitem_106 = None
        unsqueeze_350: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 4);  unsqueeze_349 = None
        expand_209: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_350, [1, 128, 1, 32, 2])
        clone_209: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
        view_737: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_209, [1, 128, 1, 64]);  clone_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_351: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_107, 2);  getitem_107 = None
        unsqueeze_352: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 4);  unsqueeze_351 = None
        expand_210: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_352, [1, 128, 1, 32, 2])
        clone_210: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_210, memory_format = torch.contiguous_format);  expand_210 = None
        view_738: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_210, [1, 128, 1, 64]);  clone_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_262: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_212, view_738)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_216: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_212, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_217: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_212, 3, 1, 9223372036854775807, 2);  slice_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_52: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_217);  slice_217 = None
        unsqueeze_353: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_52, 4);  neg_52 = None
        unsqueeze_354: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_216, 4);  slice_216 = None
        cat_105: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_353, unsqueeze_354], -1);  unsqueeze_353 = unsqueeze_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_739: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_105, [1, 128, 16, 64]);  cat_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_263: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_739, view_737);  view_739 = None
        add_239: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
        mul_264: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_214, view_738);  view_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_218: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_214, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_219: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_214, 3, 1, 9223372036854775807, 2);  slice_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_53: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_219);  slice_219 = None
        unsqueeze_359: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_53, 4);  neg_53 = None
        unsqueeze_360: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_218, 4);  slice_218 = None
        cat_106: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_359, unsqueeze_360], -1);  unsqueeze_359 = unsqueeze_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_742: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_106, [1, 128, 16, 64]);  cat_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_265: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_742, view_737);  view_742 = view_737 = None
        add_240: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_107: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_239, slice_213], -1);  add_239 = slice_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_108: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_240, slice_215], -1);  add_240 = slice_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_290: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_107, [0, 2, 1, 3]);  cat_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_291: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_108, [0, 2, 1, 3]);  cat_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1002: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_290, torch.float32);  permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_292: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1002, [0, 1, 3, 2]);  convert_element_type_1002 = None
        convert_element_type_1003: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_292, torch.bfloat16);  permute_292 = None
        convert_element_type_default_29: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_291, torch.bfloat16);  permute_291 = None
        expand_213: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_29, [1, 16, 128, 256]);  convert_element_type_default_29 = None
        view_743: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_213, [16, 128, 256]);  expand_213 = None
        expand_214: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_1003, [1, 16, 256, 128]);  convert_element_type_1003 = None
        view_744: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_214, [16, 256, 128]);  expand_214 = None
        bmm_52: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_743, view_744)
        view_745: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [1, 16, 128, 128]);  bmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_52: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_745, 16.0);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_241: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_52, where);  div_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_26: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_241, [-1], True)
        sub_55: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_241, amax_26);  add_241 = amax_26 = None
        exp_26: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_27: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_53: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1007: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_53, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_215: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_1007, [1, 16, 128, 128]);  convert_element_type_1007 = None
        view_746: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_215, [16, 128, 128]);  expand_215 = None
        expand_216: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_289, [1, 16, 128, 256]);  permute_289 = None
        view_747: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_216, [16, 128, 256]);  expand_216 = None
        bmm_53: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_746, view_747)
        view_748: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [1, 16, 128, 256]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_293: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None
        clone_214: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_749: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_214, [1, 128, 4096]);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_1010: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_295, torch.bfloat16);  primals_295 = None
        permute_294: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1010, [1, 0]);  convert_element_type_1010 = None
        view_750: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_749, [128, 4096]);  view_749 = None
        mm_107: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_750, permute_294)
        view_751: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_107, [1, 128, 4096]);  mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_1013: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_297, torch.bfloat16);  primals_297 = None
        convert_element_type_1014: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_296, torch.bfloat16);  primals_296 = None
        permute_295: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1014, [1, 0]);  convert_element_type_1014 = None
        addmm_52: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_1013, view_728, permute_295);  convert_element_type_1013 = None
        view_753: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_266: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_753, 0.5)
        convert_element_type_1019: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_753, torch.float32)
        pow_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1019, 3.0);  convert_element_type_1019 = None
        mul_267: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_242: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_753, mul_267);  view_753 = mul_267 = None
        mul_268: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_242, 0.7978845608028654);  add_242 = None
        tanh_26: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_268);  mul_268 = None
        add_243: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_26, 1.0);  tanh_26 = None
        mul_269: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, add_243);  mul_266 = add_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_1020: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_299, torch.bfloat16);  primals_299 = None
        convert_element_type_1021: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_298, torch.bfloat16);  primals_298 = None
        convert_element_type_1022: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        view_754: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1022, [128, 16384]);  convert_element_type_1022 = None
        permute_296: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1021, [1, 0]);  convert_element_type_1021 = None
        addmm_53: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_1020, view_754, permute_296);  convert_element_type_1020 = None
        view_755: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [1, 128, 4096]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_244: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_751, view_755);  view_751 = view_755 = None
        add_245: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_244, add_236);  add_244 = add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_27 = torch.ops.aten.var_mean.correction(add_245, [2], correction = 0, keepdim = True)
        getitem_108: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_27[0]
        getitem_109: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_246: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_27: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        sub_56: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_245, getitem_109);  getitem_109 = None
        mul_270: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_27);  sub_56 = None
        mul_271: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, primals_300)
        add_247: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_271, primals_301);  mul_271 = primals_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        convert_element_type_1026: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_302, torch.bfloat16);  primals_302 = None
        convert_element_type_1027: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_247, torch.bfloat16);  add_247 = None
        permute_297: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1026, [1, 0]);  convert_element_type_1026 = None
        view_756: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1027, [128, 4096]);  convert_element_type_1027 = None
        mm_108: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_756, permute_297)
        view_757: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [1, 128, 4096]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        convert_element_type_1030: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_303, torch.bfloat16);  primals_303 = None
        permute_298: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1030, [1, 0]);  convert_element_type_1030 = None
        mm_109: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_756, permute_298)
        view_759: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_109, [1, 128, 4096]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        convert_element_type_1034: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_304, torch.bfloat16);  primals_304 = None
        permute_299: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1034, [1, 0]);  convert_element_type_1034 = None
        mm_110: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_756, permute_299)
        view_761: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [1, 128, 4096]);  mm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        view_762: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_757, [1, 128, 16, 256]);  view_757 = None
        view_763: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_759, [1, 128, 16, 256]);  view_759 = None
        view_764: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_761, [1, 128, 16, 256]);  view_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_300: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_764, [0, 2, 1, 3]);  view_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_54: "f32[1, 2048, 64][131072, 64, 1]cuda:0" = torch.ops.aten.repeat.default(primals_305, [1, 1, 1]);  primals_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_27: "f32[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.gather.default(repeat_54, 1, repeat_1);  repeat_54 = repeat_1 = None
        convert_element_type_1038: "bf16[1, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gather_27, torch.bfloat16);  gather_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_27 = torch.ops.aten.split.Tensor(convert_element_type_1038, 32, -1);  convert_element_type_1038 = None
        getitem_110: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_27[0]
        getitem_111: "bf16[1, 128, 32][8192, 64, 1]cuda:0" = split_27[1];  split_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_220: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_763, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_221: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_763, 3, 64, 9223372036854775807);  view_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_222: "bf16[1, 128, 16, 64][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_762, 3, 0, 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_223: "bf16[1, 128, 16, 192][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_762, 3, 64, 9223372036854775807);  view_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        unsqueeze_362: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_110, 2);  getitem_110 = None
        unsqueeze_363: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 4);  unsqueeze_362 = None
        expand_217: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_363, [1, 128, 1, 32, 2])
        clone_217: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
        view_765: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_217, [1, 128, 1, 64]);  clone_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        unsqueeze_364: "bf16[1, 128, 1, 32][8192, 64, 32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(getitem_111, 2);  getitem_111 = None
        unsqueeze_365: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 4);  unsqueeze_364 = None
        expand_218: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_365, [1, 128, 1, 32, 2])
        clone_218: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_218, memory_format = torch.contiguous_format);  expand_218 = None
        view_766: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_218, [1, 128, 1, 64]);  clone_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_272: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_220, view_766)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_224: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_220, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_225: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_220, 3, 1, 9223372036854775807, 2);  slice_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_54: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_225);  slice_225 = None
        unsqueeze_366: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_54, 4);  neg_54 = None
        unsqueeze_367: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_224, 4);  slice_224 = None
        cat_109: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_366, unsqueeze_367], -1);  unsqueeze_366 = unsqueeze_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_767: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_109, [1, 128, 16, 64]);  cat_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_273: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_767, view_765);  view_767 = None
        add_248: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
        mul_274: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_222, view_766);  view_766 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_226: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_222, 3, 0, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_227: "bf16[1, 128, 16, 32][524288, 4096, 256, 2]cuda:0" = torch.ops.aten.slice.Tensor(slice_222, 3, 1, 9223372036854775807, 2);  slice_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        neg_55: "bf16[1, 128, 16, 32][65536, 512, 32, 1]cuda:0" = torch.ops.aten.neg.default(slice_227);  slice_227 = None
        unsqueeze_372: "bf16[1, 128, 16, 32, 1][65536, 512, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(neg_55, 4);  neg_55 = None
        unsqueeze_373: "bf16[1, 128, 16, 32, 1][524288, 4096, 256, 2, 1]cuda:0" = torch.ops.aten.unsqueeze.default(slice_226, 4);  slice_226 = None
        cat_110: "bf16[1, 128, 16, 32, 2][131072, 1024, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_372, unsqueeze_373], -1);  unsqueeze_372 = unsqueeze_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_770: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_110, [1, 128, 16, 64]);  cat_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_275: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_770, view_765);  view_770 = view_765 = None
        add_249: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        cat_111: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_248, slice_221], -1);  add_248 = slice_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        cat_112: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([add_249, slice_223], -1);  add_249 = slice_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_301: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_111, [0, 2, 1, 3]);  cat_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_302: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(cat_112, [0, 2, 1, 3]);  cat_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1040: "f32[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_301, torch.float32);  permute_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_303: "f32[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1040, [0, 1, 3, 2]);  convert_element_type_1040 = None
        convert_element_type_1041: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.prims.convert_element_type.default(permute_303, torch.bfloat16);  permute_303 = None
        convert_element_type_default_28: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_302, torch.bfloat16);  permute_302 = None
        expand_221: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_default_28, [1, 16, 128, 256]);  convert_element_type_default_28 = None
        view_771: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_221, [16, 128, 256]);  expand_221 = None
        expand_222: "bf16[1, 16, 256, 128][524288, 256, 1, 4096]cuda:0" = torch.ops.aten.expand.default(convert_element_type_1041, [1, 16, 256, 128]);  convert_element_type_1041 = None
        view_772: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.reshape.default(expand_222, [16, 256, 128]);  expand_222 = None
        bmm_54: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_771, view_772)
        view_773: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [1, 16, 128, 128]);  bmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_54: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_773, 16.0);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_250: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(div_54, where);  div_54 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_27: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_250, [-1], True)
        sub_57: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_250, amax_27);  add_250 = amax_27 = None
        exp_27: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_28: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_55: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1045: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_55, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_223: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_1045, [1, 16, 128, 128]);  convert_element_type_1045 = None
        view_774: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_223, [16, 128, 128]);  expand_223 = None
        expand_224: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.expand.default(permute_300, [1, 16, 128, 256]);  permute_300 = None
        view_775: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(expand_224, [16, 128, 256]);  expand_224 = None
        bmm_55: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_774, view_775)
        view_776: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [1, 16, 128, 256]);  bmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_304: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_776, [0, 2, 1, 3]);  view_776 = None
        clone_222: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_777: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_222, [1, 128, 4096]);  clone_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        convert_element_type_1048: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_306, torch.bfloat16);  primals_306 = None
        permute_305: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1048, [1, 0]);  convert_element_type_1048 = None
        view_778: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_777, [128, 4096]);  view_777 = None
        mm_111: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_778, permute_305)
        view_779: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_111, [1, 128, 4096]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        convert_element_type_1051: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_308, torch.bfloat16);  primals_308 = None
        convert_element_type_1052: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_307, torch.bfloat16);  primals_307 = None
        permute_306: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1052, [1, 0]);  convert_element_type_1052 = None
        addmm_54: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_1051, view_756, permute_306);  convert_element_type_1051 = None
        view_781: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 16384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_276: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_781, 0.5)
        convert_element_type_1057: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_781, torch.float32)
        pow_28: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1057, 3.0);  convert_element_type_1057 = None
        mul_277: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
        add_251: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_781, mul_277);  view_781 = mul_277 = None
        mul_278: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_251, 0.7978845608028654);  add_251 = None
        tanh_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_252: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_27, 1.0);  tanh_27 = None
        mul_279: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, add_252);  mul_276 = add_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        convert_element_type_1058: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_310, torch.bfloat16);  primals_310 = None
        convert_element_type_1059: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_309, torch.bfloat16);  primals_309 = None
        convert_element_type_1060: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_279, torch.bfloat16);  mul_279 = None
        view_782: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1060, [128, 16384]);  convert_element_type_1060 = None
        permute_307: "bf16[16384, 4096][1, 16384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1059, [1, 0]);  convert_element_type_1059 = None
        addmm_55: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_1058, view_782, permute_307);  convert_element_type_1058 = None
        view_783: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [1, 128, 4096]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_253: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_779, view_783);  view_779 = view_783 = None
        add_254: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_253, add_245);  add_253 = add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_28 = torch.ops.aten.var_mean.correction(add_254, [2], correction = 0, keepdim = True)
        getitem_112: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_28[0]
        getitem_113: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_255: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_28: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_58: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_254, getitem_113);  add_254 = getitem_113 = None
        mul_280: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_28);  sub_58 = None
        mul_281: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, primals_311)
        add_256: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_281, primals_312);  mul_281 = primals_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:623 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        convert_element_type_1064: "bf16[50400][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_314, torch.bfloat16);  primals_314 = None
        convert_element_type_1065: "bf16[50400, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_313, torch.bfloat16);  primals_313 = None
        convert_element_type_1066: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_256, torch.bfloat16);  add_256 = None
        view_785: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1066, [128, 4096]);  convert_element_type_1066 = None
        permute_308: "bf16[4096, 50400][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1065, [1, 0]);  convert_element_type_1065 = None
        addmm_56: "bf16[128, 50400][50400, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_1064, view_785, permute_308);  convert_element_type_1064 = None
        view_786: "bf16[1, 128, 50400][6451200, 50400, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [1, 128, 50400]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_1070: "f32[1, 128, 50400][6451200, 50400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_786, torch.float32)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[1, 129][129, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(primals_315, [0, 1], -100.0);  primals_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_228: "i64[1, 128][129, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_787: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1070, [-1, 50400]);  convert_element_type_1070 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_788: "i64[128][1]cuda:0" = torch.ops.aten.reshape.default(slice_228, [-1]);  slice_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_28: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(view_787, [1], True)
        sub_59: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_787, amax_28);  view_787 = None
        exp_28: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.exp.default(sub_59)
        sum_29: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_28, [1], True);  exp_28 = None
        log: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_29);  sum_29 = None
        sub_60: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_59, log);  sub_59 = None
        ne_1: "b8[128][1]cuda:0" = torch.ops.aten.ne.Scalar(view_788, -100)
        full_default_2: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[128][1]cuda:0" = torch.ops.aten.where.self(ne_1, view_788, full_default_2);  view_788 = full_default_2 = None
        unsqueeze_374: "i64[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather_28: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(sub_60, 1, unsqueeze_374);  sub_60 = unsqueeze_374 = None
        squeeze: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dim(gather_28, 1);  gather_28 = None
        neg_56: "f32[128][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_2: "f32[128][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg_56, full_default);  neg_56 = full_default = None
        sum_30: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_1);  ne_1 = None
        convert_element_type_1071: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_30, torch.float32);  sum_30 = None
        sum_31: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_2);  where_2 = None
        div_56: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_31, convert_element_type_1071);  sum_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:623 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_309: "bf16[50400, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        div_58: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_28, 4096);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_313: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_317: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_323: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_326: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_774, [0, 2, 1]);  view_774 = None
        permute_327: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_775, [0, 2, 1]);  view_775 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_328: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_771, [0, 2, 1]);  view_771 = None
        permute_329: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_772, [0, 2, 1]);  view_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_336: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_340: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_344: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_60: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_27, 4096);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_346: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_350: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_356: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_359: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_746, [0, 2, 1]);  view_746 = None
        permute_360: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_747, [0, 2, 1]);  view_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_361: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_743, [0, 2, 1]);  view_743 = None
        permute_362: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_744, [0, 2, 1]);  view_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_369: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_373: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_377: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_62: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_26, 4096);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_379: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_383: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_389: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_392: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_718, [0, 2, 1]);  view_718 = None
        permute_393: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_719, [0, 2, 1]);  view_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_394: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_715, [0, 2, 1]);  view_715 = None
        permute_395: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_716, [0, 2, 1]);  view_716 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_402: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_406: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_410: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_64: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_25, 4096);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_412: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_416: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_422: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_425: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_690, [0, 2, 1]);  view_690 = None
        permute_426: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_691, [0, 2, 1]);  view_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_427: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_687, [0, 2, 1]);  view_687 = None
        permute_428: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_688, [0, 2, 1]);  view_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_435: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_439: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_443: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_66: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 4096);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_445: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_449: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_455: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_458: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_662, [0, 2, 1]);  view_662 = None
        permute_459: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_663, [0, 2, 1]);  view_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_460: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_659, [0, 2, 1]);  view_659 = None
        permute_461: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_660, [0, 2, 1]);  view_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_468: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_472: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_476: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_68: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 4096);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_478: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_482: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_488: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_491: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_634, [0, 2, 1]);  view_634 = None
        permute_492: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_635, [0, 2, 1]);  view_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_493: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_631, [0, 2, 1]);  view_631 = None
        permute_494: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_632, [0, 2, 1]);  view_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_501: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_505: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_509: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_70: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 4096);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_511: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_515: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_521: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_524: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_606, [0, 2, 1]);  view_606 = None
        permute_525: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_607, [0, 2, 1]);  view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_526: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_603, [0, 2, 1]);  view_603 = None
        permute_527: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_604, [0, 2, 1]);  view_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_534: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_538: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_542: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_72: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 4096);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_544: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_548: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_554: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_557: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_578, [0, 2, 1]);  view_578 = None
        permute_558: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_579, [0, 2, 1]);  view_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_559: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_575, [0, 2, 1]);  view_575 = None
        permute_560: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_576, [0, 2, 1]);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_567: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_571: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_575: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_74: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 4096);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_577: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_581: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_587: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_590: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_550, [0, 2, 1]);  view_550 = None
        permute_591: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_551, [0, 2, 1]);  view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_592: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_547, [0, 2, 1]);  view_547 = None
        permute_593: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_548, [0, 2, 1]);  view_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_600: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_604: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_608: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_76: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 4096);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_610: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_614: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_620: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_623: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_522, [0, 2, 1]);  view_522 = None
        permute_624: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_523, [0, 2, 1]);  view_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_625: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None
        permute_626: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_520, [0, 2, 1]);  view_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_633: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_637: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_641: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_78: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 4096);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_643: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_647: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_653: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_656: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
        permute_657: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_658: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_491, [0, 2, 1]);  view_491 = None
        permute_659: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_492, [0, 2, 1]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_666: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_670: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_674: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_80: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 4096);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_676: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_680: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_686: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_689: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 2, 1]);  view_466 = None
        permute_690: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_691: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_463, [0, 2, 1]);  view_463 = None
        permute_692: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_464, [0, 2, 1]);  view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_699: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_703: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_707: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_82: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 4096);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_709: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_713: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_719: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_722: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_438, [0, 2, 1]);  view_438 = None
        permute_723: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_724: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_435, [0, 2, 1]);  view_435 = None
        permute_725: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_436, [0, 2, 1]);  view_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_732: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_736: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_740: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_84: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 4096);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_742: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_746: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_752: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_755: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
        permute_756: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_757: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None
        permute_758: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_765: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_769: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_773: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_86: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 4096);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_775: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_779: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_785: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_788: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
        permute_789: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_790: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_379, [0, 2, 1]);  view_379 = None
        permute_791: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_798: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_802: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_806: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_88: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 4096);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_808: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_812: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_818: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_821: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_354, [0, 2, 1]);  view_354 = None
        permute_822: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_355, [0, 2, 1]);  view_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_823: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
        permute_824: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_831: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_835: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_839: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_90: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 4096);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_841: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_845: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_851: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_854: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_326, [0, 2, 1]);  view_326 = None
        permute_855: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_327, [0, 2, 1]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_856: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
        permute_857: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_864: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_868: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_872: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_92: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 4096);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_874: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_878: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_884: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_887: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
        permute_888: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_889: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
        permute_890: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_897: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_901: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_905: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_94: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 4096);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_907: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_911: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_917: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_920: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
        permute_921: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_922: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        permute_923: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_930: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_934: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_938: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_96: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 4096);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_940: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_944: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_950: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_953: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None
        permute_954: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_955: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
        permute_956: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_963: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_967: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_971: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_98: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 4096);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_973: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_977: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_983: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_986: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
        permute_987: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_215, [0, 2, 1]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_988: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        permute_989: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_996: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_1000: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_1004: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_100: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 4096);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_1006: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_1010: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_1016: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_1019: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
        permute_1020: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_1021: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        permute_1022: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_1029: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_1033: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_1037: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_102: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 4096);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_1039: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_1043: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_1049: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_1052: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
        permute_1053: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_1054: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
        permute_1055: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_1062: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_1066: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_1070: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_104: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 4096);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_1072: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_1076: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_1082: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_1085: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_1086: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_1087: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_1088: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_1095: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_1099: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_1103: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_106: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 4096);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_1105: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_1109: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_1115: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_1118: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
        permute_1119: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_1120: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        permute_1121: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_1128: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_1132: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_1136: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_108: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 4096);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_1138: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_1142: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_1148: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_1151: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_74, [0, 2, 1]);  view_74 = None
        permute_1152: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_1153: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
        permute_1154: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_1161: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_1165: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_1169: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_110: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 4096);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_1171: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_1175: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_1181: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_1184: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
        permute_1185: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_1186: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        permute_1187: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_1194: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_1198: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_1202: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_1204: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_1208: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_1214: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_1217: "bf16[16, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
        permute_1218: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_19, [0, 2, 1]);  view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_1219: "bf16[16, 256, 128][256, 1, 4096]cuda:0" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_1220: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_1227: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_1231: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_1235: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (div_56, view_786, primals_1, primals_3, primals_14, primals_25, primals_36, primals_47, primals_58, primals_69, primals_80, primals_91, primals_102, primals_113, primals_124, primals_135, primals_146, primals_157, primals_168, primals_179, primals_190, primals_201, primals_212, primals_223, primals_234, primals_245, primals_256, primals_267, primals_278, primals_289, primals_300, primals_311, embedding, getitem_1, rsqrt, view, unsqueeze_12, unsqueeze_14, div_1, view_22, addmm, view_26, add_10, getitem_5, rsqrt_1, view_28, unsqueeze_25, unsqueeze_27, div_3, view_50, addmm_2, view_54, mul_20, view_56, unsqueeze_38, unsqueeze_40, div_5, view_78, addmm_4, view_82, mul_30, view_84, unsqueeze_51, unsqueeze_53, div_7, view_106, addmm_6, view_110, mul_40, view_112, unsqueeze_64, unsqueeze_66, div_9, view_134, addmm_8, view_138, mul_50, view_140, unsqueeze_77, unsqueeze_79, div_11, view_162, addmm_10, view_166, mul_60, view_168, unsqueeze_90, unsqueeze_92, div_13, view_190, addmm_12, view_194, mul_70, view_196, unsqueeze_103, unsqueeze_105, div_15, view_218, addmm_14, view_222, mul_80, view_224, unsqueeze_116, unsqueeze_118, div_17, view_246, addmm_16, view_250, mul_90, view_252, unsqueeze_129, unsqueeze_131, div_19, view_274, addmm_18, view_278, mul_100, view_280, unsqueeze_142, unsqueeze_144, div_21, view_302, addmm_20, view_306, mul_110, view_308, unsqueeze_155, unsqueeze_157, div_23, view_330, addmm_22, view_334, mul_120, view_336, unsqueeze_168, unsqueeze_170, div_25, view_358, addmm_24, view_362, mul_130, view_364, unsqueeze_181, unsqueeze_183, div_27, view_386, addmm_26, view_390, mul_140, view_392, unsqueeze_194, unsqueeze_196, div_29, view_414, addmm_28, view_418, mul_150, view_420, unsqueeze_207, unsqueeze_209, div_31, view_442, addmm_30, view_446, mul_160, view_448, unsqueeze_220, unsqueeze_222, div_33, view_470, addmm_32, view_474, mul_170, view_476, unsqueeze_233, unsqueeze_235, div_35, view_498, addmm_34, view_502, mul_180, view_504, unsqueeze_246, unsqueeze_248, div_37, view_526, addmm_36, view_530, mul_190, view_532, unsqueeze_259, unsqueeze_261, div_39, view_554, addmm_38, view_558, mul_200, view_560, unsqueeze_272, unsqueeze_274, div_41, view_582, addmm_40, view_586, mul_210, view_588, unsqueeze_285, unsqueeze_287, div_43, view_610, addmm_42, view_614, mul_220, view_616, unsqueeze_298, unsqueeze_300, div_45, view_638, addmm_44, view_642, mul_230, view_644, unsqueeze_311, unsqueeze_313, div_47, view_666, addmm_46, view_670, mul_240, view_672, unsqueeze_324, unsqueeze_326, div_49, view_694, addmm_48, view_698, mul_250, view_700, unsqueeze_337, unsqueeze_339, div_51, view_722, addmm_50, view_726, mul_260, view_728, unsqueeze_350, unsqueeze_352, div_53, view_750, addmm_52, view_754, mul_270, view_756, unsqueeze_363, unsqueeze_365, div_55, view_778, addmm_54, view_782, mul_280, view_785, view_786, constant_pad_nd, amax_28, log, convert_element_type_1071, permute_309, div_58, permute_313, permute_317, permute_323, permute_326, permute_327, permute_328, permute_329, permute_336, permute_340, permute_344, div_60, permute_346, permute_350, permute_356, permute_359, permute_360, permute_361, permute_362, permute_369, permute_373, permute_377, div_62, permute_379, permute_383, permute_389, permute_392, permute_393, permute_394, permute_395, permute_402, permute_406, permute_410, div_64, permute_412, permute_416, permute_422, permute_425, permute_426, permute_427, permute_428, permute_435, permute_439, permute_443, div_66, permute_445, permute_449, permute_455, permute_458, permute_459, permute_460, permute_461, permute_468, permute_472, permute_476, div_68, permute_478, permute_482, permute_488, permute_491, permute_492, permute_493, permute_494, permute_501, permute_505, permute_509, div_70, permute_511, permute_515, permute_521, permute_524, permute_525, permute_526, permute_527, permute_534, permute_538, permute_542, div_72, permute_544, permute_548, permute_554, permute_557, permute_558, permute_559, permute_560, permute_567, permute_571, permute_575, div_74, permute_577, permute_581, permute_587, permute_590, permute_591, permute_592, permute_593, permute_600, permute_604, permute_608, div_76, permute_610, permute_614, permute_620, permute_623, permute_624, permute_625, permute_626, permute_633, permute_637, permute_641, div_78, permute_643, permute_647, permute_653, permute_656, permute_657, permute_658, permute_659, permute_666, permute_670, permute_674, div_80, permute_676, permute_680, permute_686, permute_689, permute_690, permute_691, permute_692, permute_699, permute_703, permute_707, div_82, permute_709, permute_713, permute_719, permute_722, permute_723, permute_724, permute_725, permute_732, permute_736, permute_740, div_84, permute_742, permute_746, permute_752, permute_755, permute_756, permute_757, permute_758, permute_765, permute_769, permute_773, div_86, permute_775, permute_779, permute_785, permute_788, permute_789, permute_790, permute_791, permute_798, permute_802, permute_806, div_88, permute_808, permute_812, permute_818, permute_821, permute_822, permute_823, permute_824, permute_831, permute_835, permute_839, div_90, permute_841, permute_845, permute_851, permute_854, permute_855, permute_856, permute_857, permute_864, permute_868, permute_872, div_92, permute_874, permute_878, permute_884, permute_887, permute_888, permute_889, permute_890, permute_897, permute_901, permute_905, div_94, permute_907, permute_911, permute_917, permute_920, permute_921, permute_922, permute_923, permute_930, permute_934, permute_938, div_96, permute_940, permute_944, permute_950, permute_953, permute_954, permute_955, permute_956, permute_963, permute_967, permute_971, div_98, permute_973, permute_977, permute_983, permute_986, permute_987, permute_988, permute_989, permute_996, permute_1000, permute_1004, div_100, permute_1006, permute_1010, permute_1016, permute_1019, permute_1020, permute_1021, permute_1022, permute_1029, permute_1033, permute_1037, div_102, permute_1039, permute_1043, permute_1049, permute_1052, permute_1053, permute_1054, permute_1055, permute_1062, permute_1066, permute_1070, div_104, permute_1072, permute_1076, permute_1082, permute_1085, permute_1086, permute_1087, permute_1088, permute_1095, permute_1099, permute_1103, div_106, permute_1105, permute_1109, permute_1115, permute_1118, permute_1119, permute_1120, permute_1121, permute_1128, permute_1132, permute_1136, div_108, permute_1138, permute_1142, permute_1148, permute_1151, permute_1152, permute_1153, permute_1154, permute_1161, permute_1165, permute_1169, div_110, permute_1171, permute_1175, permute_1181, permute_1184, permute_1185, permute_1186, permute_1187, permute_1194, permute_1198, permute_1202, permute_1204, permute_1208, permute_1214, permute_1217, permute_1218, permute_1219, permute_1220, permute_1227, permute_1231, permute_1235)
